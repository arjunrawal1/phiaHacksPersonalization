from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import uuid
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.core.config import Settings
from app.services import db, phia

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


TAG = "[pipeline]"
REK_CONFIDENCE_THRESHOLD = 90.0
REK_SIMILARITY_THRESHOLD = 80.0
RANK_MAX_CANDIDATES = 5
AUTO_PICK_FREQUENCY_WEIGHT = 0.65
AUTO_PICK_ALIGNMENT_WEIGHT = 0.35
AUTO_PICK_FAST_FREQUENCY_THRESHOLD = 0.67
AUTO_PICK_FAST_MARGIN_THRESHOLD = 0.25
AUTO_PICK_MAX_REFERENCE_CHECKS = 3
ONLINE_FACE_DETECT_CONFIDENCE_MIN = 85.0
ONLINE_SEARCH_TIMEOUT_SECONDS = 20
ONLINE_IMAGE_TIMEOUT_SECONDS = 8
ONLINE_IMAGE_MAX_BYTES = 8 * 1024 * 1024

YOLO_WORLD_CLASS_NAMES: list[str] = [
    "shirt",
    "polo shirt",
    "t-shirt",
    "top",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "dress",
    "pants",
    "trousers",
    "chinos",
    "jeans",
    "shorts",
    "skirt",
    "shoes",
    "sneakers",
    "boots",
    "hat",
    "cap",
    "bag",
    "backpack",
    "handbag",
    "belt",
    "watch",
    "bracelet",
    "necklace",
    "tie",
    "sunglasses",
]

_RUNNING_FACE_JOBS: set[str] = set()
_RUNNING_CLOTHING_RUNS: set[str] = set()
_RUNNING_LOOKUPS: set[str] = set()
_RUNNING_STYLING_AUTO_RUNS: set[str] = set()
_RUN_LOCK = threading.Lock()
_SUPABASE_UPLOAD_LOCK = threading.Lock()
_UPLOADED_SUPABASE_MEDIA: set[str] = set()
_LOOKUP_SEMAPHORE: threading.Semaphore | None = None
_SETTINGS: Settings | None = None
_REPLICATE_YOLO_VERSION: str | None = None
_REPLICATE_VERSION_LOCK = threading.Lock()


def configure(settings: Settings) -> None:
    global _SETTINGS, _LOOKUP_SEMAPHORE
    _SETTINGS = settings
    _LOOKUP_SEMAPHORE = threading.Semaphore(max(1, settings.lookup_concurrency))
    (settings.media_dir / "photos").mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "face_tiles").mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "clothing_crops").mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "styling_person_crops").mkdir(parents=True, exist_ok=True)


def _settings() -> Settings:
    if _SETTINGS is None:
        raise RuntimeError("Pipeline is not configured")
    return _SETTINGS


def media_rel_to_url(media_base_url: str, relative_path: str) -> str:
    safe = relative_path.lstrip("/").replace("\\", "/")
    return f"{media_base_url.rstrip('/')}/{safe}"


def _supabase_enabled(settings: Settings) -> bool:
    return bool(settings.supabase_url.strip() and settings.supabase_service_role_key.strip())


def _supabase_bucket_and_key(settings: Settings, relative_path: str) -> tuple[str, str] | None:
    safe = relative_path.lstrip("/").replace("\\", "/")
    if safe.startswith("photos/"):
        return settings.supabase_photos_bucket, safe[len("photos/") :]
    if safe.startswith("face_tiles/"):
        return settings.supabase_face_tiles_bucket, safe[len("face_tiles/") :]
    if safe.startswith("clothing_crops/"):
        return settings.supabase_clothing_crops_bucket, safe[len("clothing_crops/") :]
    return None


def _supabase_public_url(settings: Settings, bucket: str, key: str) -> str:
    base = settings.supabase_url.strip().rstrip("/")
    return f"{base}/storage/v1/object/public/{bucket}/{key}"


def _upload_media_to_supabase(relative_path: str) -> bool:
    settings = _settings()
    if not _supabase_enabled(settings):
        return False

    safe = relative_path.lstrip("/").replace("\\", "/")
    bucket_key = _supabase_bucket_and_key(settings, safe)
    if not bucket_key:
        return False
    bucket, key = bucket_key

    with _SUPABASE_UPLOAD_LOCK:
        if safe in _UPLOADED_SUPABASE_MEDIA:
            return True

    abs_path = settings.media_dir / safe
    if not abs_path.exists():
        return False

    try:
        body = abs_path.read_bytes()
    except Exception:
        return False
    if not body:
        return False

    try:
        res = requests.post(
            f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{key}",
            headers={
                "Authorization": f"Bearer {settings.supabase_service_role_key}",
                "apikey": settings.supabase_service_role_key,
                "x-upsert": "true",
                "content-type": "image/jpeg",
            },
            data=body,
            timeout=20,
        )
    except Exception:
        return False

    if res.status_code >= 300:
        return False

    with _SUPABASE_UPLOAD_LOCK:
        _UPLOADED_SUPABASE_MEDIA.add(safe)
    return True


def _external_media_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    settings = _settings()
    safe = relative_path.lstrip("/").replace("\\", "/")

    if _supabase_enabled(settings):
        bucket_key = _supabase_bucket_and_key(settings, safe)
        if bucket_key:
            uploaded = _upload_media_to_supabase(safe)
            if uploaded:
                bucket, key = bucket_key
                return _supabase_public_url(settings, bucket, key)

    base = settings.public_media_base_url.strip()
    if not base:
        return None
    return f"{base.rstrip('/')}/media/{safe}"


def _debug_patch(job_id: str, patch: dict[str, Any] | None = None, event: dict[str, Any] | None = None) -> None:
    try:
        db.patch_job_debug(job_id, patch=patch, event=event)
    except Exception:
        pass


def _normalize_best_match(
    raw: Any,
    *,
    fallback_confidence: Any,
    fallback_tier: Any,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    tier = str(raw.get("source_tier") or fallback_tier or "").strip().lower()
    if tier not in {"exact", "similar"}:
        tier = "exact" if str(fallback_tier or "").strip().lower() == "exact" else "similar"

    try:
        confidence = float(raw.get("confidence"))
    except Exception:
        try:
            confidence = float(fallback_confidence or 0)
        except Exception:
            confidence = 0.0

    return {
        "title": str(raw.get("title") or ""),
        "source": str(raw.get("source") or ""),
        "price": raw.get("price"),
        "link": str(raw.get("link") or ""),
        "thumbnail": str(raw.get("thumbnail") or ""),
        "confidence": confidence,
        "reasoning": str(raw.get("reasoning") or ""),
        "source_tier": tier,
    }


def build_job_detail(job_id: str, media_base_url: str) -> dict[str, Any] | None:
    job = db.get_job(job_id)
    if not job:
        return None

    photos = db.list_photos(job_id)
    photo_by_id = {p["id"]: p for p in photos}
    face_detections = db.list_face_detections_for_job(job_id)
    clusters = db.list_face_clusters(job_id)
    selected = db.get_selected_cluster(job_id)
    items = db.list_clothing_items(job_id)
    debug = db.get_job_debug(job_id)

    photos_out = [
        {
            "id": p["id"],
            "url": media_rel_to_url(media_base_url, p["relative_path"]),
            "width": p["width"],
            "height": p["height"],
            "captured_at": p.get("captured_at"),
            "captured_at_epoch_ms": p.get("captured_at_epoch_ms"),
            "latitude": p.get("latitude"),
            "longitude": p.get("longitude"),
            "location_source": p.get("location_source"),
        }
        for p in photos
    ]

    detections_out = [
        {
            "id": d["id"],
            "photo_id": d["photo_id"],
            "cluster_id": d.get("cluster_id"),
            "bbox": d["bbox"],
            "confidence": d["confidence"],
        }
        for d in face_detections
    ]

    clusters_out = []
    for c in clusters:
        rep_photo = photo_by_id.get(c["rep_photo_id"])
        source_path = rep_photo["relative_path"] if rep_photo else ""
        rep_w = float(rep_photo.get("width") or 0) if rep_photo else 0.0
        rep_h = float(rep_photo.get("height") or 0) if rep_photo else 0.0
        rep_aspect_ratio = rep_w / rep_h if rep_w > 0 and rep_h > 0 else 1.0
        clusters_out.append(
            {
                "id": c["id"],
                "rep_photo_id": c["rep_photo_id"],
                "rep_bbox": c["rep_bbox"],
                "rep_aspect_ratio": rep_aspect_ratio,
                "member_count": c["member_count"],
                "source_url": media_rel_to_url(media_base_url, source_path),
            }
        )

    items_out = [
        {
            "id": i["id"],
            "photo_id": i["photo_id"],
            "closet_item_key": str(i.get("closet_item_key") or i["id"]),
            "category": i["category"],
            "description": i["description"],
            "colors": i["colors"],
            "pattern": i["pattern"] or "",
            "style": i["style"] or "",
            "brand_visible": i["brand_visible"],
            "visibility": i["visibility"],
            "confidence": i["confidence"],
            "bounding_box": i["bounding_box"],
            "crop_url": media_rel_to_url(media_base_url, i["crop_path"]) if i.get("crop_path") else None,
            "tier": i["tier"],
            "exact_matches": i["exact_matches"],
            "similar_products": i["similar_products"],
            "phia_products": i.get("phia_products") or [],
            "best_match": _normalize_best_match(
                i.get("best_match"),
                fallback_confidence=i.get("best_match_confidence"),
                fallback_tier=i.get("tier"),
            ),
            "best_match_confidence": i.get("best_match_confidence") or 0,
        }
        for i in items
    ]

    return {
        **job,
        "selected_cluster_id": selected["cluster_id"] if selected else None,
        "photos": photos_out,
        "face_detections": detections_out,
        "clusters": clusters_out,
        "items": items_out,
        "debug": debug,
    }


def build_styling_auto_runs_detail(job_id: str, media_base_url: str) -> dict[str, Any] | None:
    job = db.get_job(job_id)
    if not job:
        return None

    photos = db.list_photos(job_id)
    runs = db.list_styling_auto_runs(job_id)
    run_by_photo = {row["photo_id"]: row for row in runs}

    rows: list[dict[str, Any]] = []
    for photo in photos:
        photo_id = str(photo["id"])
        source_photo_path = str(photo.get("relative_path") or "")
        source_photo_url = media_rel_to_url(media_base_url, source_photo_path) if source_photo_path else ""

        run = run_by_photo.get(photo_id)
        if run is None:
            face_rel = f"face_tiles/{job_id}/{photo_id}.jpg"
            face_abs = _settings().media_dir / face_rel
            rows.append(
                {
                    "photo_id": photo_id,
                    "status": "waiting",
                    "source_photo_url": source_photo_url,
                    "face_crop_url": media_rel_to_url(media_base_url, face_rel) if face_abs.exists() else None,
                    "selected_person_crop_url": None,
                    "selected_person_bbox": None,
                    "gpt_selected_index": None,
                    "gpt_selection_reason": "",
                    "body_visible": None,
                    "prompt": "",
                    "render_id": None,
                    "best_variant_index": None,
                    "best_reason": "",
                    "skip_reason": "",
                    "error": "",
                    "variants": [],
                    "created_at": None,
                    "started_at": None,
                    "finished_at": None,
                    "updated_at": None,
                }
            )
            continue

        run_id = str(run["id"])
        variants_raw = db.list_styling_auto_variants(run_id)
        variants = [
            {
                "variant_index": int(v.get("variant_index") or 0),
                "prompt": str(v.get("prompt") or ""),
                "output_url": media_rel_to_url(media_base_url, str(v.get("output_path") or "")),
                "realism": v.get("realism"),
                "aesthetic": v.get("aesthetic"),
                "overall": v.get("overall"),
                "justification": str(v.get("justification") or ""),
                "is_best": bool(v.get("is_best")),
            }
            for v in variants_raw
        ]
        selected_person_crop_path = str(run.get("selected_person_crop_path") or "")
        face_crop_path = str(run.get("face_crop_path") or "")
        rows.append(
            {
                "photo_id": photo_id,
                "status": str(run.get("status") or "waiting"),
                "source_photo_url": source_photo_url,
                "face_crop_url": media_rel_to_url(media_base_url, face_crop_path) if face_crop_path else None,
                "selected_person_crop_url": (
                    media_rel_to_url(media_base_url, selected_person_crop_path)
                    if selected_person_crop_path
                    else None
                ),
                "selected_person_bbox": run.get("selected_person_bbox"),
                "gpt_selected_index": run.get("gpt_selected_index"),
                "gpt_selection_reason": str(run.get("gpt_selection_reason") or ""),
                "body_visible": (
                    None
                    if run.get("body_visible") is None
                    else bool(int(run.get("body_visible")))
                ),
                "prompt": str(run.get("prompt") or ""),
                "render_id": run.get("render_id"),
                "best_variant_index": run.get("best_variant_index"),
                "best_reason": str(run.get("best_reason") or ""),
                "skip_reason": str(run.get("skip_reason") or ""),
                "error": str(run.get("error") or ""),
                "variants": variants,
                "created_at": run.get("created_at"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "updated_at": run.get("updated_at"),
            }
        )

    return {
        "job_id": job_id,
        "runs": rows,
    }


def start_face_analysis(job_id: str, *, user_name: str | None = None) -> None:
    with _RUN_LOCK:
        if job_id in _RUNNING_FACE_JOBS:
            return
        _RUNNING_FACE_JOBS.add(job_id)

    def runner() -> None:
        try:
            _analyze_faces_worker(job_id, user_name=user_name)
        finally:
            with _RUN_LOCK:
                _RUNNING_FACE_JOBS.discard(job_id)

    threading.Thread(target=runner, name=f"faces-{job_id[:8]}", daemon=True).start()


def start_clothing_extraction(job_id: str, cluster_id: str) -> None:
    key = f"{job_id}:{cluster_id}"
    with _RUN_LOCK:
        if key in _RUNNING_CLOTHING_RUNS:
            return
        _RUNNING_CLOTHING_RUNS.add(key)

    def runner() -> None:
        try:
            _extract_clothing_worker(job_id, cluster_id)
        finally:
            with _RUN_LOCK:
                _RUNNING_CLOTHING_RUNS.discard(key)

    threading.Thread(target=runner, name=f"clothes-{job_id[:8]}", daemon=True).start()


def start_lookup_item(item_id: str) -> None:
    with _RUN_LOCK:
        if item_id in _RUNNING_LOOKUPS:
            return
        _RUNNING_LOOKUPS.add(item_id)

    def runner() -> None:
        sem = _LOOKUP_SEMAPHORE
        if sem is None:
            return
        sem.acquire()
        try:
            _lookup_item_worker(item_id)
        finally:
            sem.release()
            with _RUN_LOCK:
                _RUNNING_LOOKUPS.discard(item_id)

    threading.Thread(target=runner, name=f"lookup-{item_id[:8]}", daemon=True).start()


def start_styling_auto_run(
    *,
    job_id: str,
    photo_id: str,
    trigger_item_id: str | None = None,
) -> None:
    photo = db.get_photo(photo_id)
    if not photo:
        return

    source_photo_path = str(photo.get("relative_path") or "").strip()
    if not source_photo_path:
        return
    face_crop_rel = f"face_tiles/{job_id}/{photo_id}.jpg"
    face_crop_abs = _settings().media_dir / face_crop_rel
    face_crop_path = face_crop_rel if face_crop_abs.exists() else None

    claimed = db.claim_styling_auto_run(
        job_id=job_id,
        photo_id=photo_id,
        source_photo_path=source_photo_path,
        face_crop_path=face_crop_path,
        trigger_item_id=trigger_item_id,
    )
    if not bool(claimed.get("is_new")):
        return

    run_id = str(claimed["id"])
    with _RUN_LOCK:
        if run_id in _RUNNING_STYLING_AUTO_RUNS:
            return
        _RUNNING_STYLING_AUTO_RUNS.add(run_id)

    def runner() -> None:
        try:
            _run_styling_auto_worker(run_id)
        finally:
            with _RUN_LOCK:
                _RUNNING_STYLING_AUTO_RUNS.discard(run_id)

    threading.Thread(target=runner, name=f"style-auto-{run_id[:8]}", daemon=True).start()


def _analyze_faces_worker(job_id: str, *, user_name: str | None = None) -> None:
    online_pool: ThreadPoolExecutor | None = None
    try:
        print(TAG, "face analysis start", job_id)
        db.update_job(job_id, status="analyzing_faces", error=None)
        _debug_patch(
            job_id,
            patch={
                "stages": {"face_analysis": "running"},
                "face_analysis": {
                    "started_at": db.utc_now_iso(),
                    "user_name": user_name,
                    "inserted_cluster_count": 0,
                },
            },
            event={"type": "face_analysis_started", "message": "Face analysis started"},
        )
        db.clear_face_analysis(job_id)

        photos = db.list_photos(job_id)
        if not photos:
            db.update_job(job_id, status="failed", error="No photos uploaded")
            _debug_patch(
                job_id,
                patch={"stages": {"face_analysis": "failed"}},
                event={"type": "face_analysis_failed", "message": "No photos uploaded"},
            )
            return

        online_refs_future = None
        if user_name:
            online_pool = ThreadPoolExecutor(max_workers=1)
            online_refs_future = online_pool.submit(_collect_online_face_refs, job_id, user_name)

        clusters = _analyze_faces_with_rekognition(job_id, photos)
        if clusters is None:
            clusters = _analyze_faces_local(job_id, photos)

        if not clusters:
            if online_pool is not None:
                online_pool.shutdown(wait=False, cancel_futures=True)
            db.update_job(job_id, status="failed", error="No faces detected above threshold")
            _debug_patch(
                job_id,
                patch={"stages": {"face_analysis": "failed"}},
                event={"type": "face_analysis_failed", "message": "No faces detected above threshold"},
            )
            return

        total_faces = sum(len(c["members"]) for c in clusters)
        faces_by_photo: dict[str, int] = {}
        for cluster in clusters:
            for member in cluster["members"]:
                pid = member.get("photo_id")
                if pid:
                    faces_by_photo[pid] = int(faces_by_photo.get(pid, 0)) + 1
        _debug_patch(
            job_id,
            patch={
                "face_analysis": {
                    "total_clusters": len(clusters),
                    "total_detected_faces": total_faces,
                    "faces_by_photo": faces_by_photo,
                    "cluster_member_counts": [
                        len({m["photo_id"] for m in c["members"]}) for c in clusters
                    ],
                }
            },
            event={
                "type": "face_detection_complete",
                "message": f"Detected {total_faces} faces in {len(clusters)} clusters",
            },
        )

        saved_clusters: list[dict[str, Any]] = []
        for cluster in clusters:
            member_photo_ids = {m["photo_id"] for m in cluster["members"]}
            rep = max(cluster["members"], key=lambda m: float(m["confidence"]))
            cluster_id = db.insert_face_cluster(
                job_id=job_id,
                rep_photo_id=rep["photo_id"],
                rep_bbox=rep["bbox"],
                member_count=len(member_photo_ids),
            )
            db.set_detection_cluster([m["id"] for m in cluster["members"]], cluster_id)
            saved_clusters.append(
                {
                    "id": cluster_id,
                    "rep_photo_id": rep["photo_id"],
                    "rep_bbox": rep["bbox"],
                    "member_count": len(member_photo_ids),
                }
            )
            _debug_patch(
                job_id,
                patch={
                    "face_analysis": {
                        "inserted_cluster_count": len(saved_clusters),
                    }
                },
                event={
                    "type": "cluster_inserted",
                    "message": f"Inserted cluster {len(saved_clusters)}",
                    "data": {
                        "cluster_id": cluster_id,
                        "member_count": len(member_photo_ids),
                        "rep_photo_id": rep["photo_id"],
                    },
                },
            )

        online_refs: list[dict[str, Any]] = []
        if online_refs_future is not None:
            try:
                online_refs = online_refs_future.result(timeout=ONLINE_SEARCH_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                online_refs_future.cancel()
                online_refs = []
            except Exception:
                online_refs = []
            finally:
                if online_pool is not None:
                    online_pool.shutdown(wait=False, cancel_futures=True)
        elif online_pool is not None:
            online_pool.shutdown(wait=False, cancel_futures=True)

        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "running",
                    "reason": None,
                }
            },
            event={
                "type": "auto_face_scoring_started",
                "message": "Computing confidence score for automatic face selection",
            },
        )

        auto_cluster_id = _choose_auto_face_cluster(
            job_id=job_id,
            photos=photos,
            clusters=saved_clusters,
            online_face_refs=online_refs,
        )

        if auto_cluster_id:
            print(TAG, "auto-selected cluster", job_id, auto_cluster_id)
            _debug_patch(
                job_id,
                patch={"stages": {"face_analysis": "complete", "cluster_selection": "auto_selected"}},
                event={
                    "type": "auto_cluster_selected",
                    "message": "Auto-selected face cluster",
                    "data": {"cluster_id": auto_cluster_id},
                },
            )
            start_clothing_extraction(job_id, auto_cluster_id)
        else:
            db.update_job(job_id, status="awaiting_face_pick", error=None)
            _debug_patch(
                job_id,
                patch={"stages": {"face_analysis": "complete", "cluster_selection": "awaiting_manual_pick"}},
                event={
                    "type": "awaiting_manual_cluster_pick",
                    "message": "Awaiting user face selection",
                },
            )

        print(TAG, "face analysis done", job_id, "clusters", len(clusters))
    except Exception as exc:  # pragma: no cover
        if online_pool is not None:
            online_pool.shutdown(wait=False, cancel_futures=True)
        db.update_job(job_id, status="failed", error=f"Face analysis failed: {exc}")
        _debug_patch(
            job_id,
            patch={"stages": {"face_analysis": "failed"}},
            event={"type": "face_analysis_failed", "message": f"Face analysis failed: {exc}"},
        )


def _get_rekognition_client() -> Any | None:
    settings = _settings()
    if not (
        boto3 is not None
        and settings.aws_access_key_id
        and settings.aws_secret_access_key
        and settings.aws_region
    ):
        return None
    try:
        return boto3.client(
            "rekognition",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    except Exception:
        return None


def _extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = re.findall(r"https?://[^\s)\"'<>]+", text)
    return [u.strip().rstrip(".,);]") for u in found]


def _search_serpapi_image_urls(user_name: str) -> list[str]:
    settings = _settings()
    if not settings.serpapi_key:
        return []

    queries = [f'"{user_name}"']
    raw_urls: list[str] = []
    for query in queries:
        try:
            res = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_images",
                    "q": query,
                    "api_key": settings.serpapi_key,
                    "num": max(10, settings.online_face_image_limit * 4),
                    "safe": "off",
                },
                timeout=ONLINE_SEARCH_TIMEOUT_SECONDS,
            )
            if res.status_code >= 300:
                continue
            data = res.json()
        except Exception:
            continue

        for item in data.get("images_results") or []:
            # Prefer the source image URL and avoid Serp proxy thumbnails.
            original = item.get("original")
            thumbnail = item.get("thumbnail")
            candidate = original if isinstance(original, str) and original.strip() else thumbnail
            if not isinstance(candidate, str):
                continue
            if "serpapi.com/searches/" in candidate:
                continue
            raw_urls.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in raw_urls:
        cleaned = (raw or "").strip()
        if not cleaned.startswith(("http://", "https://")):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
        if len(deduped) >= max(1, settings.online_face_image_limit * 6):
            break
    print(TAG, "serpapi image candidates", user_name, "count", len(deduped))
    return deduped


def _search_online_image_urls(user_name: str) -> list[str]:
    return _search_serpapi_image_urls(user_name)


def _download_image_bytes(url: str) -> bytes | None:
    try:
        res = requests.get(url, timeout=ONLINE_IMAGE_TIMEOUT_SECONDS)
    except Exception:
        return None
    if res.status_code >= 300:
        return None
    content_type = (res.headers.get("content-type") or "").lower()
    if "image" not in content_type:
        return None
    body = res.content
    if not body or len(body) > ONLINE_IMAGE_MAX_BYTES:
        return None
    return body


def _collect_online_face_refs(job_id: str, user_name: str) -> list[dict[str, Any]]:
    urls = _search_online_image_urls(user_name)
    _debug_patch(
        job_id,
        patch={
            "serp_lookup": {
                "query": user_name,
                "candidate_count": len(urls),
                "candidate_urls": urls[:30],
                "downloaded_count": 0,
                "filtered_face_count": 0,
                "filtered_face_urls": [],
            }
        },
        event={
            "type": "serp_lookup_complete",
            "message": f"Serp returned {len(urls)} candidate image URLs",
        },
    )
    if not urls:
        return []

    settings = _settings()
    limit = max(1, settings.online_face_image_limit)
    urls = urls[: max(limit * 2, limit)]

    downloaded: list[tuple[str, bytes]] = []
    for url in urls:
        payload = _download_image_bytes(url)
        if payload:
            downloaded.append((url, payload))
        if len(downloaded) >= limit:
            break

    if not downloaded:
        _debug_patch(
            job_id,
            patch={"serp_lookup": {"downloaded_count": 0}},
            event={
                "type": "serp_download_empty",
                "message": "No Serp candidate images were downloadable",
            },
        )
        return []

    client = _get_rekognition_client()
    if client is None:
        return []

    _debug_patch(
        job_id,
        patch={
            "serp_lookup": {
                "downloaded_count": len(downloaded),
                "downloaded_urls": [u for (u, _) in downloaded[:30]],
            }
        },
        event={
            "type": "serp_download_complete",
            "message": f"Downloaded {len(downloaded)} candidate images",
        },
    )

    filtered: list[tuple[str, bytes]] = []
    for url, image_bytes in downloaded:
        try:
            detected = client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["DEFAULT"],
            )
        except Exception:
            continue
        faces = detected.get("FaceDetails") or []
        if not faces:
            continue
        best = max(float(face.get("Confidence") or 0) for face in faces)
        if best >= ONLINE_FACE_DETECT_CONFIDENCE_MIN:
            filtered.append((url, image_bytes))
        if len(filtered) >= limit:
            break

    _debug_patch(
        job_id,
        patch={
            "serp_lookup": {
                "filtered_face_count": len(filtered),
                "filtered_face_urls": [u for (u, _) in filtered[:30]],
            }
        },
        event={
            "type": "serp_face_filter_complete",
            "message": f"{len(filtered)} online references passed face detection",
        },
    )

    return [{"url": u, "bytes": img} for (u, img) in filtered]


def _crop_to_jpeg_bytes(
    photo_path: Path,
    bbox: dict[str, float],
    *,
    pad: float = 0.4,
) -> bytes | None:
    try:
        with Image.open(photo_path).convert("RGB") as image:
            width, height = image.size
            left = float(bbox.get("left") or 0)
            top = float(bbox.get("top") or 0)
            box_w = float(bbox.get("width") or 0)
            box_h = float(bbox.get("height") or 0)

            x1 = max(0, int((left - box_w * pad) * width))
            y1 = max(0, int((top - box_h * pad) * height))
            x2 = min(width, int((left + box_w * (1 + pad)) * width))
            y2 = min(height, int((top + box_h * (1 + pad)) * height))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = image.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
    except Exception:
        return None


def _choose_auto_face_cluster(
    *,
    job_id: str,
    photos: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    online_face_refs: list[dict[str, Any]],
) -> str | None:
    if not clusters:
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "scored_clusters": [],
                    "auto_selected": False,
                    "reason": "No face clusters were available",
                }
            },
            event={
                "type": "auto_face_scoring_skipped",
                "message": "Auto face scoring skipped because no clusters were available",
            },
        )
        return None

    settings = _settings()
    total_photos = max(1, len(photos))
    ranked_clusters = sorted(
        clusters,
        key=lambda c: (-int(c.get("member_count") or 0), str(c.get("id") or "")),
    )
    top_cluster = ranked_clusters[0]
    top_cluster_id = str(top_cluster["id"])
    top_frequency = min(1.0, max(0.0, float(top_cluster.get("member_count") or 0) / total_photos))
    second_frequency = (
        min(1.0, max(0.0, float(ranked_clusters[1].get("member_count") or 0) / total_photos))
        if len(ranked_clusters) > 1
        else 0.0
    )
    baseline_second_score = AUTO_PICK_FREQUENCY_WEIGHT * second_frequency

    # Fast path: if one cluster clearly dominates frequency, auto-pick without expensive compare checks.
    fast_frequency_margin = top_frequency - second_frequency
    fast_path_passed = (
        len(photos) >= 2
        and top_frequency >= AUTO_PICK_FAST_FREQUENCY_THRESHOLD
        and fast_frequency_margin >= AUTO_PICK_FAST_MARGIN_THRESHOLD
    )
    if fast_path_passed:
        fast_score = top_frequency
        fast_scored_clusters: list[dict[str, Any]] = [
            {
                "cluster_id": top_cluster_id,
                "member_count": int(top_cluster.get("member_count") or 0),
                "frequency": top_frequency,
                "alignment": 0.0,
                "score": fast_score,
                "matched_urls": [],
            }
        ]
        if len(ranked_clusters) > 1:
            fast_scored_clusters.append(
                {
                    "cluster_id": str(ranked_clusters[1]["id"]),
                    "member_count": int(ranked_clusters[1].get("member_count") or 0),
                    "frequency": second_frequency,
                    "alignment": 0.0,
                    "score": second_frequency,
                    "matched_urls": [],
                }
            )
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "mode": "frequency_fast_path",
                    "weights": {
                        "frequency": 1.0,
                        "alignment": 0.0,
                    },
                    "thresholds": {
                        "score": settings.auto_face_pick_threshold,
                        "margin": settings.auto_face_pick_margin,
                        "alignment_floor": settings.auto_face_pick_alignment_floor,
                    },
                    "total_uploaded_photos": total_photos,
                    "online_reference_count": 0,
                    "top_score": fast_score,
                    "top_cluster_id": top_cluster_id,
                    "top_alignment": 0.0,
                    "top_frequency": top_frequency,
                    "second_score": second_frequency,
                    "margin": fast_frequency_margin,
                    "top_matched_urls": [],
                    "auto_selected": True,
                    "selected_cluster_id": top_cluster_id,
                    "reason": "Top frequency cluster dominated uploaded photos",
                    "scored_clusters": fast_scored_clusters,
                }
            },
            event={
                "type": "auto_face_scored",
                "message": "Auto face scoring complete (frequency fast path)",
                "data": {
                    "top_cluster_id": top_cluster_id,
                    "top_score": fast_score,
                    "margin": fast_frequency_margin,
                    "selected": True,
                },
            },
        )
        return top_cluster_id

    if not online_face_refs:
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "mode": "top_cluster_compare",
                    "scored_clusters": [
                        {
                            "cluster_id": top_cluster_id,
                            "member_count": int(top_cluster.get("member_count") or 0),
                            "frequency": top_frequency,
                            "alignment": 0.0,
                            "score": AUTO_PICK_FREQUENCY_WEIGHT * top_frequency,
                            "matched_urls": [],
                        }
                    ],
                    "auto_selected": False,
                    "reason": "No online face references were available for comparison",
                }
            },
            event={
                "type": "auto_face_scoring_skipped",
                "message": "Auto face scoring could not compare due to missing online references",
            },
        )
        return None

    client = _get_rekognition_client()
    if client is None:
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "scored_clusters": [],
                    "auto_selected": False,
                    "reason": "Rekognition client unavailable for compare step",
                }
            },
            event={
                "type": "auto_face_scoring_skipped",
                "message": "Auto face scoring skipped because Rekognition client is unavailable",
            },
        )
        return None

    photo_by_id = {p["id"]: p for p in photos}
    top_photo = photo_by_id.get(str(top_cluster["rep_photo_id"]))
    if not top_photo:
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "scored_clusters": [],
                    "auto_selected": False,
                    "reason": "Top cluster representative photo was missing",
                }
            },
            event={
                "type": "auto_face_scoring_empty",
                "message": "Top cluster representative photo missing; cannot compare",
            },
        )
        return None

    top_photo_path = settings.media_dir / top_photo["relative_path"]
    source_bytes = _crop_to_jpeg_bytes(top_photo_path, top_cluster["rep_bbox"], pad=0.4)
    if not source_bytes:
        _debug_patch(
            job_id,
            patch={
                "auto_face_score": {
                    "state": "complete",
                    "scored_clusters": [],
                    "auto_selected": False,
                    "reason": "Could not create source face crop for top cluster",
                }
            },
            event={
                "type": "auto_face_scoring_empty",
                "message": "Could not crop top cluster face for comparison",
            },
        )
        return None

    refs_to_check = online_face_refs[: max(1, min(AUTO_PICK_MAX_REFERENCE_CHECKS, len(online_face_refs)))]
    best_similarity = 0.0
    matched_urls: set[str] = set()
    for ref in refs_to_check:
        target_bytes = ref.get("bytes")
        target_url = str(ref.get("url") or "")
        if not isinstance(target_bytes, (bytes, bytearray)):
            continue
        try:
            compared = client.compare_faces(
                SourceImage={"Bytes": source_bytes},
                TargetImage={"Bytes": target_bytes},
                SimilarityThreshold=REK_SIMILARITY_THRESHOLD,
            )
        except Exception:
            continue
        for match in compared.get("FaceMatches", []) or []:
            similarity = float(match.get("Similarity") or 0)
            if similarity > best_similarity:
                best_similarity = similarity
            if target_url:
                matched_urls.add(target_url)

    alignment = min(1.0, max(0.0, best_similarity / 100.0))
    top_score = min(
        1.0,
        max(0.0, AUTO_PICK_FREQUENCY_WEIGHT * top_frequency + AUTO_PICK_ALIGNMENT_WEIGHT * alignment),
    )
    margin = top_score - baseline_second_score

    scored: list[dict[str, Any]] = [
        {
            "cluster_id": top_cluster_id,
            "member_count": int(top_cluster.get("member_count") or 0),
            "frequency": top_frequency,
            "alignment": alignment,
            "score": top_score,
            "matched_urls": sorted(matched_urls)[:30],
        }
    ]
    if len(ranked_clusters) > 1:
        scored.append(
            {
                "cluster_id": str(ranked_clusters[1]["id"]),
                "member_count": int(ranked_clusters[1].get("member_count") or 0),
                "frequency": second_frequency,
                "alignment": 0.0,
                "score": baseline_second_score,
                "matched_urls": [],
            }
        )

    print(
        TAG,
        "auto-face score top",
        f"{float(top_score):.3f}",
        "freq",
        f"{float(top_frequency):.3f}",
        "align",
        f"{float(alignment):.3f}",
        "margin",
        f"{margin:.3f}",
    )

    passed = (
        float(top_score) >= settings.auto_face_pick_threshold
        and float(alignment) >= settings.auto_face_pick_alignment_floor
        and margin >= settings.auto_face_pick_margin
    )
    selected_cluster_id = top_cluster_id if passed else None
    _debug_patch(
        job_id,
        patch={
            "auto_face_score": {
                "state": "complete",
                "mode": "top_cluster_compare",
                "weights": {
                    "frequency": AUTO_PICK_FREQUENCY_WEIGHT,
                    "alignment": AUTO_PICK_ALIGNMENT_WEIGHT,
                },
                "thresholds": {
                    "score": settings.auto_face_pick_threshold,
                    "margin": settings.auto_face_pick_margin,
                    "alignment_floor": settings.auto_face_pick_alignment_floor,
                },
                "total_uploaded_photos": total_photos,
                "online_reference_count": len(refs_to_check),
                "top_score": float(top_score),
                "top_cluster_id": top_cluster_id,
                "top_alignment": float(alignment),
                "top_frequency": float(top_frequency),
                "second_score": baseline_second_score,
                "margin": margin,
                "top_matched_urls": sorted(matched_urls)[:30],
                "auto_selected": passed,
                "selected_cluster_id": selected_cluster_id,
                "reason": "Top frequency cluster compared against online references",
                "scored_clusters": scored[:30],
            }
        },
        event={
            "type": "auto_face_scored",
            "message": "Auto face scoring complete",
            "data": {
                "top_cluster_id": top_cluster_id,
                "top_score": float(top_score),
                "margin": margin,
                "matched_url_count": len(matched_urls),
                "selected": passed,
            },
        },
    )
    return selected_cluster_id


def _analyze_faces_with_rekognition(
    job_id: str,
    photos: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    settings = _settings()
    client = _get_rekognition_client()
    if client is None:
        return None

    collection_id = f"phiahacks-{uuid.uuid4().hex[:8]}"
    try:
        try:
            client.create_collection(CollectionId=collection_id)
        except Exception:
            pass

        all_faces: list[dict[str, Any]] = []
        for photo in photos:
            path = settings.media_dir / photo["relative_path"]
            try:
                img_bytes = path.read_bytes()
            except Exception:
                continue

            try:
                result = client.index_faces(
                    CollectionId=collection_id,
                    Image={"Bytes": img_bytes},
                    ExternalImageId=photo["id"],
                    DetectionAttributes=["DEFAULT"],
                    MaxFaces=15,
                    QualityFilter="AUTO",
                )
            except Exception:
                continue

            for rec in result.get("FaceRecords", []) or []:
                face = rec.get("Face", {}) or {}
                face_id = face.get("FaceId")
                confidence = float(face.get("Confidence") or 0)
                box = face.get("BoundingBox") or {}
                if not face_id or confidence < REK_CONFIDENCE_THRESHOLD:
                    continue

                bbox = {
                    "left": float(box.get("Left") or 0),
                    "top": float(box.get("Top") or 0),
                    "width": float(box.get("Width") or 0),
                    "height": float(box.get("Height") or 0),
                }

                detection_id = db.insert_face_detection(
                    photo_id=photo["id"],
                    bbox=bbox,
                    confidence=confidence,
                    feature=None,
                )
                all_faces.append(
                    {
                        "id": detection_id,
                        "photo_id": photo["id"],
                        "bbox": bbox,
                        "confidence": confidence,
                        "face_id": face_id,
                    }
                )

        if not all_faces:
            return []

        parent: dict[str, str] = {}
        for face in all_faces:
            parent[face["face_id"]] = face["face_id"]

        def find(x: str) -> str:
            cur = x
            while parent[cur] != cur:
                parent[cur] = parent[parent[cur]]
                cur = parent[cur]
            return cur

        def union(a: str, b: str) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[ra] = rb

        own_ids = set(parent.keys())
        for face in all_faces:
            fid = face["face_id"]
            try:
                matches = client.search_faces(
                    CollectionId=collection_id,
                    FaceId=fid,
                    FaceMatchThreshold=REK_SIMILARITY_THRESHOLD,
                    MaxFaces=50,
                )
            except Exception:
                continue

            for match in matches.get("FaceMatches", []) or []:
                m_face = match.get("Face") or {}
                m_id = m_face.get("FaceId")
                if m_id and m_id in own_ids and m_id != fid:
                    union(fid, m_id)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for face in all_faces:
            root = find(face["face_id"])
            grouped.setdefault(root, []).append(face)

        clusters = [{"members": members} for members in grouped.values()]
        clusters.sort(key=lambda c: len({m["photo_id"] for m in c["members"]}), reverse=True)
        return clusters
    except Exception:
        return None
    finally:
        try:
            client.delete_collection(CollectionId=collection_id)
        except Exception:
            pass


def _analyze_faces_local(job_id: str, photos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = _settings()
    detections: list[dict[str, Any]] = []

    for photo in photos:
        abs_path = settings.media_dir / photo["relative_path"]
        for det in _detect_faces(abs_path):
            detection_id = db.insert_face_detection(
                photo_id=photo["id"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                feature=det.get("feature"),
            )
            detections.append(
                {
                    "id": detection_id,
                    "photo_id": photo["id"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "feature": det.get("feature"),
                }
            )

    if not detections:
        return []
    return _cluster_detections(detections)


def _extract_clothing_worker(job_id: str, cluster_id: str) -> None:
    try:
        print(TAG, "clothing extraction start", job_id, cluster_id)
        db.upsert_selected_cluster(job_id, cluster_id)
        db.clear_clothing_items(job_id)
        db.update_job(job_id, status="extracting_clothing", error=None)
        _debug_patch(
            job_id,
            patch={
                "stages": {"clothing_extraction": "running"},
                "clothing_extraction": {
                    "cluster_id": cluster_id,
                    "started_at": db.utc_now_iso(),
                    "item_count": 0,
                },
            },
            event={
                "type": "clothing_extraction_started",
                "message": "Started clothing extraction",
                "data": {"cluster_id": cluster_id},
            },
        )

        detections = db.list_face_detections_for_cluster(cluster_id)
        if not detections:
            db.update_job(job_id, status="done", error=None)
            _debug_patch(
                job_id,
                patch={
                    "stages": {"clothing_extraction": "complete"},
                    "clothing_extraction": {"item_count": 0, "photo_count": 0, "per_photo_steps": []},
                },
                event={
                    "type": "clothing_extraction_complete",
                    "message": "No detections in selected cluster; extraction complete",
                },
            )
            return

        settings = _settings()
        photos = {p["id"]: p for p in db.list_photos(job_id)}

        bbox_by_photo: dict[str, dict[str, float]] = {}
        for det in detections:
            if det["photo_id"] not in bbox_by_photo:
                bbox_by_photo[det["photo_id"]] = det["bbox"]

        entries = list(bbox_by_photo.items())
        item_ids: list[str] = []
        per_photo_steps_by_photo: dict[str, dict[str, Any]] = {}

        _debug_patch(
            job_id,
            patch={
                "clothing_extraction": {
                    "photo_count": len(entries),
                    "completed_photo_count": 0,
                    "per_photo_steps": [],
                },
            },
        )

        def publish_extraction_progress() -> None:
            ordered_steps = [
                per_photo_steps_by_photo[photo_id]
                for photo_id, _ in entries
                if photo_id in per_photo_steps_by_photo
            ]
            _debug_patch(
                job_id,
                patch={
                    "clothing_extraction": {
                        "item_count": len(item_ids),
                        "photo_count": len(entries),
                        "completed_photo_count": len(per_photo_steps_by_photo),
                        "per_photo_steps": ordered_steps[:120],
                    },
                },
            )

        def process_one(entry: tuple[str, dict[str, float]]) -> dict[str, Any]:
            photo_id, face_bbox = entry
            photo = photos.get(photo_id)
            if not photo:
                return {
                    "item_ids": [],
                    "step": {
                        "photo_id": photo_id,
                        "reason": "missing_photo_record",
                        "yolo_world_detections": [],
                        "gpt_cleaned_items": [],
                        "visible_items": [],
                        "inserted_item_count": 0,
                    },
                }

            photo_path = settings.media_dir / photo["relative_path"]
            face_tile_rel = _save_face_tile(photo_path, job_id, photo_id, face_bbox)
            face_tile_path = settings.media_dir / face_tile_rel

            # Start auto styling per photo immediately (in parallel with item lookup),
            # instead of waiting for downstream product matching to finish.
            start_styling_auto_run(job_id=job_id, photo_id=photo_id)

            step: dict[str, Any] = {
                "photo_id": photo_id,
                "photo_relative_path": photo["relative_path"],
                "yolo_world_detections": [],
                "gpt_cleaned_items": [],
                "visible_items": [],
                "inserted_item_count": 0,
                "postprocess_source": "fallback",
                "yolo_requested_source": "default",
                "yolo_requested_classes": YOLO_WORLD_CLASS_NAMES[:],
                "gpt_identified_classes": [],
            }

            external_photo_url = _external_media_url(photo["relative_path"])
            step["external_photo_url"] = external_photo_url
            if not external_photo_url:
                step["reason"] = "missing_external_photo_url"
                return {"item_ids": [], "step": step}

            gpt_classes = _identify_yolo_classes_with_openai(
                photo_path=photo_path,
                face_bbox=face_bbox,
                face_tile_path=face_tile_path,
            )
            step["gpt_identified_classes"] = gpt_classes[:60]
            yolo_classes = gpt_classes if gpt_classes else YOLO_WORLD_CLASS_NAMES
            step["yolo_requested_classes"] = yolo_classes[:60]
            step["yolo_requested_source"] = "gpt" if gpt_classes else "default"

            # YOLO-World is the source of truth for regions; GPT only filters/selects among these boxes.
            yolo_debug: dict[str, Any] = {}
            detections = _detect_objects(external_photo_url, class_names=yolo_classes, debug=yolo_debug)
            step["yolo_debug"] = yolo_debug
            step["yolo_world_detections"] = [
                {
                    "label": str(det.get("label") or ""),
                    "confidence": float(det.get("confidence") or 0),
                    "bbox": [float(v) for v in (det.get("bbox") or [])[:4]],
                }
                for det in detections
            ][:120]
            if not detections:
                if str(yolo_debug.get("status") or "") in {"failed", "exception", "skipped"}:
                    step["reason"] = "yolo_error"
                else:
                    step["reason"] = "no_yolo_detections"
                return {"item_ids": [], "step": step}

            items, postprocess_source = _extract_items_from_detections(
                photo_path,
                face_bbox,
                face_tile_path,
                detections,
            )
            step["postprocess_source"] = postprocess_source
            step["gpt_cleaned_items"] = [
                {
                    "category": str(item.get("category") or ""),
                    "confidence": float(item.get("confidence") or 0),
                    "visibility": str(item.get("visibility") or "clear"),
                    "description": str(item.get("description") or ""),
                    "bounding_box": item.get("bounding_box") or {},
                }
                for item in items
            ][:80]
            # Temporary: allow low-confidence items through to product lookup.
            # We only block items explicitly marked as obscured.
            visible = [i for i in items if i.get("visibility", "clear") != "obscured"]
            step["visible_items"] = [
                {
                    "category": str(item.get("category") or ""),
                    "confidence": float(item.get("confidence") or 0),
                    "visibility": str(item.get("visibility") or "clear"),
                    "description": str(item.get("description") or ""),
                    "bounding_box": item.get("bounding_box") or {},
                }
                for item in visible
            ][:80]
            if not visible:
                step["reason"] = "all_items_filtered"
                return {"item_ids": [], "step": step}

            inserted: list[str] = []
            for item in visible:
                crop_rel = _save_item_crop(
                    photo_path=photo_path,
                    job_id=job_id,
                    photo_id=photo_id,
                    bbox=item["bounding_box"],
                )

                item_id = db.insert_clothing_item(
                    job_id=job_id,
                    photo_id=photo_id,
                    category=item.get("category") or item.get("description") or "clothing item",
                    description=item.get("description", "Unlabeled clothing item"),
                    colors=item.get("colors", []),
                    pattern=item.get("pattern", ""),
                    style=item.get("style", ""),
                    brand_visible=item.get("brand_visible"),
                    visibility=item.get("visibility", "clear"),
                    confidence=float(item.get("confidence", 0)),
                    bounding_box=item["bounding_box"],
                    crop_path=crop_rel,
                    tier="pending",
                    exact_matches=[],
                    similar_products=[],
                    phia_products=[],
                    best_match=None,
                    best_match_confidence=0,
                )
                inserted.append(item_id)

            step["inserted_item_count"] = len(inserted)
            step["inserted_item_ids"] = inserted[:80]
            step["reason"] = "inserted_items"
            return {"item_ids": inserted, "step": step}

        concurrency = max(1, settings.photo_concurrency)
        if concurrency == 1:
            for entry in entries:
                result = process_one(entry)
                item_ids.extend(result.get("item_ids", []))
                step = result.get("step", {})
                if isinstance(step, dict):
                    step_photo_id = str(step.get("photo_id") or entry[0])
                    per_photo_steps_by_photo[step_photo_id] = step
                publish_extraction_progress()
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_to_photo_id = {pool.submit(process_one, entry): entry[0] for entry in entries}
                for future in as_completed(future_to_photo_id):
                    result = future.result()
                    item_ids.extend(result.get("item_ids", []))
                    step = result.get("step", {})
                    if isinstance(step, dict):
                        fallback_photo_id = future_to_photo_id[future]
                        step_photo_id = str(step.get("photo_id") or fallback_photo_id)
                        per_photo_steps_by_photo[step_photo_id] = step
                    publish_extraction_progress()

        per_photo_steps = [
            per_photo_steps_by_photo[photo_id]
            for photo_id, _ in entries
            if photo_id in per_photo_steps_by_photo
        ]

        closet_key_by_item, dedupe_source = _assign_closet_item_keys(job_id, item_ids)
        unique_closet_item_count = (
            len(set(closet_key_by_item.values()))
            if closet_key_by_item
            else len(set(item_ids))
        )

        db.update_job(job_id, status="done", error=None)
        _debug_patch(
            job_id,
            patch={
                "stages": {"clothing_extraction": "complete"},
                "clothing_extraction": {
                    "item_count": len(item_ids),
                    "closet_item_count": unique_closet_item_count,
                    "closet_dedupe_source": dedupe_source,
                    "photo_count": len(entries),
                    "completed_photo_count": len(entries),
                    "finished_at": db.utc_now_iso(),
                    "per_photo_steps": per_photo_steps[:120],
                },
            },
            event={
                "type": "clothing_extraction_complete",
                "message": (
                    "Clothing extraction complete with "
                    f"{len(item_ids)} photo-tagged items and {unique_closet_item_count} closet cards"
                ),
            },
        )
        for item_id in item_ids:
            start_lookup_item(item_id)

        print(TAG, "clothing extraction done", job_id, "items", len(item_ids))
    except Exception as exc:  # pragma: no cover
        db.update_job(job_id, status="failed", error=f"Clothing extraction failed: {exc}")
        _debug_patch(
            job_id,
            patch={"stages": {"clothing_extraction": "failed"}},
            event={
                "type": "clothing_extraction_failed",
                "message": f"Clothing extraction failed: {exc}",
            },
        )


def _lookup_item_worker(item_id: str) -> None:
    try:
        item = db.get_clothing_item(item_id)
        if not item:
            return

        settings = _settings()
        crop_rel = item.get("crop_path")
        if not crop_rel:
            photo = db.get_photo(item["photo_id"])
            if photo:
                photo_path = settings.media_dir / photo["relative_path"]
                crop_rel = _save_item_crop(
                    photo_path=photo_path,
                    job_id=item["job_id"],
                    photo_id=item["photo_id"],
                    bbox=item["bounding_box"],
                )

        crop_path = settings.media_dir / (crop_rel or "")
        crop_path_for_lookup = crop_path if crop_path.exists() else None

        phia_products: list[dict[str, Any]] = []

        exact_matches: list[dict[str, Any]] = []
        similar_products: list[dict[str, Any]] = []
        tier: str = "generic"

        crop_external_url = _external_media_url(crop_rel)
        if settings.serpapi_key and crop_external_url:
            exact_matches = _find_exact_match(crop_external_url, item.get("description") or "")
            if exact_matches:
                tier = "exact"

        if settings.serpapi_key and tier != "exact" and item.get("description"):
            similar_products = _find_similar_products(item["description"])
            if similar_products:
                tier = "similar"

        best_match: dict[str, Any] | None = None
        best_confidence = 0.0
        final_tier = tier

        rank_source_tier: str | None = (
            "exact" if exact_matches else "similar" if similar_products else None
        )
        candidates = exact_matches if rank_source_tier == "exact" else similar_products if rank_source_tier == "similar" else []

        if candidates:
            photo = db.get_photo(item["photo_id"])
            if photo:
                photo_path = settings.media_dir / photo["relative_path"]
                face_tile_path = settings.media_dir / f"face_tiles/{item['job_id']}/{item['photo_id']}.jpg"

                ranked = _rank_candidates(
                    photo_path=photo_path,
                    face_tile_path=face_tile_path if face_tile_path.exists() else None,
                    crop_path=crop_path_for_lookup,
                    description=item.get("description") or "",
                    candidates=candidates,
                )
                if ranked:
                    if ranked.get("tier") == "none" or ranked.get("best_index") is None:
                        final_tier = "generic"
                    else:
                        idx = int(ranked["best_index"])
                        if 0 <= idx < len(candidates):
                            chosen = candidates[idx]
                            best_match = {
                                "title": chosen.get("title") or "",
                                "source": chosen.get("source") or "",
                                "price": chosen.get("price"),
                                "link": chosen.get("link") or "",
                                "thumbnail": chosen.get("thumbnail") or "",
                                "confidence": float(ranked.get("confidence") or 0),
                                "reasoning": ranked.get("reasoning") or "",
                                "source_tier": rank_source_tier,
                            }
                            best_confidence = float(ranked.get("confidence") or 0)
                            final_tier = "exact" if ranked.get("tier") == "exact" else "similar"

        # Fallback: if we picked a best match but it has no link, try Phia's
        # ProductsGoogleShoppingApi using the crop image URL.
        if best_match and not str(best_match.get("link") or "").strip():
            fallback_name = str(best_match.get("title") or item.get("description") or "").strip()
            phia_products = _find_phia_products_for_fallback(
                scraped_name=fallback_name,
                crop_external_url=crop_external_url,
            )
            if phia_products:
                top = phia_products[0]
                product_url = str(top.get("product_url") or "").strip()
                if product_url:
                    best_match["link"] = product_url
                if not str(best_match.get("thumbnail") or "").strip():
                    best_match["thumbnail"] = str(top.get("img_url") or "")
                if not str(best_match.get("source") or "").strip():
                    best_match["source"] = (
                        str(top.get("source_display_name") or "")
                        or str(top.get("primary_brand_name") or "")
                        or "Phia"
                    )
                if not best_match.get("price"):
                    price_usd = top.get("price_usd")
                    if isinstance(price_usd, (int, float)):
                        best_match["price"] = f"${float(price_usd):.2f}"

        if final_tier == "pending":
            final_tier = "generic"

        db.update_clothing_item(
            item_id=item_id,
            tier=final_tier,
            exact_matches=exact_matches,
            similar_products=similar_products,
            phia_products=phia_products,
            best_match=best_match,
            best_match_confidence=best_confidence,
            crop_path=crop_rel,
        )
    except Exception as exc:  # pragma: no cover
        print(TAG, "lookup failed", item_id, exc)


def _detect_person_boxes_for_styling(source_photo_relative_path: str) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    debug: dict[str, Any] = {}
    external_photo_url = _external_media_url(source_photo_relative_path)
    if not external_photo_url:
        debug["status"] = "failed"
        debug["error"] = "missing_external_photo_url"
        return [], debug

    detections = _detect_objects(
        external_photo_url,
        class_names=["person"],
        debug=debug,
    )
    boxes: list[tuple[int, int, int, int]] = []
    for det in detections:
        raw_bbox = det.get("bbox") or []
        if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox[:4]]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes, debug


def _run_styling_auto_worker(run_id: str) -> None:
    run = db.get_styling_auto_run(run_id)
    if not run:
        return
    job_id = str(run["job_id"])
    photo_id = str(run["photo_id"])
    db.update_styling_auto_run(
        run_id,
        status="running",
        started_at=db.utc_now_iso(),
        error=None,
        skip_reason=None,
    )
    _debug_patch(
        job_id,
        event={
            "type": "styling_auto_started",
            "message": "Automatic styling generation started",
            "data": {"photo_id": photo_id, "run_id": run_id},
        },
    )

    try:
        from app.services import model_render

        person_boxes, yolo_debug = _detect_person_boxes_for_styling(str(run["source_photo_path"]))
        _debug_patch(
            job_id,
            event={
                "type": "styling_auto_person_detection",
                "message": "Ran YOLO-World person detection for styling auto-run",
                "data": {
                    "photo_id": photo_id,
                    "run_id": run_id,
                    "person_box_count": len(person_boxes),
                    "yolo_debug": yolo_debug,
                },
            },
        )

        result = model_render.auto_generate_for_photo(
            settings=_settings(),
            job_id=job_id,
            photo_id=photo_id,
            source_photo_relative_path=str(run["source_photo_path"]),
            face_crop_relative_path=(
                str(run.get("face_crop_path"))
                if run.get("face_crop_path") is not None
                else None
            ),
            person_boxes=person_boxes,
        )

        if str(result.get("status") or "") == "skipped":
            db.update_styling_auto_run(
                run_id,
                status="skipped",
                selected_person_crop_path=result.get("selected_person_crop_path"),
                selected_person_bbox=result.get("selected_person_bbox"),
                gpt_selected_index=result.get("gpt_selected_index"),
                gpt_selection_reason=result.get("gpt_selection_reason"),
                body_visible=result.get("body_visible"),
                skip_reason=result.get("skip_reason"),
                finished_at=db.utc_now_iso(),
                error=None,
            )
            _debug_patch(
                job_id,
                event={
                    "type": "styling_auto_skipped",
                    "message": "Automatic styling generation skipped",
                    "data": {"photo_id": photo_id, "run_id": run_id},
                },
            )
            return

        db.update_styling_auto_run(
            run_id,
            status="completed",
            selected_person_crop_path=result.get("selected_person_crop_path"),
            selected_person_bbox=result.get("selected_person_bbox"),
            gpt_selected_index=result.get("gpt_selected_index"),
            gpt_selection_reason=result.get("gpt_selection_reason"),
            body_visible=result.get("body_visible"),
            prompt=result.get("prompt"),
            render_id=result.get("render_id"),
            best_variant_index=result.get("best_variant_index"),
            best_reason=result.get("best_reason"),
            finished_at=db.utc_now_iso(),
            error=None,
            skip_reason=None,
        )
        db.replace_styling_auto_variants(
            run_id,
            variants=result.get("variants") or [],
            best_variant_index=result.get("best_variant_index"),
        )
        _debug_patch(
            job_id,
            event={
                "type": "styling_auto_complete",
                "message": "Automatic styling generation completed",
                "data": {
                    "photo_id": photo_id,
                    "run_id": run_id,
                    "render_id": result.get("render_id"),
                    "best_variant_index": result.get("best_variant_index"),
                },
            },
        )
    except Exception as exc:
        skip_reason = ""
        status = "failed"
        try:
            from app.services import model_render

            if isinstance(exc, model_render.AutoStylingSkip):
                status = "skipped"
                skip_reason = str(exc) or "Target person body not visible enough"
        except Exception:
            pass

        db.update_styling_auto_run(
            run_id,
            status=status,
            finished_at=db.utc_now_iso(),
            error=None if status == "skipped" else str(exc),
            skip_reason=skip_reason if status == "skipped" else None,
        )
        _debug_patch(
            job_id,
            event={
                "type": "styling_auto_failed" if status == "failed" else "styling_auto_skipped",
                "message": (
                    f"Automatic styling generation failed: {exc}"
                    if status == "failed"
                    else "Automatic styling generation skipped"
                ),
                "data": {"photo_id": photo_id, "run_id": run_id},
            },
        )


def refresh_item_lookup(item_id: str) -> None:
    start_lookup_item(item_id)


def _detect_faces(photo_path: Path) -> list[dict[str, Any]]:
    with Image.open(photo_path).convert("RGB") as image:
        width, height = image.size
        rgb = image.copy()

    if np is None:
        return [_fallback_face(width, height)]

    arr = np.array(rgb)
    detections: list[dict[str, Any]] = []

    if cv2 is not None:
        try:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            )
            found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            for (x, y, w, h) in found:
                bbox = {
                    "left": max(0.0, min(1.0, float(x) / width)),
                    "top": max(0.0, min(1.0, float(y) / height)),
                    "width": max(0.0, min(1.0, float(w) / width)),
                    "height": max(0.0, min(1.0, float(h) / height)),
                }
                crop = arr[y : y + h, x : x + w]
                detections.append(
                    {
                        "bbox": bbox,
                        "confidence": 0.9,
                        "feature": _face_feature(crop),
                    }
                )
        except Exception:
            detections = []

    if not detections:
        return [_fallback_face(width, height)]
    return detections


def _fallback_face(width: int, height: int) -> dict[str, Any]:
    face_w = min(width * 0.35, height * 0.35)
    bbox = {
        "left": max(0.0, (width * 0.5 - face_w * 0.5) / width),
        "top": max(0.0, (height * 0.28 - face_w * 0.5) / height),
        "width": face_w / width,
        "height": face_w / height,
    }
    return {"bbox": bbox, "confidence": 0.45, "feature": [0.5, 0.5, 0.5]}


def _face_feature(face_crop: Any) -> list[float] | None:
    if np is None:
        return None
    if face_crop is None or getattr(face_crop, "size", 0) == 0:
        return None

    small = cv2.resize(face_crop, (48, 48)) if cv2 is not None else face_crop
    hist_parts = []
    for i in range(3):
        hist, _ = np.histogram(small[:, :, i], bins=8, range=(0, 255), density=True)
        hist_parts.append(hist)
    feat = np.concatenate(hist_parts)
    norm = np.linalg.norm(feat)
    if norm <= 0:
        return None
    feat = feat / norm
    return feat.astype(float).tolist()


def _cluster_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not detections:
        return []
    if np is None:
        return [{"members": detections}]

    clusters: list[dict[str, Any]] = []
    for det in detections:
        feat = det.get("feature")
        if feat is None:
            vec = np.array([det["bbox"]["left"], det["bbox"]["top"], det["bbox"]["width"]])
        else:
            vec = np.array(feat)

        best_idx = -1
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            score = _cosine(vec, cluster["centroid"])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= 0.92:
            clusters[best_idx]["members"].append(det)
            members = clusters[best_idx]["members"]
            stacked = np.stack([
                np.array(m.get("feature") if m.get("feature") is not None else [
                    m["bbox"]["left"],
                    m["bbox"]["top"],
                    m["bbox"]["width"],
                ])
                for m in members
            ])
            clusters[best_idx]["centroid"] = np.mean(stacked, axis=0)
        else:
            clusters.append({"members": [det], "centroid": vec})

    clusters.sort(key=lambda c: len({m["photo_id"] for m in c["members"]}), reverse=True)
    return clusters


def _cosine(a: Any, b: Any) -> float:
    if np is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_yolo_class_names(values: list[Any], *, max_items: int = 40) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = re.sub(r"\s+", " ", str(raw or "").strip().lower())
        if not text:
            continue
        if len(text) > 60:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _identify_yolo_classes_with_openai(
    *,
    photo_path: Path,
    face_bbox: dict[str, float],
    face_tile_path: Path | None,
) -> list[str]:
    settings = _settings()
    if not settings.openai_api_key:
        return []

    fbb = face_bbox
    face_desc = (
        f"face at normalized coordinates (x={fbb['left']:.3f}, y={fbb['top']:.3f}, "
        f"width={fbb['width']:.3f}, height={fbb['height']:.3f})"
    )
    instructions = (
        "Identify the clothing that the person with this face has on.\n"
        "Return short detector-friendly class names (1-3 words each).\n"
        "Good classes include: red shirt, blue gown, white tie, white shoes.\n"
        "If color is unclear, use the uncolored class name (for example: shirt, gown, tie, shoes).\n"
        "Do not include background objects, other people, body parts, or generic words.\n"
        f"Target hint: {face_desc}"
    )
    schema = {
        "type": "object",
        "properties": {
            "class_names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 40,
            }
        },
        "required": ["class_names"],
        "additionalProperties": False,
    }

    content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions}]
    if face_tile_path and face_tile_path.exists():
        content.append({"type": "input_image", "image_url": _data_url_for_image(face_tile_path), "detail": "high"})
    content.append({"type": "input_image", "image_url": _data_url_for_image(photo_path), "detail": "high"})

    payload = {
        "model": settings.openai_model,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "class_name_identification",
                "schema": schema,
                "strict": True,
            }
        },
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(TAG, "openai class identification failed", exc)
        return []

    raw = data.get("output_text")
    if not raw:
        for out in data.get("output", []):
            for c in out.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    raw = c["text"]
                    break
            if raw:
                break
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    class_names = parsed.get("class_names") if isinstance(parsed, dict) else None
    if not isinstance(class_names, list):
        return []
    return _normalize_yolo_class_names(class_names, max_items=40)


def _extract_items_from_detections(
    photo_path: Path,
    face_bbox: dict[str, float],
    face_tile_path: Path | None,
    detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    settings = _settings()
    if settings.openai_api_key:
        try:
            extracted = _extract_items_from_detections_with_openai(
                photo_path=photo_path,
                face_bbox=face_bbox,
                face_tile_path=face_tile_path,
                detections=detections,
            )
            if extracted:
                return extracted, "openai"
        except Exception as exc:
            print(TAG, "openai detection postprocess failed", exc)

    with Image.open(photo_path).convert("RGB") as image:
        img_w, img_h = image.size
    return _fallback_items_from_detections(detections, img_w, img_h), "fallback"


def _assign_closet_item_keys(job_id: str, item_ids: list[str]) -> tuple[dict[str, str], str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for raw in item_ids:
        item_id = str(raw or "").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        ordered_ids.append(item_id)

    if not ordered_ids:
        return {}, "none"

    rows: list[dict[str, Any]] = []
    for item_id in ordered_ids:
        row = db.get_clothing_item(item_id)
        if row:
            rows.append(row)

    if not rows:
        return {}, "none"

    mapping: dict[str, str] = {}
    source = "fallback"
    if len(rows) == 1:
        only_id = str(rows[0]["id"])
        mapping = {only_id: only_id}
        source = "single"
    else:
        openai_mapping = _dedupe_closet_items_with_openai(rows)
        if openai_mapping:
            mapping = openai_mapping
            source = "openai"
        else:
            mapping = _dedupe_closet_items_with_fallback(rows)

    valid_ids = {str(row["id"]) for row in rows}
    final: dict[str, str] = {}
    for row in rows:
        item_id = str(row["id"])
        canonical = str(mapping.get(item_id) or "").strip()
        if canonical not in valid_ids:
            canonical = item_id
        db.update_clothing_item_closet_key(item_id, canonical)
        final[item_id] = canonical

    return final, source


def _dedupe_closet_items_with_openai(items: list[dict[str, Any]]) -> dict[str, str]:
    settings = _settings()
    if not settings.openai_api_key or len(items) < 2:
        return {}

    candidates = items[:120]
    payload_items: list[dict[str, Any]] = []
    valid_ids: set[str] = set()
    for row in candidates:
        item_id = str(row.get("id") or "").strip()
        if not item_id:
            continue
        valid_ids.add(item_id)
        payload_items.append(
            {
                "item_id": item_id,
                "photo_id": str(row.get("photo_id") or ""),
                "category": str(row.get("category") or ""),
                "description": str(row.get("description") or ""),
                "colors": [str(c) for c in (row.get("colors") or []) if str(c).strip()],
                "pattern": str(row.get("pattern") or ""),
                "style": str(row.get("style") or ""),
                "brand_visible": str(row.get("brand_visible") or ""),
                "confidence": float(row.get("confidence") or 0),
            }
        )

    if len(payload_items) < 2:
        return {}

    schema = {
        "type": "object",
        "properties": {
            "canonical_assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "canonical_item_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["item_id", "canonical_item_id", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["canonical_assignments"],
        "additionalProperties": False,
    }

    instructions = (
        "You are deduplicating closet items extracted from multiple photos of the SAME person.\n"
        "Goal: assign each item to a canonical item id so duplicate detections of the same physical garment "
        "(for example, the same jacket in two photos) share one canonical id.\n"
        "Rules:\n"
        "1) Only merge when it is very likely the same physical item, not merely a similar category.\n"
        "2) If uncertain, keep separate by mapping an item to itself.\n"
        "3) canonical_item_id MUST be one of the provided item_id values.\n"
        "4) Return one assignment per provided item.\n"
        "5) Prefer the highest-confidence item as canonical within a duplicate group.\n"
        f"Items JSON:\n{json.dumps(payload_items)}"
    )

    payload = {
        "model": settings.openai_model,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": instructions}],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "closet_item_dedupe",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(TAG, "openai closet dedupe failed", exc)
        return {}

    raw = data.get("output_text")
    if not raw:
        for out in data.get("output", []):
            for content in out.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    raw = content["text"]
                    break
            if raw:
                break
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    assignments = parsed.get("canonical_assignments") if isinstance(parsed, dict) else None
    if not isinstance(assignments, list):
        return {}

    out: dict[str, str] = {}
    for entry in assignments:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("item_id") or "").strip()
        canonical_item_id = str(entry.get("canonical_item_id") or "").strip()
        if item_id not in valid_ids or canonical_item_id not in valid_ids:
            continue
        out[item_id] = canonical_item_id
    return out


def _dedupe_closet_items_with_fallback(items: list[dict[str, Any]]) -> dict[str, str]:
    canonical_by_signature: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for row in sorted(
        items,
        key=lambda r: (
            str(r.get("created_at") or ""),
            -float(r.get("confidence") or 0),
        ),
    ):
        item_id = str(row.get("id") or "").strip()
        if not item_id:
            continue
        signature = _closet_item_signature(row)
        canonical = canonical_by_signature.get(signature)
        if canonical is None:
            canonical = item_id
            canonical_by_signature[signature] = canonical
        mapping[item_id] = canonical
    return mapping


def _closet_item_signature(item: dict[str, Any]) -> str:
    description = _compact_search_description(
        str(item.get("description") or ""),
        fallback=str(item.get("category") or "clothing item"),
    )
    normalized_colors = sorted(
        {
            re.sub(r"[^a-z0-9]", "", str(color or "").lower())
            for color in (item.get("colors") or [])
        }
    )
    color_key = ",".join([c for c in normalized_colors if c][:2])
    pattern = re.sub(r"[^a-z0-9]", "", str(item.get("pattern") or "").lower())
    style = re.sub(r"[^a-z0-9]", "", str(item.get("style") or "").lower())
    brand = re.sub(r"[^a-z0-9]", "", str(item.get("brand_visible") or "").lower())
    return "|".join([description, color_key, pattern, style, brand])


def _extract_items_from_detections_with_openai(
    *,
    photo_path: Path,
    face_bbox: dict[str, float],
    face_tile_path: Path | None,
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = _settings()
    with Image.open(photo_path).convert("RGB") as image:
        img_w, img_h = image.size

    ranked = sorted(detections, key=lambda d: float(d.get("confidence") or 0), reverse=True)
    candidates = [d for d in ranked if float(d.get("confidence") or 0) >= 0.08][:24]
    if not candidates:
        return []

    schema = {
        "type": "object",
        "properties": {
            "selected_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "detection_index": {"type": "integer"},
                        "description": {"type": "string"},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "pattern": {"type": "string"},
                        "style": {"type": "string"},
                        "brand_visible": {"type": ["string", "null"]},
                        "visibility": {"type": "string", "enum": ["clear", "partial", "obscured"]},
                        "confidence": {"type": "number"},
                        "crop_quality": {"type": "string", "enum": ["good", "bad"]},
                    },
                    "required": [
                        "detection_index",
                        "description",
                        "colors",
                        "pattern",
                        "style",
                        "brand_visible",
                        "visibility",
                        "confidence",
                        "crop_quality",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selected_items"],
        "additionalProperties": False,
    }

    fbb = face_bbox
    face_desc = (
        f"face at normalized coordinates (x={fbb['left']:.3f}, y={fbb['top']:.3f}, "
        f"width={fbb['width']:.3f}, height={fbb['height']:.3f})"
    )
    detection_lines = []
    for idx, det in enumerate(candidates):
        x1, y1, x2, y2 = [int(float(v)) for v in det["bbox"]]
        detection_lines.append(
            f"{idx}: label={det.get('label','')}, score={float(det.get('confidence') or 0):.3f}, "
            f"bbox=[{x1},{y1},{x2},{y2}]"
        )

    instructions = (
        "You are post-processing YOLO-World detections for one target person.\n"
        "Important rules:\n"
        "1) Remove boxes that are not on the target with this face. Only choose detections worn by this target person.\n"
        "2) Do NOT invent, redraw, or move bounding boxes. You may only choose from listed detection indices.\n"
        "3) Exclude detections on other people or background objects (even if clothing-looking).\n"
        "4) Use visual inference across the full scene: if multiple people wear the same item and the target likely wears it too, "
        "choose the nearest visually similar detection index to the target as a proxy, and remove the rest for that item.\n"
        "5) For duplicate boxes of the same item, keep the nearest one to the target and reject the others.\n"
        "6) For accessories, especially bracelets/wristbands: if crop is tiny, blurry, or not visually recognizable, set crop_quality='bad' so it is removed.\n"
        "7) Keep only useful crops for downstream shopping lookup.\n"
        "8) `description` must be a short keyword search label (2-6 words) optimized for shopping lookup.\n"
        "9) Do NOT write sentences or context like 'worn by the target person'.\n"
        "10) Example good descriptions: 'navy quarter zip', 'black leather belt', 'white running shoes'.\n"
        f"Target hint: {face_desc}\n"
        "Detections (in order):\n"
        + "\n".join(detection_lines)
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions}]
    if face_tile_path and face_tile_path.exists():
        content.append({"type": "input_image", "image_url": _data_url_for_image(face_tile_path), "detail": "high"})
    content.append({"type": "input_image", "image_url": _data_url_for_image(photo_path), "detail": "high"})

    # Attach one crop per detection in the same order as the index list above.
    for idx, det in enumerate(candidates):
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Detection crop #{idx} (label={det.get('label','')}, "
                    f"score={float(det.get('confidence') or 0):.3f})."
                ),
            }
        )
        crop_url = _crop_data_url_for_detection(photo_path, det["bbox"], pad=0.15)
        if crop_url:
            content.append({"type": "input_image", "image_url": crop_url, "detail": "high"})

    payload = {
        "model": settings.openai_model,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "yolo_postprocess",
                "schema": schema,
                "strict": True,
            }
        },
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    raw = data.get("output_text")
    if not raw:
        for out in data.get("output", []):
            for c in out.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    raw = c["text"]
                    break
            if raw:
                break
    if not raw:
        return []

    parsed = json.loads(raw)
    selected = parsed.get("selected_items", []) or []
    face_center_x = float(face_bbox["left"]) + float(face_bbox["width"]) / 2.0
    face_center_y = float(face_bbox["top"]) + float(face_bbox["height"]) / 2.0

    normalized: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for item in selected:
        try:
            det_idx = int(item.get("detection_index"))
        except Exception:
            continue
        if det_idx < 0 or det_idx >= len(candidates) or det_idx in used_indices:
            continue
        if item.get("crop_quality") == "bad":
            continue

        det = candidates[det_idx]
        bbox = _detection_bbox_to_xywh(det["bbox"], img_w, img_h)
        box_center_x = float(bbox["x"]) + float(bbox["w"]) / 2.0
        box_center_y = float(bbox["y"]) + float(bbox["h"]) / 2.0
        distance_to_target = ((box_center_x - face_center_x) ** 2 + (box_center_y - face_center_y) ** 2) ** 0.5
        detector_label = str(det.get("label") or "").strip().lower()
        description = _compact_search_description(
            str(item.get("description") or ""),
            fallback=detector_label or "detected clothing item",
        )
        category = description

        visibility = str(item.get("visibility") or "clear")
        confidence = float(item.get("confidence") or det.get("confidence") or 0)
        confidence = max(0.0, min(1.0, confidence))

        normalized.append(
            {
                "category": category,
                "description": description,
                "colors": item.get("colors") or ["unknown"],
                "pattern": item.get("pattern") or "unknown",
                "style": item.get("style") or "casual",
                "brand_visible": item.get("brand_visible"),
                "bounding_box": bbox,
                "visibility": visibility if visibility in {"clear", "partial", "obscured"} else "clear",
                "confidence": confidence,
                "_source_label": str(det.get("label") or "").strip().lower(),
                "_distance_to_target": distance_to_target,
            }
        )
        used_indices.add(det_idx)

    # If multiple detections represent the same item, keep the nearest one to the target.
    deduped_by_item: dict[str, dict[str, Any]] = {}
    for item in normalized:
        item_key = str(item.get("_source_label") or item.get("category") or "").strip().lower()
        if not item_key:
            continue
        current = deduped_by_item.get(item_key)
        if current is None:
            deduped_by_item[item_key] = item
            continue
        current_dist = float(current.get("_distance_to_target") or 1e9)
        next_dist = float(item.get("_distance_to_target") or 1e9)
        if next_dist < current_dist:
            deduped_by_item[item_key] = item
            continue
        if abs(next_dist - current_dist) <= 1e-6 and float(item.get("confidence") or 0) > float(current.get("confidence") or 0):
            deduped_by_item[item_key] = item

    out: list[dict[str, Any]] = []
    for item in deduped_by_item.values():
        cleaned = dict(item)
        cleaned.pop("_source_label", None)
        cleaned.pop("_distance_to_target", None)
        out.append(cleaned)
    return out


def _compact_search_description(raw: str, *, fallback: str) -> str:
    text = str(raw or "").strip().lower().replace("-", " ")
    text = re.sub(r"\bquarter\s+zip\s+pullover\b", "quarter zip", text)
    text = re.sub(r"\bworn by (the )?target person\b", " ", text)
    text = re.sub(r"\btarget person\b", " ", text)
    text = re.sub(r"\bworn by\b", " ", text)
    text = re.sub(r"\bwearing\b", " ", text)
    text = re.sub(r"\bworn\b", " ", text)
    text = re.sub(r"\bthe wearer\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in re.split(r"\s+", text) if w]
    stop = {"the", "a", "an", "for", "of", "on", "with", "in", "to", "by", "this", "that"}
    words = [w for w in words if w not in stop]
    if not words:
        fb = str(fallback or "").strip().lower().replace("-", " ")
        fb = re.sub(r"[^a-z0-9\s]", " ", fb)
        words = [w for w in re.split(r"\s+", fb) if w]
    if len(words) > 6:
        words = words[:6]
    return " ".join(words) if words else "clothing item"


def _detection_bbox_to_xywh(bbox: list[float], img_w: int, img_h: int) -> dict[str, float]:
    x1 = max(0.0, min(float(img_w), float(bbox[0])))
    y1 = max(0.0, min(float(img_h), float(bbox[1])))
    x2 = max(x1 + 1.0, min(float(img_w), float(bbox[2])))
    y2 = max(y1 + 1.0, min(float(img_h), float(bbox[3])))
    return {
        "x": max(0.0, min(1.0, x1 / img_w)),
        "y": max(0.0, min(1.0, y1 / img_h)),
        "w": max(0.01, min(1.0, (x2 - x1) / img_w)),
        "h": max(0.01, min(1.0, (y2 - y1) / img_h)),
    }

def _crop_data_url_for_detection(photo_path: Path, bbox_xyxy: list[float], pad: float = 0.1) -> str | None:
    try:
        with Image.open(photo_path).convert("RGB") as image:
            width, height = image.size
            x1 = max(0, int(float(bbox_xyxy[0])))
            y1 = max(0, int(float(bbox_xyxy[1])))
            x2 = min(width, int(float(bbox_xyxy[2])))
            y2 = min(height, int(float(bbox_xyxy[3])))
            if x2 <= x1 or y2 <= y1:
                return None

            pad_x = int((x2 - x1) * pad)
            pad_y = int((y2 - y1) * pad)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(width, x2 + pad_x)
            cy2 = min(height, y2 + pad_y)
            crop = image.crop((cx1, cy1, cx2, cy2))

            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _fallback_items_from_detections(
    detections: list[dict[str, Any]],
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for det in sorted(detections, key=lambda d: float(d.get("confidence") or 0), reverse=True):
        conf = float(det.get("confidence") or 0)
        if conf < 0.18:
            continue
        detector_label = str(det.get("label") or "").strip().lower()
        if not detector_label:
            continue
        if detector_label in seen_labels:
            continue
        out.append(
            {
                "category": detector_label,
                "description": detector_label,
                "colors": ["unknown"],
                "pattern": "unknown",
                "style": "casual",
                "brand_visible": None,
                "bounding_box": _detection_bbox_to_xywh(det["bbox"], img_w, img_h),
                "visibility": "partial",
                "confidence": min(0.95, conf),
            }
        )
        seen_labels.add(detector_label)
        if len(out) >= 6:
            break
    return out


def _detect_objects(
    image_url: str,
    *,
    class_names: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    yolo_debug = debug if debug is not None else {}
    settings = _settings()
    if not settings.replicate_api_token:
        yolo_debug["status"] = "skipped"
        yolo_debug["error"] = "missing_replicate_api_token"
        yolo_debug["detection_count"] = 0
        return []
    if not image_url:
        yolo_debug["status"] = "skipped"
        yolo_debug["error"] = "missing_image_url"
        yolo_debug["detection_count"] = 0
        return []

    chosen_classes = _normalize_yolo_class_names(class_names or [], max_items=60)
    if not chosen_classes:
        chosen_classes = YOLO_WORLD_CLASS_NAMES
    yolo_debug["requested_class_count"] = len(chosen_classes)
    yolo_debug["requested_classes"] = chosen_classes[:60]
    yolo_debug["status"] = "running"

    try:
        version = _replicate_yolo_world_version(settings.replicate_api_token)
        if not version:
            yolo_debug["status"] = "failed"
            yolo_debug["error"] = "version_lookup_failed"
            yolo_debug["detection_count"] = 0
            return []
        yolo_debug["model_version"] = version

        start_data: dict[str, Any] | None = None
        start_attempts: list[dict[str, Any]] = []
        for wait_s in (0, 8, 16):
            if wait_s:
                time.sleep(wait_s)
            start = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {settings.replicate_api_token}",
                    "Content-Type": "application/json",
                    "Prefer": "wait=10",
                },
                json={
                    "version": version,
                    "input": {
                        "input_media": image_url,
                        "class_names": ",".join(chosen_classes),
                        "max_num_boxes": 120,
                        "nms_thr": 0.6,
                        "score_thr": 0.1,
                        "return_json": True,
                    },
                },
                timeout=40,
            )
            start_attempts.append({"wait_seconds": wait_s, "http_status": int(start.status_code)})
            if start.status_code == 429:
                continue
            if start.status_code >= 300:
                yolo_debug["status"] = "failed"
                yolo_debug["start_attempts"] = start_attempts
                yolo_debug["error"] = f"start_http_{int(start.status_code)}"
                yolo_debug["error_body"] = (start.text or "")[:400]
                yolo_debug["detection_count"] = 0
                return []
            start_data = start.json()
            break

        if not start_data:
            yolo_debug["status"] = "failed"
            yolo_debug["start_attempts"] = start_attempts
            yolo_debug["error"] = "start_failed_no_response"
            yolo_debug["detection_count"] = 0
            return []
        yolo_debug["start_attempts"] = start_attempts

        data = start_data
        yolo_debug["prediction_status"] = str(data.get("status") or "")
        if data.get("status") not in {"succeeded", "failed", "canceled"} and data.get("urls", {}).get("get"):
            poll_url = data["urls"]["get"]
            deadline = time.time() + 30
            poll_count = 0
            while time.time() < deadline:
                time.sleep(1)
                poll = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {settings.replicate_api_token}"},
                    timeout=30,
                )
                if poll.status_code >= 300:
                    yolo_debug["poll_error_http_status"] = int(poll.status_code)
                    break
                data = poll.json()
                poll_count += 1
                yolo_debug["prediction_status"] = str(data.get("status") or "")
                if data.get("status") in {"succeeded", "failed", "canceled"}:
                    break
            yolo_debug["poll_count"] = poll_count

        if data.get("status") != "succeeded":
            yolo_debug["status"] = "failed"
            yolo_debug["prediction_status"] = str(data.get("status") or "")
            yolo_debug["error"] = str(data.get("error") or "prediction_not_succeeded")
            yolo_debug["detection_count"] = 0
            return []

        dets: list[dict[str, Any]] = []
        output = data.get("output")
        if isinstance(output, dict):
            raw_json = output.get("json_str")
            if isinstance(raw_json, str) and raw_json.strip():
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict):
                    for value in parsed.values():
                        if isinstance(value, dict):
                            dets.append(value)
                elif isinstance(parsed, list):
                    for value in parsed:
                        if isinstance(value, dict):
                            dets.append(value)
            elif isinstance(output.get("detections"), list):
                # Compatibility with detectors that emit detections directly
                dets = output.get("detections") or []

        out: list[dict[str, Any]] = []
        for det in dets:
            bbox: list[float] | None = None
            if isinstance(det.get("bbox"), list) and len(det.get("bbox") or []) == 4:
                raw_bbox = det.get("bbox") or []
                bbox = [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]
            elif all(k in det for k in ("x0", "y0", "x1", "y1")):
                bbox = [float(det["x0"]), float(det["y0"]), float(det["x1"]), float(det["y1"])]

            if not bbox:
                continue

            label = str(det.get("label") or det.get("cls") or "").strip().lower()
            confidence = float(det.get("confidence") or det.get("score") or 0)
            if not label:
                continue

            out.append({"bbox": bbox, "label": label, "confidence": confidence})

        out.sort(key=lambda d: float(d.get("confidence") or 0), reverse=True)
        yolo_debug["status"] = "succeeded"
        yolo_debug["prediction_status"] = "succeeded"
        yolo_debug["detection_count"] = len(out)
        return out
    except Exception as exc:
        yolo_debug["status"] = "exception"
        yolo_debug["error"] = str(exc)
        yolo_debug["detection_count"] = 0
        return []


def _replicate_yolo_world_version(token: str) -> str | None:
    global _REPLICATE_YOLO_VERSION
    if _REPLICATE_YOLO_VERSION:
        return _REPLICATE_YOLO_VERSION

    with _REPLICATE_VERSION_LOCK:
        if _REPLICATE_YOLO_VERSION:
            return _REPLICATE_YOLO_VERSION

        try:
            res = requests.get(
                "https://api.replicate.com/v1/models/franz-biz/yolo-world-xl",
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            if res.status_code >= 300:
                return None
            data = res.json()
            version = ((data.get("latest_version") or {}).get("id") or "").strip()
            if not version:
                return None
            _REPLICATE_YOLO_VERSION = version
            return version
        except Exception:
            return None


def _find_exact_match(public_image_url: str, description: str) -> list[dict[str, Any]]:
    settings = _settings()
    params: dict[str, Any] = {
        "engine": "google_lens",
        "type": "all",
        "url": public_image_url,
        "api_key": settings.serpapi_key,
    }
    if description:
        params["q"] = description

    try:
        res = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        if res.status_code >= 300:
            return []
        data = res.json()
        matches = data.get("visual_matches") or []
        out: list[dict[str, Any]] = []
        for match in matches:
            price = _parse_price(match.get("price"))
            source = match.get("source")
            if price is None or not source:
                continue
            out.append(
                {
                    "title": match.get("title") or "",
                    "source": source,
                    "price": price,
                    "link": match.get("link") or "",
                    "thumbnail": match.get("thumbnail") or "",
                }
            )
            if len(out) >= 5:
                break
        return out
    except Exception:
        return []


def _find_similar_products(description: str) -> list[dict[str, Any]]:
    settings = _settings()
    params = {
        "engine": "google_shopping",
        "q": description,
        "api_key": settings.serpapi_key,
    }
    try:
        res = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        if res.status_code >= 300:
            return []
        data = res.json()
        results = data.get("shopping_results") or []
        out: list[dict[str, Any]] = []
        for result in results[:3]:
            link = (
                result.get("link")
                or result.get("product_link")
                or result.get("serpapi_product_api")
                or ""
            )
            thumbnail = result.get("thumbnail") or result.get("serpapi_thumbnail") or ""
            out.append(
                {
                    "title": result.get("title") or "",
                    "source": result.get("source") or "",
                    "price": result.get("price") or None,
                    "link": link,
                    "thumbnail": thumbnail,
                }
            )
        return out
    except Exception:
        return []


def _find_phia_products_for_fallback(
    *,
    scraped_name: str,
    crop_external_url: str | None = None,
) -> list[dict[str, Any]]:
    settings = _settings()
    cleaned_name = str(scraped_name or "").strip()
    cleaned_url = str(crop_external_url or "").strip()
    if not cleaned_name and not cleaned_url:
        return []
    try:
        return phia.products_google_shopping(
            settings=settings,
            scraped_name=cleaned_name,
            image_urls=[cleaned_url] if cleaned_url else None,
            limit=3,
        )
    except Exception:
        return []


def _parse_price(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, str):
            return value
        num = raw.get("extracted_value")
        if isinstance(num, (int, float)):
            return f"${num}"
    return None


def _rank_candidates(
    *,
    photo_path: Path,
    face_tile_path: Path | None,
    crop_path: Path | None,
    description: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    settings = _settings()
    if not settings.openai_api_key:
        return None
    if not crop_path or not crop_path.exists() or not photo_path.exists() or not candidates:
        return None

    trimmed = candidates[:RANK_MAX_CANDIDATES]
    catalog = "\n".join(
        f"{idx}. {(c.get('title') or '(no title)')}"
        f"{' — ' + str(c.get('price')) if c.get('price') else ''}"
        f"{' @ ' + str(c.get('source')) if c.get('source') else ''}"
        for idx, c in enumerate(trimmed)
    )

    instructions = (
        "You are matching a clothing item a specific person is wearing to candidate products. "
        "Choose the best candidate and assign exact/similar/none. "
        f"Item description: {description}.\nCandidates:\n{catalog}"
    )

    schema = {
        "type": "object",
        "properties": {
            "best_index": {"type": ["integer", "null"]},
            "tier": {"type": "string", "enum": ["exact", "similar", "none"]},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["best_index", "tier", "confidence", "reasoning"],
        "additionalProperties": False,
    }

    content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions}]
    if face_tile_path and face_tile_path.exists():
        content.append({"type": "input_image", "image_url": _data_url_for_image(face_tile_path), "detail": "high"})
    content.append({"type": "input_image", "image_url": _data_url_for_image(photo_path), "detail": "high"})
    content.append({"type": "input_image", "image_url": _data_url_for_image(crop_path), "detail": "high"})
    for cand in trimmed:
        thumb = cand.get("thumbnail")
        if thumb:
            content.append({"type": "input_image", "image_url": thumb, "detail": "high"})

    payload = {
        "model": settings.openai_model,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "candidate_rank",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        res = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if res.status_code >= 300:
            return None

        data = res.json()
        raw = data.get("output_text")
        if not raw:
            for out in data.get("output", []):
                for c in out.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        raw = c["text"]
                        break
                if raw:
                    break
        if not raw:
            return None

        parsed = json.loads(raw)
        return {
            "best_index": parsed.get("best_index"),
            "tier": parsed.get("tier"),
            "confidence": float(parsed.get("confidence") or 0),
            "reasoning": parsed.get("reasoning") or "",
        }
    except Exception:
        return None


def _data_url_for_image(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _save_face_tile(
    photo_path: Path,
    job_id: str,
    photo_id: str,
    bbox: dict[str, float],
) -> str:
    rel = f"face_tiles/{job_id}/{photo_id}.jpg"
    out_path = _settings().media_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _crop_to_path(photo_path=photo_path, out_path=out_path, bbox=bbox, pad=0.4, xywh=False)
    return rel


def _save_item_crop(
    photo_path: Path,
    job_id: str,
    photo_id: str,
    bbox: dict[str, float],
) -> str:
    item_id = str(uuid.uuid4())
    rel = f"clothing_crops/{job_id}/{photo_id}/{item_id}.jpg"
    out_path = _settings().media_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _crop_to_path(photo_path=photo_path, out_path=out_path, bbox=bbox, pad=0.1, xywh=True)
    return rel


def _crop_to_path(
    *,
    photo_path: Path,
    out_path: Path,
    bbox: dict[str, float],
    pad: float,
    xywh: bool,
) -> None:
    with Image.open(photo_path).convert("RGB") as image:
        width, height = image.size

        if xywh:
            left = float(bbox.get("x") or 0)
            top = float(bbox.get("y") or 0)
            box_w = float(bbox.get("w") or 0)
            box_h = float(bbox.get("h") or 0)
        else:
            left = float(bbox.get("left") or 0)
            top = float(bbox.get("top") or 0)
            box_w = float(bbox.get("width") or 0)
            box_h = float(bbox.get("height") or 0)

        x1 = max(0, int((left - box_w * pad) * width))
        y1 = max(0, int((top - box_h * pad) * height))
        x2 = min(width, int((left + box_w * (1 + pad)) * width))
        y2 = min(height, int((top + box_h * (1 + pad)) * height))

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height

        crop = image.crop((x1, y1, x2, y2))
        crop.save(out_path, format="JPEG", quality=88)
