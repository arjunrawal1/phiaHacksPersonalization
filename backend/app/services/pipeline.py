from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.core.config import Settings
from app.services import db

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

CATEGORY_QUERY_TERMS: dict[str, list[str]] = {
    "top": ["shirt", "top", "sweater", "t-shirt", "blouse"],
    "bottom": ["pants", "trousers", "shorts", "skirt", "jeans"],
    "dress": ["dress", "gown"],
    "outerwear": ["jacket", "coat", "hoodie", "blazer"],
    "shoes": ["shoes", "sneakers", "boots"],
    "hat": ["hat", "cap", "beanie"],
    "bag": ["bag", "backpack", "purse", "handbag"],
    "accessory": ["watch", "bracelet", "necklace", "tie", "sunglasses", "belt"],
}

_RUNNING_FACE_JOBS: set[str] = set()
_RUNNING_CLOTHING_RUNS: set[str] = set()
_RUNNING_LOOKUPS: set[str] = set()
_RUN_LOCK = threading.Lock()
_LOOKUP_SEMAPHORE: threading.Semaphore | None = None
_SETTINGS: Settings | None = None
_REPLICATE_VERSION: str | None = None
_REPLICATE_VERSION_LOCK = threading.Lock()


def configure(settings: Settings) -> None:
    global _SETTINGS, _LOOKUP_SEMAPHORE
    _SETTINGS = settings
    _LOOKUP_SEMAPHORE = threading.Semaphore(max(1, settings.lookup_concurrency))
    (settings.media_dir / "photos").mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "face_tiles").mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "clothing_crops").mkdir(parents=True, exist_ok=True)


def _settings() -> Settings:
    if _SETTINGS is None:
        raise RuntimeError("Pipeline is not configured")
    return _SETTINGS


def media_rel_to_url(media_base_url: str, relative_path: str) -> str:
    safe = relative_path.lstrip("/").replace("\\", "/")
    return f"{media_base_url.rstrip('/')}/{safe}"


def _external_media_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    settings = _settings()
    base = settings.public_media_base_url.strip()
    if not base:
        return None
    safe = relative_path.lstrip("/").replace("\\", "/")
    return f"{base.rstrip('/')}/media/{safe}"


def build_job_detail(job_id: str, media_base_url: str) -> dict[str, Any] | None:
    job = db.get_job(job_id)
    if not job:
        return None

    photos = db.list_photos(job_id)
    photo_by_id = {p["id"]: p for p in photos}
    clusters = db.list_face_clusters(job_id)
    selected = db.get_selected_cluster(job_id)
    items = db.list_clothing_items(job_id)

    photos_out = [
        {
            "id": p["id"],
            "url": media_rel_to_url(media_base_url, p["relative_path"]),
            "width": p["width"],
            "height": p["height"],
        }
        for p in photos
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
            "best_match": i.get("best_match"),
            "best_match_confidence": i.get("best_match_confidence") or 0,
        }
        for i in items
    ]

    return {
        **job,
        "selected_cluster_id": selected["cluster_id"] if selected else None,
        "photos": photos_out,
        "clusters": clusters_out,
        "items": items_out,
    }


def start_face_analysis(job_id: str) -> None:
    with _RUN_LOCK:
        if job_id in _RUNNING_FACE_JOBS:
            return
        _RUNNING_FACE_JOBS.add(job_id)

    def runner() -> None:
        try:
            _analyze_faces_worker(job_id)
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


def _analyze_faces_worker(job_id: str) -> None:
    try:
        print(TAG, "face analysis start", job_id)
        db.update_job(job_id, status="analyzing_faces", error=None)
        db.clear_face_analysis(job_id)

        photos = db.list_photos(job_id)
        if not photos:
            db.update_job(job_id, status="failed", error="No photos uploaded")
            return

        clusters = _analyze_faces_with_rekognition(job_id, photos)
        if clusters is None:
            clusters = _analyze_faces_local(job_id, photos)

        if not clusters:
            db.update_job(job_id, status="failed", error="No faces detected above threshold")
            return

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

        db.update_job(job_id, status="awaiting_face_pick", error=None)
        print(TAG, "face analysis done", job_id, "clusters", len(clusters))
    except Exception as exc:  # pragma: no cover
        db.update_job(job_id, status="failed", error=f"Face analysis failed: {exc}")


def _analyze_faces_with_rekognition(
    job_id: str,
    photos: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    settings = _settings()
    if not (
        boto3 is not None
        and settings.aws_access_key_id
        and settings.aws_secret_access_key
        and settings.aws_region
    ):
        return None

    try:
        client = boto3.client(
            "rekognition",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    except Exception:
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

        detections = db.list_face_detections_for_cluster(cluster_id)
        if not detections:
            db.update_job(job_id, status="done", error=None)
            return

        settings = _settings()
        photos = {p["id"]: p for p in db.list_photos(job_id)}

        bbox_by_photo: dict[str, dict[str, float]] = {}
        for det in detections:
            if det["photo_id"] not in bbox_by_photo:
                bbox_by_photo[det["photo_id"]] = det["bbox"]

        item_ids: list[str] = []
        entries = list(bbox_by_photo.items())

        def process_one(entry: tuple[str, dict[str, float]]) -> list[str]:
            photo_id, face_bbox = entry
            photo = photos.get(photo_id)
            if not photo:
                return []

            photo_path = settings.media_dir / photo["relative_path"]
            face_tile_rel = _save_face_tile(photo_path, job_id, photo_id, face_bbox)
            face_tile_path = settings.media_dir / face_tile_rel

            items = _extract_items_from_photo(photo_path, face_bbox, face_tile_path)
            visible = [
                i for i in items if i.get("visibility", "clear") != "obscured" and float(i.get("confidence", 0)) >= 0.4
            ]
            if not visible:
                return []

            refined = visible
            external_photo_url = _external_media_url(photo["relative_path"])
            if photo.get("width") and photo.get("height") and external_photo_url:
                dino = _detect_objects(external_photo_url, _build_grounding_query(visible))
                if dino:
                    used: set[int] = set()
                    refined_next: list[dict[str, Any]] = []
                    for item in visible:
                        new_bbox = _find_refined_bbox(
                            item,
                            dino,
                            used,
                            int(photo["width"]),
                            int(photo["height"]),
                        )
                        if new_bbox:
                            merged = dict(item)
                            merged["bounding_box"] = new_bbox
                            refined_next.append(merged)
                        else:
                            refined_next.append(item)
                    refined = refined_next

            inserted: list[str] = []
            for item in refined:
                crop_rel = _save_item_crop(
                    photo_path=photo_path,
                    job_id=job_id,
                    photo_id=photo_id,
                    bbox=item["bounding_box"],
                )

                item_id = db.insert_clothing_item(
                    job_id=job_id,
                    photo_id=photo_id,
                    category=item.get("category", "top"),
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
                    best_match=None,
                    best_match_confidence=0,
                )
                inserted.append(item_id)

            return inserted

        concurrency = max(1, settings.photo_concurrency)
        if concurrency == 1:
            for entry in entries:
                item_ids.extend(process_one(entry))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for ids in pool.map(process_one, entries):
                    item_ids.extend(ids)

        db.update_job(job_id, status="done", error=None)
        for item_id in item_ids:
            start_lookup_item(item_id)

        print(TAG, "clothing extraction done", job_id, "items", len(item_ids))
    except Exception as exc:  # pragma: no cover
        db.update_job(job_id, status="failed", error=f"Clothing extraction failed: {exc}")


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
                crop_path = settings.media_dir / (crop_rel or "")

                ranked = _rank_candidates(
                    photo_path=photo_path,
                    face_tile_path=face_tile_path if face_tile_path.exists() else None,
                    crop_path=crop_path if crop_path.exists() else None,
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

        if final_tier == "pending":
            final_tier = "generic"

        db.update_clothing_item(
            item_id=item_id,
            tier=final_tier,
            exact_matches=exact_matches,
            similar_products=similar_products,
            best_match=best_match,
            best_match_confidence=best_confidence,
            crop_path=crop_rel,
        )
    except Exception as exc:  # pragma: no cover
        print(TAG, "lookup failed", item_id, exc)


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


def _extract_items_from_photo(
    photo_path: Path,
    face_bbox: dict[str, float],
    face_tile_path: Path | None,
) -> list[dict[str, Any]]:
    settings = _settings()
    if settings.openai_api_key:
        try:
            extracted = _extract_items_with_openai(photo_path, face_bbox, face_tile_path)
            if extracted:
                return extracted
        except Exception as exc:
            print(TAG, "openai extraction failed", exc)
    return _fallback_items(face_bbox)


def _extract_items_with_openai(
    photo_path: Path,
    face_bbox: dict[str, float],
    face_tile_path: Path | None,
) -> list[dict[str, Any]]:
    settings = _settings()

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["top", "bottom", "dress", "outerwear", "shoes", "hat", "bag", "accessory"],
                        },
                        "description": {"type": "string"},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "pattern": {"type": "string"},
                        "style": {"type": "string"},
                        "brand_visible": {"type": ["string", "null"]},
                        "bounding_box": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "w": {"type": "number"},
                                "h": {"type": "number"},
                            },
                            "required": ["x", "y", "w", "h"],
                            "additionalProperties": False,
                        },
                        "visibility": {"type": "string", "enum": ["clear", "partial", "obscured"]},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "category",
                        "description",
                        "colors",
                        "pattern",
                        "style",
                        "brand_visible",
                        "bounding_box",
                        "visibility",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    fbb = face_bbox
    face_desc = (
        f"face at normalized coordinates (x={fbb['left']:.3f}, y={fbb['top']:.3f}, "
        f"width={fbb['width']:.3f}, height={fbb['height']:.3f})"
    )

    if face_tile_path and face_tile_path.exists():
        instructions = (
            "You are analyzing a photo to identify clothing items worn by a specific person. "
            "IMAGE 1 is a close-up crop of the target person's face. IMAGE 2 is the full photo. "
            f"Their {face_desc}; use that as a hint, but trust face matching from IMAGE 1. "
            "Identify every visible clothing or accessory item worn by this person only. "
            "Return short product-title-style descriptions."
        )
    else:
        instructions = (
            "You are analyzing a photo to identify clothing items worn by a specific person. "
            f"The target person has a {face_desc}. "
            "Identify visible clothing/accessories worn by this person only and return short product-title-style descriptions."
        )

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
                "name": "outfit_analysis",
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
        timeout=90,
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
    items = parsed.get("items", [])
    normalized: list[dict[str, Any]] = []
    for item in items:
        box = item.get("bounding_box") or {}
        normalized.append(
            {
                "category": item.get("category", "top"),
                "description": item.get("description", "Unlabeled clothing item"),
                "colors": item.get("colors", []),
                "pattern": item.get("pattern", ""),
                "style": item.get("style", ""),
                "brand_visible": item.get("brand_visible"),
                "bounding_box": {
                    "x": float(max(0, min(1, box.get("x", 0)))),
                    "y": float(max(0, min(1, box.get("y", 0)))),
                    "w": float(max(0.05, min(1, box.get("w", 0.2)))),
                    "h": float(max(0.05, min(1, box.get("h", 0.2)))),
                },
                "visibility": item.get("visibility", "clear"),
                "confidence": float(item.get("confidence", 0.5)),
            }
        )
    return normalized


def _fallback_items(face_bbox: dict[str, float]) -> list[dict[str, Any]]:
    top_x = max(0.02, face_bbox["left"] - face_bbox["width"] * 0.2)
    top_y = min(0.9, face_bbox["top"] + face_bbox["height"] * 0.8)
    top_w = min(0.96 - top_x, face_bbox["width"] * 1.5)
    top_h = min(0.95 - top_y, face_bbox["height"] * 1.6)

    bottom_x = max(0.02, face_bbox["left"] - face_bbox["width"] * 0.15)
    bottom_y = min(0.94, top_y + top_h * 0.9)
    bottom_w = min(0.96 - bottom_x, face_bbox["width"] * 1.4)
    bottom_h = min(0.95 - bottom_y, face_bbox["height"] * 1.3)

    return [
        {
            "category": "top",
            "description": "Detected upper-body clothing",
            "colors": ["unknown"],
            "pattern": "unknown",
            "style": "casual",
            "brand_visible": None,
            "bounding_box": {"x": top_x, "y": top_y, "w": top_w, "h": top_h},
            "visibility": "partial",
            "confidence": 0.45,
        },
        {
            "category": "bottom",
            "description": "Detected lower-body clothing",
            "colors": ["unknown"],
            "pattern": "unknown",
            "style": "casual",
            "brand_visible": None,
            "bounding_box": {"x": bottom_x, "y": bottom_y, "w": bottom_w, "h": bottom_h},
            "visibility": "partial",
            "confidence": 0.4,
        },
    ]


def _build_grounding_query(items: list[dict[str, Any]]) -> str:
    terms: set[str] = set()
    for item in items:
        for t in CATEGORY_QUERY_TERMS.get(str(item.get("category")), []):
            terms.add(t)
    return ", ".join(sorted(terms))


def _detect_objects(image_url: str, query: str) -> list[dict[str, Any]]:
    settings = _settings()
    if not settings.replicate_api_token or not image_url:
        return []

    try:
        version = _replicate_grounding_dino_version(settings.replicate_api_token)
        if not version:
            return []

        start_data: dict[str, Any] | None = None
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
                        "image": image_url,
                        "query": query,
                        "box_threshold": 0.25,
                        "text_threshold": 0.2,
                        "show_visualisation": False,
                    },
                },
                timeout=40,
            )
            if start.status_code == 429:
                continue
            if start.status_code >= 300:
                return []
            start_data = start.json()
            break

        if not start_data:
            return []

        data = start_data
        if data.get("status") not in {"succeeded", "failed", "canceled"} and data.get("urls", {}).get("get"):
            poll_url = data["urls"]["get"]
            deadline = time.time() + 30
            while time.time() < deadline:
                time.sleep(1)
                poll = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {settings.replicate_api_token}"},
                    timeout=30,
                )
                if poll.status_code >= 300:
                    break
                data = poll.json()
                if data.get("status") in {"succeeded", "failed", "canceled"}:
                    break

        if data.get("status") != "succeeded":
            return []
        dets = (data.get("output") or {}).get("detections") or []
        out: list[dict[str, Any]] = []
        for det in dets:
            bbox = det.get("bbox") or []
            if len(bbox) != 4:
                continue
            out.append(
                {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "label": str(det.get("label") or ""),
                    "confidence": float(det.get("confidence") or 0),
                }
            )
        return out
    except Exception:
        return []


def _replicate_grounding_dino_version(token: str) -> str | None:
    global _REPLICATE_VERSION
    if _REPLICATE_VERSION:
        return _REPLICATE_VERSION

    with _REPLICATE_VERSION_LOCK:
        if _REPLICATE_VERSION:
            return _REPLICATE_VERSION

        try:
            res = requests.get(
                "https://api.replicate.com/v1/models/adirik/grounding-dino",
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            if res.status_code >= 300:
                return None
            data = res.json()
            version = ((data.get("latest_version") or {}).get("id") or "").strip()
            if not version:
                return None
            _REPLICATE_VERSION = version
            return version
        except Exception:
            return None


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    a_area = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    b_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0


def _find_refined_bbox(
    item: dict[str, Any],
    detections: list[dict[str, Any]],
    used_indices: set[int],
    img_w: int,
    img_h: int,
) -> dict[str, float] | None:
    category = str(item.get("category"))
    terms = [t.lower() for t in CATEGORY_QUERY_TERMS.get(category, [])]
    if not terms:
        return None

    candidates: list[tuple[int, dict[str, Any]]] = []
    for idx, det in enumerate(detections):
        if idx in used_indices:
            continue
        label = str(det.get("label") or "").lower()
        if any(t in label or label in t for t in terms):
            candidates.append((idx, det))

    if not candidates:
        return None

    box = item.get("bounding_box") or {}
    gpt_box = [
        float(box.get("x", 0)) * img_w,
        float(box.get("y", 0)) * img_h,
        float(box.get("x", 0) + box.get("w", 0.2)) * img_w,
        float(box.get("y", 0) + box.get("h", 0.2)) * img_h,
    ]

    best_idx: int | None = None
    best_det: dict[str, Any] | None = None
    best_score = -1.0
    for idx, det in candidates:
        det_box = det["bbox"]
        score = _iou(det_box, gpt_box) + float(det.get("confidence") or 0) * 0.1
        if score > best_score:
            best_score = score
            best_idx = idx
            best_det = det

    if best_idx is None or best_det is None or best_score < 0.1:
        return None

    used_indices.add(best_idx)
    x1, y1, x2, y2 = best_det["bbox"]
    return {
        "x": max(0.0, min(1.0, x1 / img_w)),
        "y": max(0.0, min(1.0, y1 / img_h)),
        "w": max(0.01, min(1.0, (x2 - x1) / img_w)),
        "h": max(0.01, min(1.0, (y2 - y1) / img_h)),
    }


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
            out.append(
                {
                    "title": result.get("title") or "",
                    "source": result.get("source") or "",
                    "price": result.get("price") or None,
                    "link": result.get("link") or "",
                    "thumbnail": result.get("thumbnail") or "",
                }
            )
        return out
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
