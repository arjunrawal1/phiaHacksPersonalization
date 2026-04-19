from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.config import get_settings
from app.models.pipeline import (
    AutoStylingRunsResponse,
    BackfillFavoritesRequest,
    BackfillFavoritesResponse,
    CreateJobResponse,
    EvaluateVariantsRequest,
    EvaluateVariantsResponse,
    ItemModelRenderRequest,
    JobDetail,
    JobSummary,
    ModelRenderResponse,
    PersonalizationSummaryResponse,
    SelectClusterRequest,
    SelectClusterResponse,
    SimulateMobileFeedRequest,
)
from app.services import db, model_render, personalization, phia, pipeline
from app.services.image_utils import decode_to_rgb

router = APIRouter()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    return out


def _to_epoch_ms(value: Any) -> int | None:
    try:
        raw = int(float(value))
    except Exception:
        return None
    if raw <= 0:
        return None
    # If client accidentally sent seconds, normalize to ms.
    if raw < 100_000_000_000:
        raw *= 1000
    return raw


def _parse_photo_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    location = parsed.get("location")
    lat_raw = parsed.get("latitude")
    lon_raw = parsed.get("longitude")
    if isinstance(location, dict):
        lat_raw = lat_raw if lat_raw is not None else location.get("latitude")
        lon_raw = lon_raw if lon_raw is not None else location.get("longitude")

    lat = _to_float(lat_raw)
    lon = _to_float(lon_raw)
    if lat is not None and (lat < -90 or lat > 90):
        lat = None
    if lon is not None and (lon < -180 or lon > 180):
        lon = None

    captured_at_epoch_ms = _to_epoch_ms(
        parsed.get("captured_at_epoch_ms")
        or parsed.get("creation_time_ms")
        or parsed.get("creationTime")
        or parsed.get("creation_time")
    )
    captured_at = str(
        parsed.get("captured_at")
        or parsed.get("captured_at_iso")
        or parsed.get("capturedAt")
        or ""
    ).strip() or None
    if not captured_at and captured_at_epoch_ms:
        captured_at = datetime.fromtimestamp(captured_at_epoch_ms / 1000, tz=UTC).isoformat()

    location_source = str(parsed.get("location_source") or "").strip()
    if not location_source and lat is not None and lon is not None:
        location_source = "media_library"

    return {
        "captured_at": captured_at,
        "captured_at_epoch_ms": captured_at_epoch_ms,
        "latitude": lat,
        "longitude": lon,
        "location_source": location_source or None,
        "metadata_json": json.dumps(parsed),
    }


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "phia backend is running"}


@router.get("/api/jobs", response_model=list[JobSummary], tags=["pipeline"])
def list_jobs() -> list[dict[str, object]]:
    settings = get_settings()
    return db.list_jobs(limit=max(1, settings.recent_limit))


@router.get("/api/jobs/{job_id}", response_model=JobDetail, tags=["pipeline"])
def get_job(job_id: str, request: Request) -> dict[str, object]:
    detail = pipeline.build_job_detail(job_id, _media_base_url(request))
    if not detail:
        raise HTTPException(status_code=404, detail="Job not found")
    return detail


@router.get(
    "/api/personalization",
    response_model=PersonalizationSummaryResponse,
    tags=["personalization"],
)
def get_personalization(job_id: str, force: bool = False) -> dict[str, object]:
    summary = personalization.build_personalization_summary(job_id, force=force)
    if not summary:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


@router.get(
    "/api/styling/auto-runs",
    response_model=AutoStylingRunsResponse,
    tags=["styling"],
)
def get_styling_auto_runs(job_id: str, request: Request) -> dict[str, object]:
    detail = pipeline.build_styling_auto_runs_detail(job_id, _media_base_url(request))
    if not detail:
        raise HTTPException(status_code=404, detail="Job not found")
    return detail


@router.post("/api/jobs", response_model=CreateJobResponse, tags=["pipeline"])
async def create_job(
    request: Request,
    photos: list[UploadFile] = File(...),
    photo_metadata: list[str] = Form(default=[]),
    user_name: str | None = Form(None),
) -> dict[str, object]:
    if not photos:
        raise HTTPException(status_code=400, detail="No photos uploaded")

    settings = get_settings()
    expected_count = len(photos)
    job_id = db.create_job(photo_count=0)
    clean_user_name = (user_name or "").strip()
    db.patch_job_debug(
        job_id,
        patch={
            "upload": {
                "expected_count": expected_count,
                "saved_count": 0,
                "started_at": db.utc_now_iso(),
                "user_name": clean_user_name or None,
            },
            "stages": {"upload": "running"},
        },
        event={"type": "upload_started", "message": f"Upload started for {expected_count} photos"},
    )
    saved_count = 0

    for idx, upload in enumerate(photos):
        payload = await upload.read()
        if not payload:
            continue

        try:
            metadata = _parse_photo_metadata(photo_metadata[idx] if idx < len(photo_metadata) else None)
            image = decode_to_rgb(payload)
            width, height = image.size
            rel_path = f"photos/{job_id}/{saved_count:04d}.jpg"
            abs_path = settings.media_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(abs_path, format="JPEG", quality=90)
            image.close()
            db.create_photo(
                job_id=job_id,
                relative_path=rel_path,
                width=width,
                height=height,
                captured_at=metadata.get("captured_at"),
                captured_at_epoch_ms=metadata.get("captured_at_epoch_ms"),
                latitude=metadata.get("latitude"),
                longitude=metadata.get("longitude"),
                location_source=metadata.get("location_source"),
                metadata_json=metadata.get("metadata_json"),
            )
            saved_count += 1
            db.set_job_photo_count(job_id, saved_count)
            db.patch_job_debug(
                job_id,
                patch={"upload": {"saved_count": saved_count}},
                event={
                    "type": "photo_saved",
                    "message": f"Saved photo {saved_count}/{expected_count}",
                    "data": {"relative_path": rel_path, "width": width, "height": height},
                },
            )
        except Exception:
            continue

    if saved_count == 0:
        db.update_job(job_id, status="failed", error="No valid image files found")
        db.patch_job_debug(
            job_id,
            patch={"stages": {"upload": "failed"}},
            event={"type": "upload_failed", "message": "No valid image files found"},
        )
        raise HTTPException(status_code=400, detail="No valid image files found")

    db.patch_job_debug(
        job_id,
        patch={"stages": {"upload": "complete"}, "upload": {"finished_at": db.utc_now_iso()}},
        event={"type": "upload_complete", "message": f"Upload complete with {saved_count} photos"},
    )

    pipeline.start_face_analysis(job_id, user_name=clean_user_name if clean_user_name else None)

    detail = pipeline.build_job_detail(job_id, _media_base_url(request))
    if not detail:
        raise HTTPException(status_code=500, detail="Failed to initialize job")

    return {"job": detail}


@router.post(
    "/api/jobs/{job_id}/select-cluster",
    response_model=SelectClusterResponse,
    tags=["pipeline"],
)
def select_cluster(
    job_id: str,
    body: SelectClusterRequest,
    request: Request,
) -> dict[str, object]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clusters = db.list_face_clusters(job_id)
    cluster_ids = {c["id"] for c in clusters}
    if body.cluster_id not in cluster_ids:
        raise HTTPException(status_code=400, detail="Cluster does not belong to this job")

    pipeline.start_clothing_extraction(job_id, body.cluster_id)
    db.patch_job_debug(
        job_id,
        patch={"stages": {"cluster_selection": "manual_selected"}},
        event={
            "type": "manual_cluster_selected",
            "message": "User manually selected face cluster",
            "data": {"cluster_id": body.cluster_id},
        },
    )

    detail = pipeline.build_job_detail(job_id, _media_base_url(request))
    if not detail:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job": detail}


@router.post("/api/items/{item_id}/refresh", tags=["pipeline"])
def refresh_item(item_id: str) -> dict[str, object]:
    item = db.get_clothing_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    pipeline.refresh_item_lookup(item_id)
    return {"ok": True, "item_id": item_id}


@router.post(
    "/api/styling/render-upload",
    response_model=ModelRenderResponse,
    tags=["styling"],
)
async def render_upload(
    request: Request,
    photo: UploadFile = File(...),
    identity_face_crop: UploadFile | None = File(default=None),
    reference_images: list[UploadFile] | None = File(default=None),
    style_preset: str = Form("aesthetic"),
    render_engine: str = Form("openai"),
    subject_hint: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    scene_hint: str | None = Form(None),
    variant_count: int = Form(3),
    aspect_ratio: str = Form("portrait"),
    quality: str = Form("high"),
    input_fidelity: str = Form("high"),
) -> dict[str, object]:
    settings = get_settings()
    run_id = uuid.uuid4().hex

    source_relative_paths: list[str] = []

    if identity_face_crop is not None:
        face_payload = await identity_face_crop.read()
        if face_payload:
            try:
                source_relative_paths.append(
                    model_render.save_uploaded_input_image(
                        settings=settings,
                        payload=face_payload,
                        run_id=run_id,
                        file_stem="identity_face",
                    )
                )
            except model_render.ModelRenderError:
                pass

    primary_payload = await photo.read()
    if not primary_payload:
        raise HTTPException(status_code=400, detail="Primary photo was empty")
    try:
        source_relative_paths.append(
            model_render.save_uploaded_input_image(
                settings=settings,
                payload=primary_payload,
                run_id=run_id,
                file_stem="primary",
            )
        )
    except model_render.ModelRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for idx, upload in enumerate(reference_images or []):
        payload = await upload.read()
        if not payload:
            continue
        try:
            source_relative_paths.append(
                model_render.save_uploaded_input_image(
                    settings=settings,
                    payload=payload,
                    run_id=run_id,
                    file_stem=f"reference_{idx + 1:02d}",
                )
            )
        except model_render.ModelRenderError:
            continue

    try:
        result = model_render.render_model_variants(
            settings=settings,
            source_relative_paths=source_relative_paths,
            style_preset=style_preset,
            render_engine=render_engine,
            subject_hint=subject_hint,
            custom_prompt=custom_prompt,
            scene_hint=scene_hint,
            variant_count=variant_count,
            aspect_ratio=aspect_ratio,
            quality=quality,
            input_fidelity=input_fidelity,
        )
    except model_render.ModelRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_base_url = _media_base_url(request)
    source_urls = [
        pipeline.media_rel_to_url(media_base_url, rel)
        for rel in source_relative_paths
    ]
    variants = [
        {
            "variant_index": idx + 1,
            "prompt": prompt,
            "output_url": pipeline.media_rel_to_url(media_base_url, out_rel),
        }
        for idx, (prompt, out_rel) in enumerate(
            zip(
                result["prompts_used"],
                result["output_relative_paths"],
                strict=False,
            )
        )
    ]
    return {
        "render_id": result["render_id"],
        "style_preset": result["style_preset"],
        "render_engine": result["render_engine"],
        "source_urls": source_urls,
        "variants": variants,
    }


@router.post(
    "/api/styling/evaluate",
    response_model=EvaluateVariantsResponse,
    tags=["styling"],
)
def evaluate_variants(body: EvaluateVariantsRequest) -> dict[str, object]:
    settings = get_settings()
    try:
        return model_render.evaluate_variants(
            settings=settings,
            render_id=body.render_id,
        )
    except model_render.VariantEvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/api/items/{item_id}/render-model",
    response_model=ModelRenderResponse,
    tags=["styling"],
)
def render_item_model(
    item_id: str,
    body: ItemModelRenderRequest,
    request: Request,
) -> dict[str, object]:
    settings = get_settings()
    item = db.get_clothing_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    crop_path = (item.get("crop_path") or "").strip()
    if not crop_path:
        raise HTTPException(status_code=400, detail="Item does not have a crop image yet")

    source_relative_paths = [crop_path]
    if body.include_face_tile:
        face_tile_rel = f"face_tiles/{item['job_id']}/{item['photo_id']}.jpg"
        face_tile_abs = settings.media_dir / face_tile_rel
        if face_tile_abs.exists():
            source_relative_paths.append(face_tile_rel)
    if body.include_original_photo:
        photo = db.get_photo(item["photo_id"])
        photo_path = (photo or {}).get("relative_path")
        if isinstance(photo_path, str) and photo_path.strip():
            source_relative_paths.append(photo_path)

    try:
        result = model_render.render_model_variants(
            settings=settings,
            source_relative_paths=source_relative_paths,
            style_preset=body.style_preset,
            render_engine=body.render_engine,
            subject_hint=None,
            custom_prompt=body.custom_prompt,
            scene_hint=body.scene_hint,
            variant_count=body.variant_count,
            aspect_ratio=body.aspect_ratio,
            quality=body.quality,
            input_fidelity=body.input_fidelity,
        )
    except model_render.ModelRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_base_url = _media_base_url(request)
    source_urls = [
        pipeline.media_rel_to_url(media_base_url, rel)
        for rel in source_relative_paths
    ]
    variants = [
        {
            "variant_index": idx + 1,
            "prompt": prompt,
            "output_url": pipeline.media_rel_to_url(media_base_url, out_rel),
        }
        for idx, (prompt, out_rel) in enumerate(
            zip(
                result["prompts_used"],
                result["output_relative_paths"],
                strict=False,
            )
        )
    ]
    return {
        "render_id": result["render_id"],
        "style_preset": result["style_preset"],
        "render_engine": result["render_engine"],
        "source_urls": source_urls,
        "variants": variants,
    }


@router.post(
    "/api/phia/backfill-favorites",
    response_model=BackfillFavoritesResponse,
    tags=["phia"],
)
def backfill_favorites(
    body: BackfillFavoritesRequest,
) -> dict[str, object]:
    product_urls = [u.strip() for u in body.product_urls if u and u.strip()]
    if not product_urls:
        raise HTTPException(status_code=400, detail="No product URLs provided")

    get_settings.cache_clear()
    settings = get_settings()
    try:
        return phia.backfill_favorites(
            product_urls=product_urls,
            settings=settings,
            collection_id=body.collection_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/phia/mobile-feed", tags=["phia"])
def get_mobile_feed(request: Request) -> dict[str, object]:
    get_settings.cache_clear()
    settings = get_settings()
    try:
        auth_override = _auth_override_from_headers(settings=settings, request=request)
        return phia.fetch_mobile_feed(
            settings=settings,
            auth_override=auth_override,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/phia/mobile-feed/simulate-session", tags=["phia"])
def simulate_mobile_feed_session(
    body: SimulateMobileFeedRequest,
) -> dict[str, object]:
    get_settings.cache_clear()
    settings = get_settings()
    try:
        auth_override = phia.resolve_phia_auth_for_session(
            settings=settings,
            phia_id=body.auth.phia_id,
            session_cookie=body.auth.session_cookie,
            cookie_header=body.auth.cookie_header,
            bearer_token=body.auth.bearer_token,
            authorization_header=body.auth.authorization_header,
            platform=body.auth.platform,
            platform_version=body.auth.platform_version,
            inherit_default_auth=body.inherit_default_auth,
            source="api_simulation",
        )
        feed = phia.fetch_mobile_feed(
            settings=settings,
            auth_override=auth_override,
            explore_feed_input=body.explore_feed_input,
        )
        feed["simulation"] = {
            "enabled": True,
            "source": "api_simulation",
            "auth_source": auth_override.source,
            "phia_id": auth_override.phia_id,
        }
        return feed
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _media_base_url(request: Request) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/media"


def _auth_override_from_headers(settings, request: Request):
    headers = request.headers
    phia_id = headers.get("x-phia-id")
    cookie_header = headers.get("cookie")
    authorization_header = headers.get("authorization")
    platform = headers.get("x-platform")
    platform_version = headers.get("x-platform-version")

    has_override = any(
        [
            (phia_id or "").strip(),
            (cookie_header or "").strip(),
            (authorization_header or "").strip(),
            (platform or "").strip(),
            (platform_version or "").strip(),
        ]
    )
    if not has_override:
        return None

    return phia.resolve_phia_auth_for_session(
        settings=settings,
        phia_id=phia_id,
        cookie_header=cookie_header,
        authorization_header=authorization_header,
        platform=platform,
        platform_version=platform_version,
        inherit_default_auth=True,
        source="request_headers",
    )
