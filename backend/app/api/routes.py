from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.core.config import get_settings
from app.models.pipeline import (
    CreateJobResponse,
    JobDetail,
    JobSummary,
    SelectClusterRequest,
    SelectClusterResponse,
)
from app.services import db, pipeline

router = APIRouter()


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


@router.post("/api/jobs", response_model=CreateJobResponse, tags=["pipeline"])
async def create_job(
    request: Request,
    photos: list[UploadFile] = File(...),
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

    for upload in photos:
        payload = await upload.read()
        if not payload:
            continue

        try:
            with Image.open(BytesIO(payload)).convert("RGB") as image:
                width, height = image.size
                rel_path = f"photos/{job_id}/{saved_count:04d}.jpg"
                abs_path = settings.media_dir / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(abs_path, format="JPEG", quality=90)
                db.create_photo(
                    job_id=job_id,
                    relative_path=rel_path,
                    width=width,
                    height=height,
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


def _media_base_url(request: Request) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/media"
