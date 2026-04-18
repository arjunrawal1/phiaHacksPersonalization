from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal[
    "uploading",
    "analyzing_faces",
    "awaiting_face_pick",
    "extracting_clothing",
    "done",
    "failed",
]

ItemTier = Literal["exact", "similar", "generic", "pending"]
Visibility = Literal["clear", "partial", "obscured"]


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class ItemBoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class JobSummary(BaseModel):
    id: str
    status: str
    photo_count: int
    error: str | None = None
    created_at: str
    updated_at: str


class PhotoOut(BaseModel):
    id: str
    url: str
    width: int | None = None
    height: int | None = None


class FaceClusterOut(BaseModel):
    id: str
    rep_photo_id: str
    rep_bbox: BoundingBox
    rep_aspect_ratio: float = 1
    member_count: int
    source_url: str


class BestMatch(BaseModel):
    title: str
    source: str
    price: str | None = None
    link: str
    thumbnail: str
    confidence: float
    reasoning: str
    source_tier: Literal["exact", "similar"]


class ClothingItemOut(BaseModel):
    id: str
    photo_id: str
    category: str
    description: str
    colors: list[str] = Field(default_factory=list)
    pattern: str = ""
    style: str = ""
    brand_visible: str | None = None
    visibility: str = "clear"
    confidence: float = 0
    bounding_box: ItemBoundingBox
    crop_url: str | None = None
    tier: str = "generic"
    exact_matches: list[dict] = Field(default_factory=list)
    similar_products: list[dict] = Field(default_factory=list)
    best_match: BestMatch | None = None
    best_match_confidence: float = 0


class JobDetail(JobSummary):
    selected_cluster_id: str | None = None
    photos: list[PhotoOut] = Field(default_factory=list)
    clusters: list[FaceClusterOut] = Field(default_factory=list)
    items: list[ClothingItemOut] = Field(default_factory=list)


class CreateJobResponse(BaseModel):
    job: JobDetail


class SelectClusterRequest(BaseModel):
    cluster_id: str


class SelectClusterResponse(BaseModel):
    job: JobDetail
