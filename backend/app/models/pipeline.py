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
    captured_at: str | None = None
    captured_at_epoch_ms: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_source: str | None = None


class FaceClusterOut(BaseModel):
    id: str
    rep_photo_id: str
    rep_bbox: BoundingBox
    rep_aspect_ratio: float = 1
    member_count: int
    source_url: str


class FaceDetectionOut(BaseModel):
    id: str
    photo_id: str
    cluster_id: str | None = None
    bbox: BoundingBox
    confidence: float


class BestMatch(BaseModel):
    title: str = ""
    source: str = ""
    price: str | None = None
    link: str = ""
    thumbnail: str = ""
    confidence: float = 0
    reasoning: str = ""
    source_tier: Literal["exact", "similar"] = "similar"


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
    phia_products: list[dict] = Field(default_factory=list)
    best_match: BestMatch | None = None
    best_match_confidence: float = 0


class JobDetail(JobSummary):
    selected_cluster_id: str | None = None
    photos: list[PhotoOut] = Field(default_factory=list)
    face_detections: list[FaceDetectionOut] = Field(default_factory=list)
    clusters: list[FaceClusterOut] = Field(default_factory=list)
    items: list[ClothingItemOut] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class CreateJobResponse(BaseModel):
    job: JobDetail


class SelectClusterRequest(BaseModel):
    cluster_id: str


class SelectClusterResponse(BaseModel):
    job: JobDetail


class BackfillFavoritesRequest(BaseModel):
    product_urls: list[str] = Field(default_factory=list)
    collection_id: str = "all_favorites"


class BackfillFavoritesItemResult(BaseModel):
    product_url: str
    ok: bool
    message: str


class BackfillFavoritesResponse(BaseModel):
    phia_id: str
    collection_id: str
    source: str
    requested_count: int
    attempted_count: int
    added_count: int
    failed_count: int
    results: list[BackfillFavoritesItemResult] = Field(default_factory=list)


class PersonalizationPhotoInsight(BaseModel):
    photo_id: str
    summary: str
    style_tags: list[str] = Field(default_factory=list)
    brand_hints: list[str] = Field(default_factory=list)
    captured_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class PersonalizationSummaryResponse(BaseModel):
    job_id: str
    generated_at: str
    source: Literal["openai", "fallback", "cached"]
    photo_count: int
    collection_titles: list[str] = Field(default_factory=list)
    notifications: list[str] = Field(default_factory=list)
    favorite_brands: list[str] = Field(default_factory=list)
    style_summary: str = ""
    photo_insights: list[PersonalizationPhotoInsight] = Field(default_factory=list)


class PhiaSessionAuthInput(BaseModel):
    phia_id: str | None = None
    session_cookie: str | None = None
    cookie_header: str | None = None
    bearer_token: str | None = None
    authorization_header: str | None = None
    platform: str | None = None
    platform_version: str | None = None


class SimulateMobileFeedRequest(BaseModel):
    auth: PhiaSessionAuthInput
    inherit_default_auth: bool = True
    explore_feed_input: dict = Field(default_factory=dict)


StylePreset = Literal[
    "aesthetic",
    "editorial",
    "streetwear",
    "studio",
    "lookbook",
    "lifestyle",
    "runway",
]
RenderAspectRatio = Literal["portrait", "square", "landscape"]
RenderQuality = Literal["low", "medium", "high", "auto"]
RenderInputFidelity = Literal["low", "high"]
RenderEngine = Literal["openai", "nano_banana"]


class ItemModelRenderRequest(BaseModel):
    style_preset: StylePreset = "aesthetic"
    render_engine: RenderEngine = "openai"
    custom_prompt: str | None = None
    scene_hint: str | None = None
    variant_count: int = Field(default=3, ge=1, le=6)
    aspect_ratio: RenderAspectRatio = "portrait"
    quality: RenderQuality = "high"
    input_fidelity: RenderInputFidelity = "high"
    include_original_photo: bool = True
    include_face_tile: bool = True


class ModelRenderVariantOut(BaseModel):
    variant_index: int
    prompt: str
    output_url: str


class ModelRenderResponse(BaseModel):
    render_id: str
    style_preset: StylePreset
    render_engine: RenderEngine
    source_urls: list[str] = Field(default_factory=list)
    variants: list[ModelRenderVariantOut] = Field(default_factory=list)


class EvaluateVariantsRequest(BaseModel):
    render_id: str


class VariantEvaluationOut(BaseModel):
    variant_index: int
    realism: float
    aesthetic: float
    overall: float
    justification: str = ""


class EvaluateVariantsResponse(BaseModel):
    render_id: str
    variants: list[VariantEvaluationOut] = Field(default_factory=list)
    best_variant_index: int
    best_reason: str = ""


AutoStylingStatus = Literal["waiting", "queued", "running", "completed", "skipped", "failed"]


class AutoStylingVariantOut(BaseModel):
    variant_index: int
    prompt: str
    output_url: str
    realism: float | None = None
    aesthetic: float | None = None
    overall: float | None = None
    justification: str = ""
    is_best: bool = False


class AutoStylingRunOut(BaseModel):
    photo_id: str
    status: AutoStylingStatus
    source_photo_url: str
    face_crop_url: str | None = None
    selected_person_crop_url: str | None = None
    selected_person_bbox: dict | None = None
    gpt_selected_index: int | None = None
    gpt_selection_reason: str = ""
    body_visible: bool | None = None
    prompt: str = ""
    render_id: str | None = None
    best_variant_index: int | None = None
    best_reason: str = ""
    skip_reason: str = ""
    error: str = ""
    variants: list[AutoStylingVariantOut] = Field(default_factory=list)
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None


class AutoStylingRunsResponse(BaseModel):
    job_id: str
    runs: list[AutoStylingRunOut] = Field(default_factory=list)
