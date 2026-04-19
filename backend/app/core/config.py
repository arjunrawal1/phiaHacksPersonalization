from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "phia-backend"
    environment: str = "development"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: list[str] = ["*"]

    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/phia.db")
    media_dir: Path = Path("./data/media")
    public_media_base_url: str = ""
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_photos_bucket: str = "photos"
    supabase_face_tiles_bucket: str = "face-tiles"
    supabase_clothing_crops_bucket: str = "clothing-crops"

    openai_api_key: str = ""
    openai_model: str = "gpt-5.4"
    openai_image_tool_model: str = "gpt-5.4"
    gemini_api_key: str = ""
    gemini_image_model: str = "gemini-2.5-flash-image"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    perplexity_api_key: str = ""
    perplexity_model: str = "sonar"

    serpapi_key: str = ""
    replicate_api_token: str = ""
    google_places_api_key: str = ""
    google_places_nearby_radius_m: int = 75

    recent_limit: int = 50
    photo_concurrency: int = 10
    lookup_concurrency: int = 5
    online_face_image_limit: int = 8
    auto_face_pick_threshold: float = 0.75
    auto_face_pick_margin: float = 0.12
    auto_face_pick_alignment_floor: float = 0.5

    phia_graphql_url: str = "https://api.phia.com/v2/graphql"
    phia_collection_id: str = "all_favorites"
    phia_id: str = ""
    phia_session_cookie: str = ""
    phia_bearer_token: str = ""
    phia_platform: str = "IOS_APP"
    phia_platform_version: str = "2.3.11.362"
    phia_capture_dirs: list[str] = []
    phia_allow_insecure_tls: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    return settings
