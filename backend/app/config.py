from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://karima:karima@localhost:5432/karima_cv"

    # Auth
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # API
    api_prefix: str = "/api"
    cors_origins: list[str] = ["*"]

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    uploads_dir: Path = base_dir / "uploads"
    cache_dir: Path = base_dir / "cache"
    config_dir: Path = base_dir / "config"
    models_dir: Path = base_dir / "models"

    # YOLO models (paths relative to old backends — updated during migration)
    ppe_model: str = ""
    parking_model: str = ""
    intrusion_model: str = ""
    interaction_model: str = ""
    speed_model: str = ""

    # Streaming
    jpeg_quality: int = 72
    max_frame_width: int = 1280


def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Ensure required directories exist
for _dir in (settings.uploads_dir, settings.cache_dir, settings.config_dir):
    _dir.mkdir(parents=True, exist_ok=True)
