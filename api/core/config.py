"""API configuration using Pydantic BaseSettings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "MedScript AI"
    app_env: str = "development"
    debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_version: str = "0.1.0"

    # Security / JWT
    secret_key: str = "change-me-to-a-random-64-char-string"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # OAuth 2.0 (Google)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/oauth/google/callback"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "medscript"
    postgres_user: str = "medscript"
    postgres_password: str = "medscript_dev_password"

    # MongoDB
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    mongo_db: str = "medscript"
    mongo_user: str = "medscript"
    mongo_password: str = "medscript_dev_password"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Storage
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "medscript-data"

    # Model
    model_checkpoint_path: str = "checkpoints/best.ckpt"
    model_device: str = "cpu"

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def mongo_url(self) -> str:
        return (
            f"mongodb://{self.mongo_user}:{self.mongo_password}"
            f"@{self.mongo_host}:{self.mongo_port}/{self.mongo_db}?authSource=admin"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
