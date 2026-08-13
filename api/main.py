"""FastAPI application factory — MedScript AI backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.core.config import settings
from api.core.middleware import AuditLoggingMiddleware, RateLimitMiddleware

# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — load model on startup, cleanup on shutdown."""
    from medscript.utils.logging import setup_logging, get_logger

    setup_logging(
        level="DEBUG" if settings.debug else "INFO",
        json_format=not settings.debug,
    )
    logger = get_logger("api")
    logger.info("starting_medscript_api", version=settings.app_version)

    # Load ML models (EasyOCR + BiomedBERT NER)
    from medscript.inference.predictor import MedScriptPredictor

    app.state.predictor = MedScriptPredictor(
        device=settings.model_device,
    )
    logger.info("predictor_ready", engine="easyocr+biomedbert")

    yield

    # Cleanup
    logger.info("shutting_down_medscript_api")


# ── App Factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MedScript AI",
        description=(
            "Privacy-first AI system for transcribing doctor handwriting "
            "into structured digital text. Extracts medicine names, dosages, "
            "frequencies, and durations from prescription images."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AuditLoggingMiddleware)

    # ── Routes ───────────────────────────────────────────────────────────
    from api.v1.routes import auth, transcription, health, feedback, collection

    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(transcription.router, prefix="/api/v1", tags=["Transcription"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
    app.include_router(collection.router, prefix="/api/v1", tags=["Data Collection"])

    return app


# ── Application Instance ─────────────────────────────────────────────────────

app = create_app()


def run() -> None:
    """Run the API server (for CLI entry point)."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
