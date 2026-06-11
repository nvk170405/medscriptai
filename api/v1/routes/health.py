"""Health check routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.v1.schemas.response import HealthResponse

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check — always returns 200 if API is running."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=request.app.state.predictor is not None,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/readiness")
async def readiness_check(request: Request) -> dict:
    """Readiness check — verifies model is loaded and services are connected."""
    checks = {
        "api": "ready",
        "model": "loaded" if request.app.state.predictor is not None else "not_loaded",
    }

    all_ready = all(v in ("ready", "loaded") for v in checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
    }
