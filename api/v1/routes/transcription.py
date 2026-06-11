"""Transcription routes — upload prescription images for AI transcription."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from api.core.security import get_current_user_id
from api.core.rbac import require_transcriber
from api.v1.schemas.response import Entity, TranscriptionResponse

router = APIRouter()

# In-memory store for transcription results (replace with DB)
_transcriptions: dict[str, dict[str, Any]] = {}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(require_transcriber)],
)
async def transcribe_prescription(
    request: Request,
    file: UploadFile = File(..., description="Prescription image (JPG, PNG, or PDF)"),
    user_id: str = Depends(get_current_user_id),
) -> TranscriptionResponse:
    """
    Transcribe a prescription image into structured text.

    Accepts JPG, PNG, or PDF images up to 10 MB.
    Returns transcribed text with extracted entities and confidence scores.
    """
    # Validate file type
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
            )

    # Read file
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    # Run inference
    predictor = request.app.state.predictor

    if predictor is not None:
        import io
        import numpy as np
        from PIL import Image

        # Load image from bytes
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = predictor.predict(image, run_ner=True)

        entities = [
            Entity(
                type=e.get("type", "unknown"),
                value=e.get("value", ""),
                confidence=round(e.get("confidence", 0.0), 4),
            )
            for e in result.entities
        ]

        response = TranscriptionResponse(
            transcription=result.transcription,
            entities=entities,
            word_confidences=[round(c, 4) for c in result.word_confidences],
            model_version=result.model_version,
            needs_review=any(c < 0.5 for c in result.word_confidences),
        )
    else:
        # No model loaded — return mock response for development
        response = TranscriptionResponse(
            transcription="Amoxicillin 500mg TID for 7 days | Paracetamol 650mg SOS",
            entities=[
                Entity(type="medicine", value="Amoxicillin", confidence=0.94),
                Entity(type="dosage", value="500mg", confidence=0.91),
                Entity(type="frequency", value="TID", confidence=0.88),
                Entity(type="duration", value="7 days", confidence=0.85),
                Entity(type="medicine", value="Paracetamol", confidence=0.92),
                Entity(type="dosage", value="650mg", confidence=0.89),
                Entity(type="frequency", value="SOS", confidence=0.87),
            ],
            word_confidences=[0.94, 0.91, 0.88, 0.85, 0.92, 0.89, 0.87],
            model_version="medscript-ai-v0.1-mock",
            needs_review=False,
        )

    # Store result
    result_id = str(uuid.uuid4())
    _transcriptions[result_id] = {
        "id": result_id,
        "user_id": user_id,
        "response": response.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
    }

    return response


@router.get("/transcriptions", dependencies=[Depends(require_transcriber)])
async def list_transcriptions(
    user_id: str = Depends(get_current_user_id),
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List user's transcription history."""
    user_results = [
        t for t in _transcriptions.values() if t["user_id"] == user_id
    ]
    user_results.sort(key=lambda x: x["created_at"], reverse=True)

    return {
        "results": user_results[offset:offset + limit],
        "total": len(user_results),
        "limit": limit,
        "offset": offset,
    }
