"""Feedback routes — human-in-the-loop corrections."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.core.security import get_current_user_id
from api.core.rbac import require_reviewer

router = APIRouter()

# In-memory feedback store
_feedback: list[dict[str, Any]] = []


class FeedbackRequest(BaseModel):
    """Human correction submission."""
    transcription_id: str
    original_text: str
    corrected_text: str
    corrected_entities: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    feedback_id: str
    message: str = "Correction submitted successfully"


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_reviewer)])
async def submit_feedback(
    feedback: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
) -> FeedbackResponse:
    """Submit a human correction for a transcription."""
    feedback_id = str(uuid.uuid4())

    _feedback.append({
        "id": feedback_id,
        "transcription_id": feedback.transcription_id,
        "user_id": user_id,
        "original_text": feedback.original_text,
        "corrected_text": feedback.corrected_text,
        "corrected_entities": feedback.corrected_entities,
        "notes": feedback.notes,
        "created_at": datetime.utcnow().isoformat(),
    })

    return FeedbackResponse(feedback_id=feedback_id)


@router.get("/feedback/pending", dependencies=[Depends(require_reviewer)])
async def get_pending_feedback() -> dict:
    """Get pending corrections for review."""
    return {
        "pending": _feedback[-50:],  # Last 50
        "total": len(_feedback),
    }
