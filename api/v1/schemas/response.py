"""Pydantic schemas for API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A single extracted entity from a prescription."""
    type: str = Field(..., description="Entity type: medicine, dosage, frequency, duration, instruction")
    value: str = Field(..., description="Extracted entity value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class TranscriptionResponse(BaseModel):
    """Response from the transcription endpoint."""
    transcription: str = Field(..., description="Full transcribed text")
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    word_confidences: list[float] = Field(default_factory=list, description="Per-word confidence scores")
    model_version: str = Field(default="medscript-ai-v0.1", description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    needs_review: bool = Field(default=False, description="True if low-confidence words detected")


class TranscriptionListItem(BaseModel):
    """Summary of a transcription for listing."""
    id: str
    transcription: str
    num_entities: int
    avg_confidence: float
    created_at: datetime
    status: str = "completed"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"
    model_loaded: bool = False
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
