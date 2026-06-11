"""Data collection routes — upload raw prescription images for dataset building."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from api.core.security import get_current_user_id
from api.core.rbac import require_collector

router = APIRouter()

# In-memory collection store
_collections: list[dict[str, Any]] = []


@router.post("/collect/upload", dependencies=[Depends(require_collector)])
async def upload_collection_image(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Upload a raw prescription image for dataset collection."""
    content = await file.read()

    collection_id = str(uuid.uuid4())
    _collections.append({
        "id": collection_id,
        "user_id": user_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded",
        "annotated": False,
        "created_at": datetime.utcnow().isoformat(),
    })

    return {
        "collection_id": collection_id,
        "message": "Image uploaded successfully",
    }


@router.get("/collect/stats", dependencies=[Depends(require_collector)])
async def get_collection_stats() -> dict:
    """Get data collection progress statistics."""
    total = len(_collections)
    annotated = sum(1 for c in _collections if c.get("annotated"))

    return {
        "total_images": total,
        "annotated": annotated,
        "pending_annotation": total - annotated,
        "contributors": len(set(c["user_id"] for c in _collections)),
    }
