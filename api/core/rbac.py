"""Role-Based Access Control (RBAC) for MedScript AI."""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Any, Callable

from fastapi import Depends, HTTPException, status

from api.core.security import get_current_user_role


class Role(str, Enum):
    """User roles with hierarchical permissions."""
    VIEWER = "viewer"
    TRANSCRIBER = "transcriber"
    COLLECTOR = "collector"
    REVIEWER = "reviewer"
    ADMIN = "admin"


# ── Role Hierarchy ───────────────────────────────────────────────────────────
# Higher level roles inherit all permissions of lower levels

ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.TRANSCRIBER: 1,
    Role.COLLECTOR: 2,
    Role.REVIEWER: 3,
    Role.ADMIN: 4,
}

# ── Permission Matrix ────────────────────────────────────────────────────────

PERMISSIONS: dict[str, list[Role]] = {
    # Transcription
    "transcribe": [Role.TRANSCRIBER, Role.REVIEWER, Role.ADMIN],
    "view_transcriptions": [Role.VIEWER, Role.TRANSCRIBER, Role.REVIEWER, Role.ADMIN],

    # Review (human-in-the-loop)
    "review_transcriptions": [Role.REVIEWER, Role.ADMIN],
    "submit_feedback": [Role.REVIEWER, Role.ADMIN],

    # Data collection
    "upload_collection": [Role.COLLECTOR, Role.REVIEWER, Role.ADMIN],
    "annotate_collection": [Role.COLLECTOR, Role.REVIEWER, Role.ADMIN],

    # Admin
    "manage_users": [Role.ADMIN],
    "view_admin_panel": [Role.ADMIN],
    "export_data": [Role.ADMIN],
}


def has_permission(user_role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    try:
        role = Role(user_role)
    except ValueError:
        return False

    allowed_roles = PERMISSIONS.get(permission, [])
    return role in allowed_roles


def require_role(*required_roles: Role) -> Callable:
    """
    FastAPI dependency that checks if the current user has one of the required roles.

    Usage:
        @router.post("/transcribe", dependencies=[Depends(require_role(Role.TRANSCRIBER))])
        async def transcribe(...):
            ...
    """
    async def _check_role(role: str = Depends(get_current_user_role)) -> str:
        try:
            user_role = Role(role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid role: {role}",
            )

        # Check if user role is in required roles or is admin
        if user_role not in required_roles and user_role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in required_roles]}",
            )

        return role

    return _check_role


# ── Convenience Dependencies ─────────────────────────────────────────────────

require_viewer = require_role(Role.VIEWER, Role.TRANSCRIBER, Role.REVIEWER, Role.ADMIN)
require_transcriber = require_role(Role.TRANSCRIBER, Role.REVIEWER, Role.ADMIN)
require_collector = require_role(Role.COLLECTOR, Role.REVIEWER, Role.ADMIN)
require_reviewer = require_role(Role.REVIEWER, Role.ADMIN)
require_admin = require_role(Role.ADMIN)
