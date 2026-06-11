"""Authentication routes — register, login, OAuth, token refresh."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from api.core.rbac import require_admin
from api.v1.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenResponse,
    TokenRefreshRequest,
    UserProfile,
    RoleUpdate,
)

router = APIRouter()

# ── In-memory user store (replace with PostgreSQL in production) ─────────────
# This is a simplified store for MVP. In production, use SQLAlchemy + PostgreSQL.

_users_db: dict[str, dict] = {}
_users_by_username: dict[str, str] = {}  # username → user_id


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister) -> TokenResponse:
    """Register a new user account."""
    # Check if username exists
    if user_data.username in _users_by_username:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    user_id = str(uuid.uuid4())

    # Create user
    user = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "organization": user_data.organization,
        "password_hash": hash_password(user_data.password),
        "role": "reviewer",  # Default role (gives access to all features in dev)
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
    }

    _users_db[user_id] = user
    _users_by_username[user_data.username] = user_id

    # Generate tokens
    token_data = {"sub": user_id, "role": user["role"], "username": user["username"]}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin) -> TokenResponse:
    """Login with username and password."""
    user_id = _users_by_username.get(login_data.username)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    user = _users_db[user_id]

    if not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Generate tokens
    token_data = {"sub": user_id, "role": user["role"], "username": user["username"]}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/token/refresh", response_model=TokenResponse)
async def refresh_token(request: TokenRefreshRequest) -> TokenResponse:
    """Refresh an access token using a refresh token."""
    payload = decode_token(request.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    if not user_id or user_id not in _users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    user = _users_db[user_id]
    token_data = {"sub": user_id, "role": user["role"], "username": user["username"]}

    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
    )


@router.get("/me", response_model=UserProfile)
async def get_current_user(user_id: str = Depends(get_current_user_id)) -> UserProfile:
    """Get current user profile."""
    user = _users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserProfile(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        organization=user.get("organization"),
        is_active=user["is_active"],
        created_at=user.get("created_at"),
    )


@router.put("/users/role", dependencies=[Depends(require_admin)])
async def update_user_role(role_update: RoleUpdate) -> dict:
    """Update a user's role (admin only)."""
    user = _users_db.get(role_update.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user["role"] = role_update.new_role
    return {"message": f"Role updated to {role_update.new_role}", "user_id": role_update.user_id}
