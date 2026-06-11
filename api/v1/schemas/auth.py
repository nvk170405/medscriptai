"""Pydantic schemas for authentication."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=100)
    organization: str | None = None


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserProfile(BaseModel):
    """User profile response."""
    id: str
    username: str
    email: str
    full_name: str
    role: str
    organization: str | None = None
    is_active: bool = True
    created_at: str | None = None


class RoleUpdate(BaseModel):
    """Role update request (admin only)."""
    user_id: str
    new_role: str = Field(..., pattern="^(viewer|transcriber|collector|reviewer|admin)$")


class OAuthCallbackResponse(BaseModel):
    """OAuth callback response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserProfile
