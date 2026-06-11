"""API middleware — audit logging, rate limiting."""

from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from medscript.utils.logging import get_logger

logger = get_logger("api.middleware")


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests for HIPAA audit compliance."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log (skip health checks to reduce noise)
        if not path.endswith("/health"):
            logger.info(
                "api_request",
                method=method,
                path=path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
            )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting (use Redis in production)."""

    def __init__(self, app, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip]
                if current_time - t < 60
            ]
        else:
            self._requests[client_ip] = []

        # Check rate limit
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        self._requests[client_ip].append(current_time)
        return await call_next(request)
