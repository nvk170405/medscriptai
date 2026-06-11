"""Structured logging — JSON-formatted logs with context for audit trails."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for MedScript AI.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output JSON-formatted logs (for production).
        log_file: Optional file path for log output.
    """
    # Determine processors based on format
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy loggers
    for noisy_logger in ["urllib3", "botocore", "boto3", "s3transfer"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)


# ── Audit Logger ─────────────────────────────────────────────────────────────
# For HIPAA compliance — logs all data access events


class AuditLogger:
    """Audit logger for HIPAA/GDPR compliance."""

    def __init__(self) -> None:
        self._logger = get_logger("audit")

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a data access event for compliance."""
        self._logger.info(
            "data_access",
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
        )

    def log_transcription(
        self,
        user_id: str,
        image_id: str,
        result_id: str,
    ) -> None:
        """Log a transcription request."""
        self._logger.info(
            "transcription",
            user_id=user_id,
            image_id=image_id,
            result_id=result_id,
        )

    def log_correction(
        self,
        user_id: str,
        result_id: str,
        original: str,
        corrected: str,
    ) -> None:
        """Log a human correction for audit trail."""
        self._logger.info(
            "correction",
            user_id=user_id,
            result_id=result_id,
            original_length=len(original),
            corrected_length=len(corrected),
        )

    def log_auth_event(
        self,
        event: str,
        user_id: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
    ) -> None:
        """Log an authentication event."""
        self._logger.info(
            "auth_event",
            event=event,
            user_id=user_id,
            ip_address=ip_address,
            success=success,
        )


# Singleton audit logger
audit_logger = AuditLogger()
