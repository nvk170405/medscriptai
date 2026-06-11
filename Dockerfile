# =============================================================================
# MedScript AI — Production Dockerfile
# Multi-stage build for FastAPI + ML inference
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -e .

# ── Stage 2: Production ─────────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r medscript && useradd -r -g medscript medscript
RUN mkdir -p /app/data /app/checkpoints /app/logs && \
    chown -R medscript:medscript /app

# Copy application code
COPY --chown=medscript:medscript src/ /app/src/
COPY --chown=medscript:medscript api/ /app/api/
COPY --chown=medscript:medscript configs/ /app/configs/

# Switch to non-root user
USER medscript

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
