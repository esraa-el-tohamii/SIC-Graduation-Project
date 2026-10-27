# ─────────────────────────────────────────────────────────────────────────────
# Electricity Theft Detection — Inference Service
#
# Build:
#   docker build -t theft-api:latest .
#
# Run (after copying artifacts into ./artifacts/):
#   docker run --rm -p 8000:8000 theft-api:latest
#
# Test the running container:
#   curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

# ── Base image ─────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the layer small; no CUDA needed for inference.
FROM python:3.11-slim

# ── System packages ────────────────────────────────────────────────────────
# libgomp1 is required by XGBoost for OpenMP parallelism.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ──────────────────────────────────────────────────────────
RUN useradd --create-home appuser
WORKDIR /home/appuser/api
USER appuser

# ── Python path ───────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# ── Install dependencies ───────────────────────────────────────────────────
# Install PyTorch CPU-only wheel first (avoids pulling CUDA libs ~4 GB).
COPY --chown=appuser:appuser requirements.txt .

RUN pip install --upgrade pip --quiet \
    && pip install --quiet \
        torch==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && grep -v "^torch" requirements.txt > /tmp/req_no_torch.txt \
    && pip install --quiet -r /tmp/req_no_torch.txt

# ── Copy application source ────────────────────────────────────────────────
COPY --chown=appuser:appuser app/ ./app/

# ── Copy model artifacts ───────────────────────────────────────────────────
# The categorized artifacts/ directory is copied into the image at build time.
COPY --chown=appuser:appuser artifacts/ ./artifacts/

# ── Expose port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start the server ───────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
