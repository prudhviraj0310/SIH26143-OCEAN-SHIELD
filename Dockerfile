# OCEAN-SHIELD: Defense-Grade Maritime Oil Spill Attribution & Tracking System
# Problem Statement SIH26143 (NTRO / Indian Coast Guard)

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install essential system libraries for scientific computing, NetCDF, and headless image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    libnetcdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (CPU-optimized PyTorch and scientific stack)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code and resources
COPY src/ /app/src/
COPY models/ /app/models/
COPY datasets/ /app/datasets/
COPY reports/ /app/reports/
COPY server.py /app/server.py

# Create unprivileged user for defense-grade security hardening
RUN useradd -m -u 1001 oceanshield && \
    chown -R oceanshield:oceanshield /app
USER oceanshield

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.ocean_shield.server:app", "--host", "0.0.0.0", "--port", "8000"]
