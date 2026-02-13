# Multi-stage build for Deep-Dive Video Note Taker
FROM python:3.11-slim AS base

# Install system dependencies including Node.js for yt-dlp JS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry: no virtualenv inside container
RUN poetry config virtualenvs.create false

# Install dependencies (without dev)
RUN poetry install --no-root --without dev --no-interaction

# Copy application source
COPY config.default.yaml ./
COPY src/ ./src/

# Install the application itself
RUN poetry install --only-root --no-interaction

# Create data directory
RUN mkdir -p /root/.notetaker

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: run the web server
CMD ["notetaker", "serve", "--host", "0.0.0.0", "--port", "8000"]
