# Multi-stage build for Deep-Dive Video Note Taker
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
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

# Default: run the web server
CMD ["notetaker", "serve", "--host", "0.0.0.0", "--port", "8000"]
