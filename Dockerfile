FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    MALLOC_TRIM_THRESHOLD_=0 \
    MALLOC_ARENA_MAX=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

# ---------- dependency layer (cached unless lock changes) ----------
COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev --no-root

# ---------- source layer ----------
COPY . .
RUN poetry install --with dev

# Defaults: results & logs live here at runtime
RUN mkdir -p results logs out keys

# HF cache lives here; mount a host volume to avoid re-downloading models
ENV HF_HOME=/app/.hf_cache
RUN mkdir -p /app/.hf_cache

CMD ["bash"]
