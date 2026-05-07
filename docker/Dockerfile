# syntax=docker/dockerfile:1.6
#
# Submission image per §3.2 / §3.4. Two-stage build: builder runs `uv export`
# and installs Python deps into the system site-packages, final image only
# carries runtime artefacts (no compilers / no uv).
#
# Opt-in CN mirror:
#   docker build \
#       --build-arg UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
#       -t <team_id>:v<N> -f docker/Dockerfile .
# main's old `docker/pip.conf` was dead — uv does not read pip.conf, only
# UV_INDEX_URL / [tool.uv] in pyproject.toml. Build-arg is the correct knob.

FROM --platform=linux/amd64 python:3.11-slim AS builder

ARG UV_INDEX_URL=""
ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# All listed deps (numpy/pandas/pyarrow/duckdb/polars/pydantic/...) ship
# manylinux wheels for x86_64, so no compilers needed. If a future dep
# requires C extensions, re-add: build-essential gcc g++.
RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# --no-emit-project: skip the `-e .` self-install line. The final stage uses
#   PYTHONPATH=/app instead of an installed package, so a self-install would
#   force us to also COPY README.md + src/ into the deps layer — wasteful and
#   it'd bust the deps cache on every code change.
# --no-hashes: trades integrity for build speed; acceptable since the .tar.gz
#   we ship carries an sha256 in the email body anyway (see scripts/submit.sh).
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt \
 && uv pip install --system --no-cache -r requirements.txt

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Pure-version tools handle csv / json / md / sqlite / xlsx only — no OCR,
# no PDF rasterisation, no image processing — so the image-side stack
# (tesseract, poppler-utils, libgl1, libglib2.0-0) is unneeded. python:3.11-slim
# already ships everything Python deps load at runtime.

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
# Configs first (rarely change), then code (changes most). .dockerignore
# excludes configs/*.local.yaml so dev secrets never leak into the image.
COPY configs ./configs
COPY src/data_agent_baseline ./data_agent_baseline

# Mount points (the judge mounts these via `docker run`, see §3.4):
#   /input  ro - all task_<id>/{task.json,context/}
#   /output rw - prediction.csv targets, one dir per task
#   /logs   rw - §3.7: minimal status only; never raw input rows / gold

ENTRYPOINT ["python", "-m", "data_agent_baseline.submission"]
