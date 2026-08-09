# HaluRISC backend — CPU-compatible FastAPI inference server (blueprint B11).
# CUDA is an OPTIONAL local acceleration path; this image is intentionally CPU
# so a clean clone can run the full /predict + /explain stack anywhere.
#
# Build & run (from repo root):
#   docker build -t halurisc-api .
#   docker run --rm -p 8000:8000 halurisc-api
#   curl http://127.0.0.1:8000/health

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HALU_API_DEVICE=cpu \
    HALU_API_PRELOAD=1 \
    HALU_XGB_DEVICE=cpu \
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8000

# libgomp1: OpenMP runtime required by xgboost's native code on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

RUN useradd --create-home --shell /usr/sbin/nologin halurisc \
    && chown -R halurisc:halurisc /app
USER halurisc

EXPOSE 8000

# One worker, no --reload (roadmap B7.11): the API is a single-serve inference
# process; heavy models live in the image, not a reload watcher.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
