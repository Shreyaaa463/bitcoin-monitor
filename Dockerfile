# Optional container image. Docker is NOT required -- ./run.sh on a plain Linux host is
# the primary deployment. The image needs no Internet access at RUNTIME.
#
# Build (on a machine that can reach PyPI, OR offline if wheelhouse/ was populated by
# scripts/build_offline_bundle.sh with PY_VERSION=3.12):
#   docker build -t trace-x:2.0.0 .
# Move to the offline host:
#   docker save trace-x:2.0.0 | gzip > trace-x-2.0.0.tar.gz      # on the build host
#   docker load < trace-x-2.0.0.tar.gz                           # on the offline host
# Run (dashboard reachable only from the host itself):
#   docker run --rm -p 127.0.0.1:8501:8501 \
#       -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" trace-x:2.0.0
#   -> http://localhost:8501     (extra launcher args go after the image name,
#                                 e.g. ... trace-x:2.0.0 --input data/raw/other.json)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (cached layer). Uses the local wheelhouse if present, else PyPI.
COPY requirements.txt requirements.lock.txt ./
COPY wheelhouse/ /wheelhouse/
RUN if ls /wheelhouse/*.whl >/dev/null 2>&1; then \
        pip install --no-index --find-links /wheelhouse -r requirements.lock.txt; \
    else \
        pip install -r requirements.lock.txt; \
    fi && rm -rf /wheelhouse

COPY . .
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/runs /app/data/raw /app/data/processed /app/data/geoip \
    && chown -R app:app /app
USER app

VOLUME ["/app/data", "/app/runs"]
EXPOSE 8501
# Inside the container the server must listen on all interfaces; publish it to the
# host's loopback only (-p 127.0.0.1:8501:8501) to keep it local.
ENTRYPOINT ["python", "-m", "src.launcher", "--host", "0.0.0.0", "--port", "8501"]
