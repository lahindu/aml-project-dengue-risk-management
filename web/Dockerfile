# syntax=docker/dockerfile:1
#
# Dengue risk web app (FastAPI + EBM model)
#
# Build (after python main.py has created srilanka_dengue_ebm_model.pkl):
#   docker build -t dengue-web:1.0 .
# Run:
#   docker run -d --name dengue-web -p 8000:8000 --restart unless-stopped dengue-web:1.0
# Open http://localhost:8000

ARG PYTHON_VERSION=3.12

# ---------- stage 1: install the Python packages into a virtual env ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements-prod.txt


# ---------- stage 2: small runtime image ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="dengue-web" \
      org.opencontainers.image.description="Dengue risk dashboard and next-week prediction for Sri Lankan districts (EBM)"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # data_prep.py imports matplotlib: no display in a container, and a writable cache folder
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000 \
    WEB_CONCURRENCY=2

# non-root user to run the app
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# only what the app needs at runtime (no training code, no notebooks)
COPY --chown=app:app app.py data_prep.py ./
COPY --chown=app:app static ./static
COPY --chown=app:app srilanka_dengue_weekly_by_district.csv srilanka_dengue_ebm_model.pkl ./

USER app

EXPOSE 8000

# the container is "healthy" when the API answers
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/districts', timeout=4)" || exit 1

# exec form through sh so $PORT and $WEB_CONCURRENCY can be changed with -e
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*' --no-server-header --timeout-graceful-shutdown 20"]
