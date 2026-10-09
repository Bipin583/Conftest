# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Final Runtime Image
FROM python:3.11-slim AS runtime

# Git SHA of the commit the image was built from. Render passes this
# automatically on every deploy; locally, docker-compose.yml sets it from
# `git rev-parse --short HEAD` or it stays "unknown" (dev containers have no
# meaningful commit identity). Surfaced in /health so "which code is the
# deployed bot running" is answerable with one curl instead of a dashboard.
ARG GIT_SHA=unknown
ENV CONFTEST_GIT_SHA=${GIT_SHA}

WORKDIR /app

# nginx is the front proxy exposing the platform port; gettext-base provides
# envsubst, which the entrypoint uses to bind nginx to Render's PORT.
# The default site is removed so only deploy/nginx.conf.template is served.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    curl \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

# Run the whole stack as an unprivileged user instead of root, so a compromise
# of uvicorn, Streamlit or nginx cannot write outside the app's own data dirs
# or escalate inside the container. --create-home gives Streamlit and
# matplotlib a writable HOME for their caches.
RUN useradd --create-home --uid 10001 appuser
ENV HOME=/home/appuser

# Copy installed wheels/site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
# Runtime artifacts the API and dashboard serve from -- docker-compose mounts
# these from the host for local development, but a platform build (Render)
# has no host to mount from, so they must ship inside the image. models/ is
# ~0.2 MB, reports/ ~1.4 MB, sample_suite/ a few KB of demo tests.
COPY models/ ./models/
COPY reports/ ./reports/
COPY tests/sample_suite/ ./tests/sample_suite/
COPY deploy/ ./deploy/
COPY pyproject.toml ./

# Rootless nginx: replace the stock root-assuming main config (see
# deploy/nginx-main.conf for why) and let appuser render the per-deploy server
# block into conf.d at container start.
COPY deploy/nginx-main.conf /etc/nginx/nginx.conf

RUN chmod +x /app/deploy/entrypoint.sh

ENV PYTHONPATH=/app/src:/app
ENV CONFTEST_DATABASE_URL=sqlite:////app/data/conftest.db
ENV PYTHONUNBUFFERED=1

# Create the dirs the unprivileged runtime user must write to and hand them
# over: the SQLite data dir and the model/report dirs config.py recreates on
# import, plus the nginx conf.d the entrypoint renders default.conf into.
# nginx's own pid/log/temp paths are redirected to /tmp by nginx-main.conf, so
# they need no chown.
RUN mkdir -p /app/data /app/reports /app/models \
    && chown -R appuser:appuser /app/data /app/reports /app/models /etc/nginx/conf.d

EXPOSE 10000

# Drop privileges: the supervised uvicorn (:8000), Streamlit (:8501) and nginx
# (:10000) all bind unprivileged ports, so none of them needs root.
USER appuser

# Single-service stack: entrypoint supervises uvicorn (127.0.0.1:8000) and
# Streamlit (127.0.0.1:8501) behind nginx on the platform port. Local
# docker-compose overrides this command to run each backend separately.
CMD ["/app/deploy/entrypoint.sh"]
