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

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels/site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

ENV PYTHONPATH=/app/src:/app
ENV CONFTEST_DATABASE_URL=sqlite:////app/data/conftest.db
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/data /app/reports /app/models

EXPOSE 8000 8501

CMD ["uvicorn", "conftest.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
