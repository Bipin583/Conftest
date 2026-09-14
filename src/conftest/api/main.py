"""
ConfTest FastAPI Main Application Entry Point.

Provides RESTful endpoints for repository registration, AST analysis,
test-selection prediction, calibrated confidence reporting, model explainability,
analytics, and CI/CD webhook processing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from conftest.config import settings
from conftest.logging_config import get_logger
from conftest.db.init_db import init_db
from conftest.api.routes.health import router as health_router
from conftest.api.routes.selection import router as selection_router
from conftest.api.routes.explain import router as explain_router
from conftest.api.routes.calibration import router as calibration_router
from conftest.api.routes.repositories import router as repositories_router
from conftest.api.routes.analytics import router as analytics_router
from conftest.api.routes.github_webhook import router as github_router
from conftest.api.routes.flakiness import router as flakiness_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: runs startup and shutdown tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.version} [{settings.env}]...")
    # Initialize database schema automatically on startup
    init_db()
    yield
    logger.info(f"Shutting down {settings.app_name}...")


# Create main FastAPI application
app = FastAPI(
    title="ConfTest API",
    description=(
        "Confidence-Calibrated Regression Test Selection API with Selective Prediction "
        "and Safe Full-Suite Fallback for CI/CD Optimization."
    ),
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS).
#
# `allow_origins=["*"]` together with `allow_credentials=True` is the one
# combination the CORS spec forbids, and browsers reject it outright -- so the
# previous configuration was both unsafe in intent and broken in effect for the
# credentialed requests it was trying to enable. Origins are now an explicit
# per-deployment setting, and a wildcard is refused in production by
# Settings._reject_insecure_production_defaults.
_cors_origins = list(settings.cors_allow_origins)
_wildcard = "*" in _cors_origins
if _wildcard:
    logger.warning(
        "CORS is configured with a wildcard origin; credentials are disabled "
        "because browsers refuse that combination. Set "
        "CONFTEST_CORS_ALLOW_ORIGINS to the dashboard's real origins."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-GitHub-Event", "X-Hub-Signature-256"],
)

# Register routes.
# Health is mounted twice on purpose: unprefixed for container/orchestrator
# probes that expect /health, and under /api/v1 alongside everything else.
app.include_router(health_router, prefix="", tags=["System & Diagnostics"])
app.include_router(health_router, prefix="/api/v1", tags=["System & Diagnostics"])
app.include_router(selection_router, prefix="/api/v1", tags=["Test Selection"])
app.include_router(explain_router, prefix="/api/v1", tags=["Model Explainability"])
app.include_router(calibration_router, prefix="/api/v1", tags=["Confidence Calibration"])
app.include_router(repositories_router, prefix="/api/v1", tags=["Repositories"])
app.include_router(analytics_router, prefix="/api/v1", tags=["Analytics & Telemetry"])
app.include_router(github_router, prefix="/api/v1", tags=["GitHub Integration"])
app.include_router(flakiness_router, prefix="/api/v1", tags=["Flaky Test Detection"])


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    tags=["Root"],
    summary="ConfTest API Root",
    description="Returns welcome information and link to API documentation.",
)
def root():
    """Root endpoint welcoming clients and directing to interactive documentation."""
    return JSONResponse(
        content={
            "app": settings.app_name,
            "version": settings.version,
            "description": "Confidence-Calibrated Regression Test Selection API",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "select": "/api/v1/select",
                "explain": "/api/v1/explain",
                "calibration": "/api/v1/calibration",
                "repositories": "/api/v1/repositories",
                "analytics": "/api/v1/analytics",
                "flakiness": "/api/v1/flakiness/predict",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "conftest.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers,
    )
