"""
Health Check and System Diagnostics Endpoint.
"""

import time
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from conftest.config import settings
from conftest.db.session import get_db
from conftest.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["System & Diagnostics"])

# Record application boot timestamp
START_TIME = time.time()


class HealthResponse(BaseModel):
    """Structured response model for system health checks."""
    status: str = Field(..., examples=["healthy"], description="Overall health status")
    service: str = Field(..., examples=["ConfTest API"], description="Service name")
    version: str = Field(..., examples=["0.1.0"], description="Semantic application version")
    environment: str = Field(..., examples=["development"], description="Runtime environment")
    git_sha: str = Field(
        ...,
        examples=["8f0d6b1", "unknown"],
        description=(
            "Git commit this process was built from, read from CONFTEST_GIT_SHA "
            "(set by the Docker build; 'unknown' when unset, e.g. bare dev runs)."
        ),
    )
    database: str = Field(..., examples=["connected"], description="Database connectivity status")
    uptime_seconds: float = Field(..., examples=[42.5], description="Process uptime in seconds")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check & system status",
    description="Returns service health, database connectivity, environment mode, and uptime.",
)
def check_health(db: Session = Depends(get_db)) -> HealthResponse:
    """Perform liveness and database connectivity checks."""
    db_status = "connected"
    try:
        # Verify database connection with a lightweight probe
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning(f"Health probe database check failed: {exc}")
        db_status = "unreachable"

    overall_status = "healthy" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        service=settings.app_name,
        version=settings.version,
        environment=settings.env,
        git_sha=settings.git_sha,
        database=db_status,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )
