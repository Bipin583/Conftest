"""
ConfTest Visual Analytics & Telemetry API Route.

Endpoint: GET /api/v1/analytics
Aggregates repository stats, time savings, selective prediction abstention rates, and regression failure recall.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from conftest.api.schemas import AnalyticsSummarySchema
from conftest.db.session import get_db
from conftest.db.models import Repository, Commit, SelectionDecision, Outcome
from conftest.db import crud

router = APIRouter(prefix="/analytics", tags=["Analytics & Telemetry"])


@router.get("", response_model=AnalyticsSummarySchema, status_code=status.HTTP_200_OK)
def get_analytics_summary(db: Session = Depends(get_db)) -> AnalyticsSummarySchema:
    """
    Retrieve system-wide aggregated telemetry and efficiency metrics.
    """
    total_repos = db.execute(select(func.count(Repository.id))).scalar() or 0
    total_commits = db.execute(select(func.count(Commit.id))).scalar() or 0
    total_decisions = db.execute(select(func.count(SelectionDecision.id))).scalar() or 0

    fast_mode_count = db.execute(
        select(func.count(SelectionDecision.id)).where(SelectionDecision.abstained == False)
    ).scalar() or 0
    abstention_count = db.execute(
        select(func.count(SelectionDecision.id)).where(SelectionDecision.abstained == True)
    ).scalar() or 0

    # An average over no decisions does not exist. These were `or 0.0` and
    # `or 0.015`: an empty database reported a plausible-looking 0.015 mean
    # uncertainty in the same field, and with the same type, as a real average.
    # None serialises to null, which a caller can tell apart from a measurement.
    avg_savings = db.execute(
        select(func.avg(SelectionDecision.estimated_saving))
    ).scalar()

    avg_uncertainty = db.execute(
        select(func.avg(SelectionDecision.uncertainty_score))
    ).scalar()

    total_detected_fails = db.execute(
        select(func.sum(Outcome.detected_failures))
    ).scalar() or 0

    total_missed_fails = db.execute(
        select(func.sum(Outcome.missed_failures))
    ).scalar() or 0

    # Recent decisions
    recent_stmt = (
        select(SelectionDecision, Commit.sha)
        .join(Commit, Commit.id == SelectionDecision.commit_id)
        .order_by(SelectionDecision.created_at.desc())
        .limit(10)
    )
    recent_rows = db.execute(recent_stmt).all()
    recent_list = []
    for dec, sha in recent_rows:
        recent_list.append({
            "commit_sha": sha[:8],
            "mode": dec.mode,
            "abstained": dec.abstained,
            "selected_count": dec.selected_count,
            "total_count": dec.total_count,
            # estimated_saving is 1 - selected/total: a count ratio. Reporting it
            # as time saved asserts that every test costs the same.
            "test_reduction_pct": dec.estimated_saving,
            "uncertainty": dec.uncertainty_score,
            "created_at": dec.created_at.isoformat(),
        })

    return AnalyticsSummarySchema(
        total_repositories=total_repos,
        total_commits_evaluated=total_commits,
        total_decisions=total_decisions,
        total_selective_fast_mode=fast_mode_count,
        total_safe_abstentions=abstention_count,
        average_test_reduction_pct=(
            None if avg_savings is None else round(float(avg_savings), 2)
        ),
        total_failures_detected=int(total_detected_fails),
        total_missed_failures=int(total_missed_fails),
        average_uncertainty=(
            None if avg_uncertainty is None else round(float(avg_uncertainty), 4)
        ),
        recent_decisions=recent_list,
    )
