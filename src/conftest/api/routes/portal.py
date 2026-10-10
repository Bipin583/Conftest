"""
ConfTest Portal Data API Routes.

The React dashboard cannot import the Python package or read report artifacts
off disk the way the Streamlit app did, so these endpoints expose the same
artifacts over HTTP. Every loader reuses `conftest.evaluation.headline`, and a
missing artifact becomes a 503 naming the script that produces it -- the same
contract the calibration route already follows, so no endpoint ever serves a
fabricated stand-in.

Endpoints (mounted under /api/v1):
    GET /headline      front-page KPI row + provenance
    GET /baseline      reports/baseline_comparison.csv as parsed JSON rows
    GET /uncertainty   uncertainty analysis + policy config + ensemble metadata
    GET /explanations  global SHAP report reports/explanations.json
"""

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from conftest.config import PROJECT_ROOT
from conftest.evaluation.headline import (
    BASELINE_COLS,
    STRATEGY_COL,
    MissingArtifact,
    as_number,
    headline_metrics,
    provenance,
    read_baseline_rows,
    read_json,
)
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Dashboard Data"])

ENSEMBLE_METADATA = Path("models/ensembles/5_seed_lgbm/ensemble_metadata.json")
UNCERTAINTY_JSON = Path("reports/uncertainty_analysis.json")
POLICY_CONFIG = Path("models/policy_config.json")
EXPLANATIONS_JSON = Path("reports/explanations.json")


def _artifact_unavailable(exc: MissingArtifact) -> HTTPException:
    """A missing report as the 503 the calibration route established."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            f"{exc.path.as_posix()} has not been produced yet. This endpoint "
            f"reports measurements only; it has no default to serve. Produce it "
            f"with: {exc.produced_by}"
        ),
    )


@router.get("/headline")
def get_headline() -> Dict[str, Any]:
    """
    The front-page KPI row and the provenance banner.

    Each metric carries `measured=False` and `value=None` when its artifact is
    absent, exactly as the Streamlit home page rendered it -- the React client
    shows "not measured", never a placeholder number.
    """
    prov = provenance(PROJECT_ROOT)
    return {
        "provenance": {"real_labels": prov.real_labels, "detail": prov.detail},
        "metrics": [
            {
                "label": h.label,
                "value": h.value,
                "note": h.note,
                "source": h.source,
                "measured": h.measured,
            }
            for h in headline_metrics(PROJECT_ROOT)
        ],
    }


@router.get("/baseline")
def get_baseline() -> Dict[str, Any]:
    """The RTS baseline-comparison table with every percentage cell parsed to a float."""
    try:
        rows = read_baseline_rows(PROJECT_ROOT)
    except MissingArtifact as exc:
        raise _artifact_unavailable(exc)

    prov = provenance(PROJECT_ROOT)
    parsed = [
        {
            "strategy": row.get(STRATEGY_COL),
            "test_reduction_pct": as_number(row.get(BASELINE_COLS["trr"])),
            "time_reduction_pct": as_number(row.get(BASELINE_COLS["etr"])),
            "failure_recall_pct": as_number(row.get(BASELINE_COLS["fr"])),
            "missed_failure_pct": as_number(row.get(BASELINE_COLS["mfr"])),
            "abstention_rate_pct": as_number(row.get(BASELINE_COLS["ar"])),
            "escaped_commits": as_number(row.get(BASELINE_COLS["escaped"])),
        }
        for row in rows
    ]
    return {
        "rows": parsed,
        "provenance": {"real_labels": prov.real_labels, "detail": prov.detail},
    }


@router.get("/uncertainty")
def get_uncertainty() -> Dict[str, Any]:
    """Ensemble disagreement analysis, the abstention policy thresholds, and ensemble metadata."""
    try:
        analysis = read_json(UNCERTAINTY_JSON, "python scripts/uncertainty_eval.py", PROJECT_ROOT)
        policy = read_json(POLICY_CONFIG, "python scripts/tune_policy.py", PROJECT_ROOT)
        ensemble = read_json(ENSEMBLE_METADATA, "python scripts/train_ensemble.py", PROJECT_ROOT)
    except MissingArtifact as exc:
        raise _artifact_unavailable(exc)
    return {"analysis": analysis, "policy": policy, "ensemble": ensemble}


@router.get("/explanations")
def get_explanations() -> Dict[str, Any]:
    """The global and per-test SHAP attribution report."""
    try:
        return read_json(EXPLANATIONS_JSON, "python scripts/generate_explanations.py", PROJECT_ROOT)
    except MissingArtifact as exc:
        raise _artifact_unavailable(exc)
