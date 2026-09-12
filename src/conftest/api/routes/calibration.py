"""
ConfTest Calibration API Route.

Endpoint: GET /api/v1/calibration
Serves the calibration report produced by `scripts/calibrate_model.py`: ECE, MCE
and Brier for the uncalibrated model and the chosen method, each with the paired
bootstrap interval the choice was made on, plus the reliability bins.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from conftest.api.schemas import CalibrationResponseSchema, CalibrationMetricItem
from conftest.config import PROJECT_ROOT
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/calibration", tags=["Confidence Calibration"])

# Anchored to the project root, not to the working directory. As a relative
# path this resolved against wherever uvicorn happened to be started, so
# launching the API from any other directory made the endpoint answer 503
# "no calibration report" while the report sat on disk, and told the caller to
# re-run a measurement that had already been run.
REPORT_PATH = PROJECT_ROOT / "reports" / "calibration_report.json"

# Keys of a paired-difference block in the report, as score_record writes them.
DIFFERENCE_KEYS = (
    "ece_vs_uncalibrated",
    "mce_vs_uncalibrated",
    "brier_score_vs_uncalibrated",
)


def _metric_item(block: Dict[str, Any]) -> CalibrationMetricItem:
    """
    One method's metrics, carrying whatever intervals the report recorded.

    The differences are optional because a report built without cluster labels has
    none to give; absent is the honest representation of that, and the response's
    `selection_basis` says so in words.
    """
    return CalibrationMetricItem(
        method=block.get("method"),
        ece=block["ece"],
        mce=block["mce"],
        brier_score=block["brier_score"],
        ece_reduction_pct=block.get("ece_reduction_pct"),
        **{key: block.get(key) for key in DIFFERENCE_KEYS},
    )


def _calibrated_block(data: Dict[str, Any], best_method: str) -> Optional[Dict[str, Any]]:
    """
    The chosen method's metrics, or None when no method was chosen.

    `best_method == "uncalibrated"` is the outcome when no candidate improved ECE
    by more than the noise in the measurement. Looking that up in `test_metrics`
    succeeds -- the uncalibrated row is right there -- and returning it as the
    'calibrated' model would report a calibrated model where the selection
    declined to produce one.
    """
    if best_method == "uncalibrated":
        return None
    metrics = data.get("test_metrics", {})
    return metrics.get(f"{best_method}_calibration") or metrics.get(best_method)


@router.get("", response_model=CalibrationResponseSchema, status_code=status.HTTP_200_OK)
def get_calibration_diagnostics() -> CalibrationResponseSchema:
    """
    Retrieve empirical calibration diagnostics and reliability diagram data.

    503 until the report exists. There is no default: the previous fallback served
    a full set of literals (ECE 0.0192, MCE 0.8943, a 25.47% reduction, T=0.9275)
    that a client could not tell apart from a measurement, and which asserted a
    calibration gain the paired intervals do not support.
    """
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No calibration report at {REPORT_PATH}. Run "
                "`python scripts/calibrate_model.py` to measure calibration on the "
                "validation and test splits. This endpoint reports measurements "
                "only; it has no default values to serve."
            ),
        )

    try:
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        best_method = data.get("best_method", "uncalibrated")
        selection = data.get("selection", {})
        calibrated = _calibrated_block(data, best_method)

        return CalibrationResponseSchema(
            best_method=best_method,
            uncalibrated=_metric_item(data["test_metrics"]["uncalibrated"]),
            calibrated=_metric_item(calibrated) if calibrated else None,
            # Read from the report, not assumed: a rerun that fits a different T
            # must not be reported under the old one.
            temperature=(
                data.get("fitted_temperature")
                if best_method == "temperature_scaling"
                else None
            ),
            # The metric blocks above are held-out test measurements; the
            # selection was made on validation. Both labels are read from the
            # report so a rerun that changes either cannot be misreported.
            metrics_split="held-out test split",
            selection_basis=selection.get("basis"),
            selection_reason=selection.get("reason"),
            selection_split=selection.get("chosen_on"),
            resampling_unit=data.get("test_metrics", {}).get("resampling_unit"),
            reliability_diagram_bins=data.get("reliability_diagram_bins", {}).get(
                best_method, []
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed loading calibration report: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load calibration metrics: {str(exc)}",
        )
