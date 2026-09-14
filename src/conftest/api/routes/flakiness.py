"""
ConfTest Flaky Test Detection API Route.

Endpoint: POST /api/v1/flakiness/predict

Backed by the standalone hybrid flaky-test-detection subproject in ``train/``.
The request is answered by running ``python train/predict.py --json`` in a
bounded subprocess rather than importing the subproject: ``train/`` is a
self-contained script collection, not a package, and the process boundary is
what keeps it that way. predict.py serves the hybrid model (fine-tuned
CodeBERT + XGBoost) when the checkpoint is on disk and torch/transformers
import, and the tabular-only XGBoost otherwise -- the response's ``model_path``
says which one ran.
"""

import json
import subprocess
import sys
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from conftest.api.schemas import FlakinessPredictRequestSchema, FlakinessPredictResponseSchema
from conftest.config import PROJECT_ROOT
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/flakiness", tags=["Flaky Test Detection"])

TRAIN_DIR = PROJECT_ROOT / "train"
PREDICT_SCRIPT = TRAIN_DIR / "predict.py"

# The tabular path loads a 16 MB model and returns in well under a second; the
# hybrid path loads a 500 MB CodeBERT first. A bound well above both, but one
# that keeps a wedged subprocess from holding a worker forever.
SUBPROCESS_TIMEOUT_SECONDS = 120


def _extract_json(stdout: str) -> Dict[str, Any]:
    """Pull the JSON object out of predict.py's banner-decorated stdout.

    The script prints progress banners before and a rule after the JSON, so
    the object is located by its first opening brace rather than assumed to be
    the whole stream.
    """
    start = stdout.find("{")
    if start < 0:
        raise ValueError("predict.py printed no JSON object")
    result, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return result


@router.post(
    "/predict",
    response_model=FlakinessPredictResponseSchema,
    status_code=status.HTTP_200_OK,
)
def predict_flakiness(request: FlakinessPredictRequestSchema) -> FlakinessPredictResponseSchema:
    """
    Predict whether a test touched by a commit is flaky.

    503 when the subproject's model artifacts are absent: the endpoint reports
    model output only, and the error names the training command that produces
    the missing artifacts rather than serving a guess.
    """
    if not PREDICT_SCRIPT.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Flaky-test-detection subproject missing: {PREDICT_SCRIPT} not found.",
        )

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--message", request.message,
        "--body", request.body,
        "--failure_rate", str(request.failure_rate),
        "--test_complexity", str(request.test_complexity),
        "--lines_added", str(request.lines_added),
        "--lines_deleted", str(request.lines_deleted),
        "--files_changed", str(request.files_changed),
        "--json",
    ]

    try:
        # cwd is pinned to train/ so the script's sibling-relative model paths
        # resolve no matter where the API process itself was started from.
        completed = subprocess.run(
            cmd,
            cwd=str(TRAIN_DIR),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"train/predict.py did not answer within {SUBPROCESS_TIMEOUT_SECONDS}s.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Flakiness prediction timed out after {SUBPROCESS_TIMEOUT_SECONDS}s.",
        )

    if completed.returncode != 0:
        # predict.py exits 2 with a stderr naming the producing phase when its
        # artifacts are missing; anything else is unexpected and logged whole.
        logger.error(
            f"train/predict.py exited {completed.returncode}: {completed.stderr.strip()}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Flakiness model artifacts unavailable (predict.py exit "
                f"{completed.returncode}): {completed.stderr.strip()} "
                f"Train them with `python phase2_xgboost.py` in train/."
            ),
        )

    try:
        payload = _extract_json(completed.stdout)
        return FlakinessPredictResponseSchema(**payload)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        logger.error(f"Could not parse predict.py output: {exc}\n{completed.stdout}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Flakiness predictor returned an unreadable result: {exc}",
        )
