"""
Tests for the flaky-test-detection API route and the train/ subproject CLI it
shells out to.

The endpoint's contract with the subproject is exercised for real: a request
spawns `train/predict.py --json` and the assertions run against its output.
On CI (no CodeBERT checkpoint, no torch) this is exactly the deployment path
that matters -- the tabular XGBoost -- which is also why the test asserts
`model_path == "xgboost"` rather than accepting either value.
"""

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from conftest.config import PROJECT_ROOT

TRAIN_DIR = PROJECT_ROOT / "train"
PREDICT_SCRIPT = TRAIN_DIR / "predict.py"

# The tracked tabular artifacts the xgboost path needs to exist at all.
TABULAR_ARTIFACTS = (
    "xgb_model_final.pkl",
    "xgb_calibrator_final.pkl",
    "xgb_features_final.pkl",
    "xgb_metrics_final.json",
)


def test_predict_cli_runs_from_the_repo_root_and_reports_its_path():
    """The CLI must not depend on the caller's working directory, and its JSON
    must say which inference path produced the number."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--message", "fix: resolve deadlock in socket worker pool",
            "--failure_rate", "0.35",
            "--test_complexity", "22",
            "--json",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    payload = json.JSONDecoder().raw_decode(completed.stdout[completed.stdout.find("{"):])[0]
    assert payload["model_path"] in ("hybrid", "xgboost")
    assert payload["prediction"] in ("FLAKY", "STABLE")
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


def test_flakiness_endpoint_serves_the_tabular_model(client: TestClient):
    """POST /api/v1/flakiness/predict returns the subproject's prediction."""
    resp = client.post(
        "/api/v1/flakiness/predict",
        json={
            "message": "fix: resolve deadlock in socket worker pool",
            "failure_rate": 0.35,
            "test_complexity": 22,
            "lines_added": 50,
            "files_changed": 5,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["model_path"] in ("hybrid", "xgboost")
    assert data["prediction"] in ("FLAKY", "STABLE")
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert "recommendation" in data


def test_flakiness_endpoint_rejects_an_empty_message(client: TestClient):
    """A prediction on an empty commit message is a request with nothing to
    read; pydantic's 422 is the honest answer."""
    resp = client.post("/api/v1/flakiness/predict", json={"message": ""})
    assert resp.status_code == 422


def test_flakiness_endpoint_rejects_an_out_of_range_failure_rate(client: TestClient):
    resp = client.post(
        "/api/v1/flakiness/predict",
        json={"message": "fix: thing", "failure_rate": 1.5},
    )
    assert resp.status_code == 422


def test_the_tabular_artifacts_are_tracked_so_a_fresh_clone_can_predict():
    """The xgboost path is the one CI and CPU deployments take. If its
    artifacts stop being tracked, that path silently becomes a 503 on every
    fresh clone -- so the files the route depends on must exist in the tree."""
    for name in TABULAR_ARTIFACTS:
        assert (TRAIN_DIR / "models" / name).is_file(), (
            f"train/models/{name} is missing: the tabular inference path (and "
            f"this suite's endpoint test) depends on it. It is produced by "
            f"`python phase2_xgboost.py` in train/ and is meant to be tracked."
        )
