"""
Comprehensive Integration and Unit tests for FastAPI REST Endpoints.
"""

import json

from fastapi.testclient import TestClient
import pytest

from conftest.features.pipeline import FEATURE_NAMES


def test_root_endpoint(client: TestClient):
    """Verify GET / returns documentation links and metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "app" in data
    assert "endpoints" in data
    assert data["endpoints"]["select"] == "/api/v1/select"


def test_health_endpoint(client: TestClient):
    """Verify GET /api/v1/health probe."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_repositories_crud_endpoints(client: TestClient):
    """Verify POST and GET /api/v1/repositories."""
    # 1. Create repository
    payload = {
        "full_name": "owner/api-test-repo",
        "url": "https://github.com/owner/api-test-repo",
        "local_path": "./tests/sample_suite",
    }
    resp = client.post("/api/v1/repositories", json=payload)
    assert resp.status_code == 201
    created = resp.json()
    repo_id = created["id"]
    assert created["full_name"] == "owner/api-test-repo"

    # 2. List repositories
    list_resp = client.get("/api/v1/repositories")
    assert list_resp.status_code == 200
    repos = list_resp.json()
    assert len(repos) >= 1

    # 3. Get repository details
    detail_resp = client.get(f"/api/v1/repositories/{repo_id}")
    assert detail_resp.status_code == 200
    det = detail_resp.json()
    assert det["id"] == repo_id
    assert "total_test_cases" in det


def test_select_endpoint_fast_and_fallback(client: TestClient):
    """Verify POST /api/v1/select executes end-to-end RTS prediction."""
    payload = {
        "repository_name": "owner/api-test-repo",
        "commit_sha": "abc123456789",
        "changed_files": [
            {"file_path": "src_app/auth.py", "change_type": "M", "lines_added": 15, "lines_deleted": 3}
        ],
        "commit_message": "fix: update authentication token expiration",
        "budget_ratio": 0.25,
        "repo_path": "./tests/sample_suite",
        "execute": False,
    }
    resp = client.post("/api/v1/select", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "decision_mode" in data
    assert data["decision_mode"] in ("FAST_SELECTED", "SAFE_FULL_SUITE")
    assert "selected_test_ids" in data
    assert "ranked_tests" in data
    assert len(data["ranked_tests"]) >= 5
    assert "markdown_summary" in data
    assert "🛡️ ConfTest" in data["markdown_summary"]


def test_explain_endpoint_shap_and_rules(client: TestClient):
    """Verify POST /api/v1/explain returns SHAP drivers and developer cards."""
    # Build 32-feature vector dict
    dummy_feats = {name: 0.0 for name in FEATURE_NAMES}
    dummy_feats["dep_is_direct_import"] = 1.0
    dummy_feats["hist_recent_10_failure_rate"] = 0.20
    dummy_feats["diff_total_churn"] = 45.0

    payload = {
        "test_id": "tests/sample_suite/tests/test_auth.py::test_password_hashing",
        "features": dummy_feats,
        "top_k": 3,
    }
    resp = client.post("/api/v1/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["test_id"] == payload["test_id"]
    assert "predicted_probability" in data
    assert "primary_reasons" in data
    assert len(data["top_risk_increasing_features"]) <= 3


def test_calibration_endpoint(client: TestClient):
    """
    Verify GET /api/v1/calibration returns ECE and Brier diagnostics.

    Reads whatever report the repo currently carries, so the assertion has to hold
    for either outcome: a method was chosen, or none beat the noise. This
    previously required `data["calibrated"]["ece"]` unconditionally, which made
    declining to calibrate look like an endpoint failure.
    """
    resp = client.get("/api/v1/calibration")
    assert resp.status_code == 200
    data = resp.json()

    assert "best_method" in data
    assert "ece" in data["uncalibrated"], "the baseline row is always measured"
    if data["best_method"] == "uncalibrated":
        assert data["calibrated"] is None
    else:
        assert "ece" in data["calibrated"]


def test_analytics_endpoint(client: TestClient):
    """Verify GET /api/v1/analytics aggregates database metrics."""
    resp = client.get("/api/v1/analytics")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_repositories" in data
    assert "total_decisions" in data
    assert "average_test_reduction_pct" in data
    assert "recent_decisions" in data


# --------------------------------------------------------------------------
# The calibration endpoint reports measurements, or nothing
# --------------------------------------------------------------------------


def _report(best_method: str, **extra) -> dict:
    """A calibration report of the shape scripts/calibrate_model.py writes."""
    def block(method, ece, mce, brier, **rest):
        return {"method": method, "ece": ece, "mce": mce, "brier_score": brier, **rest}

    return {
        "best_method": best_method,
        "fitted_temperature": 1.1834,
        "selection": {
            "basis": "bootstrap",
            "reason": "chose uncalibrated: no candidate improved ECE measurably",
            "resampling_unit": "mutant",
        },
        "test_metrics": {
            "resampling_unit": "mutant",
            "uncalibrated": block("uncalibrated", 0.0258, 0.2222, 0.0449),
            "temperature_scaling": block(
                "temperature_scaling", 0.0192, 0.8943, 0.0449,
                ece_reduction_pct=25.47,
                ece_vs_uncalibrated={
                    "point": -0.00657, "ci_lower": -0.01118,
                    "ci_upper": 0.01099, "excludes_zero": False,
                },
            ),
        },
        "reliability_diagram_bins": {"temperature_scaling": [{"bin": 1}]},
        **extra,
    }


@pytest.fixture
def report_at(tmp_path, monkeypatch):
    """Point the route at a report this test controls, or at a missing path."""
    from conftest.api.routes import calibration as route

    def place(payload):
        path = tmp_path / "calibration_report.json"
        if payload is not None:
            path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(route, "REPORT_PATH", path)
        return path

    return place


def test_a_missing_report_is_503_and_not_a_set_of_defaults(client: TestClient, report_at):
    """
    The endpoint invented a complete calibration result when no report existed.

    It served ECE 0.0192, MCE 0.8943, a 25.47% ECE reduction and T=0.9275 as
    literals, indistinguishable from measurement, and asserted a gain that the
    paired interval over resampled mutants does not support. Absent measurement is
    503; there is nothing to fall back to.
    """
    report_at(None)
    resp = client.get("/api/v1/calibration")

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "calibrate_model.py" in detail
    assert "0.0192" not in detail and "25.47" not in detail


def test_declining_to_calibrate_is_reported_as_no_calibrated_model(
    client: TestClient, report_at
):
    """
    `best_method: uncalibrated` must not come back as a calibrated model.

    `test_metrics["uncalibrated"]` exists, so the old lookup found it and returned
    it in the `calibrated` field -- reporting a calibrated model for a run that
    decided against calibrating. That decision is now the expected outcome
    whenever no candidate's ECE gain clears the noise.
    """
    report_at(_report("uncalibrated"))
    data = client.get("/api/v1/calibration").json()

    assert data["best_method"] == "uncalibrated"
    assert data["calibrated"] is None
    assert data["temperature"] is None
    assert "no candidate improved ECE" in data["selection_reason"]
    assert data["selection_basis"] == "bootstrap"


def test_the_served_temperature_is_the_one_that_was_fitted(client: TestClient, report_at):
    """
    The route returned a hardcoded 0.9275 for every temperature-scaled run.

    A rerun that fits a different T -- which any change to the validation split
    produces -- was reported under the old constant.
    """
    report_at(_report("temperature_scaling"))
    data = client.get("/api/v1/calibration").json()

    assert data["temperature"] == pytest.approx(1.1834)
    assert data["calibrated"]["method"] == "temperature_scaling"
    assert data["reliability_diagram_bins"] == [{"bin": 1}]


def test_the_interval_travels_with_the_metric(client: TestClient, report_at):
    """
    A client shown three bare ECE values will read the smallest as the winner.

    Here the chosen method's ECE is 25% lower and its paired difference spans
    zero, so the interval is the only part of the payload that says the gain is
    not established.
    """
    report_at(_report("temperature_scaling"))
    cal = client.get("/api/v1/calibration").json()["calibrated"]

    assert cal["ece_reduction_pct"] == 25.47
    assert cal["ece_vs_uncalibrated"]["excludes_zero"] is False
    assert cal["ece_vs_uncalibrated"]["ci_upper"] > 0.0
    assert cal["mce_vs_uncalibrated"] is None, "absent in the report, absent here"


def test_a_report_without_intervals_still_serves_its_point_estimates(
    client: TestClient, report_at
):
    """
    Reports predating the bootstrap path, and any built without cluster labels.

    `selection_basis` is what tells a client which standard the choice was made
    on, so the metrics stay readable without it while the missing evidence shows.
    """
    stale = _report("temperature_scaling")
    del stale["selection"]
    del stale["fitted_temperature"]
    del stale["test_metrics"]["temperature_scaling"]["ece_vs_uncalibrated"]

    report_at(stale)
    data = client.get("/api/v1/calibration").json()

    assert data["calibrated"]["ece"] == 0.0192
    assert data["calibrated"]["ece_vs_uncalibrated"] is None
    assert data["selection_basis"] is None
    assert data["temperature"] is None
