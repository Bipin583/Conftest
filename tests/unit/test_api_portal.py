"""
Tests for the portal data endpoints that back the React dashboard.

These serve report artifacts over HTTP so the SPA needs no filesystem or Python
access. The repo ships the artifacts, so the happy path is 200 with the shape
the frontend consumes; the missing-artifact path is covered by pointing a loader
at an absent file and asserting the 503 contract.
"""

from fastapi.testclient import TestClient


def test_headline_shape(client: TestClient):
    """GET /api/v1/headline returns provenance + the four-metric KPI row."""
    resp = client.get("/api/v1/headline")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["provenance"]) == {"real_labels", "detail"}
    assert len(data["metrics"]) == 4
    for metric in data["metrics"]:
        assert set(metric) == {"label", "value", "note", "source", "measured"}
        # Honesty contract: a value is present iff the metric was measured.
        assert metric["measured"] == (metric["value"] is not None)


def test_baseline_shape(client: TestClient):
    """GET /api/v1/baseline parses the percentage cells to numbers (or None)."""
    resp = client.get("/api/v1/baseline")
    assert resp.status_code == 200
    data = resp.json()
    assert "provenance" in data
    assert len(data["rows"]) >= 1
    row = data["rows"][0]
    assert "strategy" in row
    for key in ("failure_recall_pct", "time_reduction_pct", "test_reduction_pct"):
        assert row[key] is None or isinstance(row[key], (int, float))


def test_uncertainty_shape(client: TestClient):
    """GET /api/v1/uncertainty bundles analysis, policy thresholds and ensemble metadata."""
    resp = client.get("/api/v1/uncertainty")
    assert resp.status_code == 200
    assert set(resp.json()) == {"analysis", "policy", "ensemble"}


def test_explanations_served(client: TestClient):
    """GET /api/v1/explanations returns the global SHAP report object."""
    resp = client.get("/api/v1/explanations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_missing_artifact_is_503(client: TestClient, monkeypatch):
    """A loader that cannot find its artifact answers 503 naming the producing script."""
    from conftest.evaluation.headline import MissingArtifact
    import conftest.api.routes.portal as portal

    def _raise(*_args, **_kwargs):
        raise MissingArtifact(portal.EXPLANATIONS_JSON, "python scripts/generate_explanations.py")

    monkeypatch.setattr(portal, "read_json", _raise)
    resp = client.get("/api/v1/explanations")
    assert resp.status_code == 503
    assert "generate_explanations.py" in resp.json()["detail"]
