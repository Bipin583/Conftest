"""Flask service exposing the selection pipeline over HTTP.

Endpoints:

========================  ======  ====================================
Route                     Method  Purpose
========================  ======  ====================================
``/predict``              POST    Select tests for one change
``/predict/batch``        POST    Select tests for several changes
``/health``               GET     Liveness plus artefact readiness
``/metrics``              GET     Held-out metrics and the guarantee
========================  ======  ====================================

The artefacts are loaded once at import and reused, because loading them per
request would dominate the response time and defeat the point of the service.
A failure to load is not fatal at start-up: ``/health`` reports the service as
degraded and the prediction routes return 503, which is what lets a container
come up and be diagnosed rather than crash-looping.

Example:
    $ python api/server.py
    $ curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \\
        -d '{"tests": [{"test_id": "tests/test_api.py::test_ok"}]}'
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, load_config  # noqa: E402
from models.pipeline import PipelineError, SelectionPipeline  # noqa: E402

LOGGER = get_logger(__name__)

#: Refuse payloads above this many candidate tests, so one request cannot
#: exhaust the worker's memory. Large suites should use /predict/batch.
MAX_TESTS_PER_REQUEST = 20_000
MAX_CHANGES_PER_BATCH = 100


class _State:
    """Process-wide holder for the lazily loaded pipeline.

    Attributes:
        pipeline: The loaded pipeline, or ``None`` if loading failed.
        error: The load failure message, if any.
    """

    pipeline: Optional[SelectionPipeline] = None
    error: Optional[str] = None


def _ensure_pipeline() -> Tuple[Optional[SelectionPipeline], Optional[str]]:
    """Return the loaded pipeline, attempting a load on first use.

    Returns:
        A ``(pipeline, error)`` pair; exactly one is non-``None``.
    """
    if _State.pipeline is None and _State.error is None:
        try:
            _State.pipeline = SelectionPipeline.load()
            LOGGER.info("Pipeline loaded; serving with a %s guarantee.", _State.pipeline.selector.guarantee)
        except PipelineError as exc:
            _State.error = str(exc)
            LOGGER.error("Pipeline unavailable: %s", exc)
    return _State.pipeline, _State.error


def _extract_tests(payload: Any) -> List[Dict[str, Any]]:
    """Pull the candidate-test list out of a request body.

    Accepts ``{"tests": [...]}``, a bare list, or a single object, and merges
    any top-level ``change`` block into every test row so a caller does not have
    to repeat the change features per test.

    Args:
        payload: The decoded JSON body.

    Returns:
        A list of record dictionaries.

    Raises:
        ValueError: If the payload has no usable test list.
    """
    if isinstance(payload, list):
        records = payload
        change: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        records = payload.get("tests", payload.get("candidates"))
        change = payload.get("change") or {}
        if records is None:
            # A single flat object is a valid one-test request.
            records = [payload] if any(k not in {"change", "min_tests", "max_tests"} for k in payload) else None
    else:
        raise ValueError("Request body must be a JSON object or array.")

    if not records:
        raise ValueError("No candidate tests supplied; expected a non-empty 'tests' array.")
    if not isinstance(records, list):
        raise ValueError("'tests' must be an array.")
    if len(records) > MAX_TESTS_PER_REQUEST:
        raise ValueError(f"Too many candidate tests ({len(records)}); the limit is {MAX_TESTS_PER_REQUEST}.")

    merged: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every entry in 'tests' must be an object.")
        # Test-level keys win, so a caller can override the shared change block.
        merged.append({**change, **record})
    return merged


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Build the Flask application.

    Args:
        config: Parsed configuration. Defaults to ``config.yaml``.

    Returns:
        The configured application.
    """
    app = Flask(__name__)
    cfg = config or load_config()
    app.config["CTS_CONFIG"] = cfg

    @app.errorhandler(404)
    def _not_found(_: Any):
        """Return a JSON 404 rather than Flask's HTML page."""
        return jsonify({"error": "not found", "routes": ["/predict", "/predict/batch", "/health", "/metrics"]}), 404

    @app.errorhandler(405)
    def _bad_method(_: Any):
        """Return a JSON 405."""
        return jsonify({"error": "method not allowed"}), 405

    @app.errorhandler(500)
    def _server_error(exc: Any):  # pragma: no cover - safety net
        """Return a JSON 500 without leaking a stack trace to the caller."""
        LOGGER.exception("Unhandled error: %s", exc)
        return jsonify({"error": "internal server error"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """Report liveness and whether every artefact loaded.

        Returns:
            ``200`` when ready, ``503`` when the artefacts are missing.
        """
        pipeline, error = _ensure_pipeline()
        if pipeline is None:
            return jsonify({"status": "degraded", "ready": False, "error": error}), 503
        return jsonify(
            {
                "status": "ok",
                "ready": True,
                "guarantee": pipeline.selector.guarantee,
                "target_coverage": pipeline.selector.coverage,
                "confidence": pipeline.selector.confidence,
                "probability_floor": round(float(pipeline.selector.probability_floor), 6),
                "n_features": len(pipeline.feature_columns),
            }
        ), 200

    @app.route("/metrics", methods=["GET"])
    def metrics():
        """Expose the most recent held-out evaluation.

        Returns:
            ``200`` with the evaluation report, or ``404`` if none exists yet.
        """
        import json

        from common import resolve_path

        report_path = resolve_path(Path(cfg["artifacts"]["reports_dir"]) / "evaluation.json")
        if not report_path.is_file():
            return jsonify({"error": "no evaluation report; run 'python cli.py evaluate'"}), 404

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return jsonify({"error": f"could not read evaluation report: {exc}"}), 500

        selected = report.get("metrics", {})
        return jsonify(
            {
                "split": report.get("split"),
                "n_rows": report.get("n_rows"),
                "n_failures": report.get("n_failures"),
                "coverage": selected.get("empirical_coverage"),
                "selection_rate": selected.get("selection_rate"),
                "cost_reduction": selected.get("cost_reduction"),
                "precision": selected.get("precision"),
                "ece": selected.get("ece"),
                "commit_full_catch_rate": selected.get("commit_full_catch_rate"),
                "guarantee": report.get("guarantee"),
                "acceptance": report.get("acceptance"),
                "all_targets_passed": report.get("all_targets_passed"),
                "baselines": report.get("baselines"),
            }
        ), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        """Select the tests to run for a single change.

        Returns:
            ``200`` with predictions, ``400`` on a malformed body, ``503`` when
            the artefacts are unavailable.
        """
        pipeline, error = _ensure_pipeline()
        if pipeline is None:
            return jsonify({"error": error, "hint": "run the training pipeline first"}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "request body must be valid JSON"}), 400

        try:
            records = _extract_tests(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        overrides = payload if isinstance(payload, dict) else {}
        try:
            result = pipeline.predict(
                records,
                min_tests=overrides.get("min_tests"),
                max_tests=overrides.get("max_tests"),
            )
        except PipelineError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result), 200

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """Select tests for several changes in one call.

        Each change is scored independently, so one malformed entry returns an
        error for that entry alone rather than failing the whole batch.

        Returns:
            ``200`` with one result per change, ``400`` on a malformed body.
        """
        pipeline, error = _ensure_pipeline()
        if pipeline is None:
            return jsonify({"error": error, "hint": "run the training pipeline first"}), 503

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "changes" not in payload:
            return jsonify({"error": "expected a JSON object with a 'changes' array"}), 400

        changes = payload["changes"]
        if not isinstance(changes, list) or not changes:
            return jsonify({"error": "'changes' must be a non-empty array"}), 400
        if len(changes) > MAX_CHANGES_PER_BATCH:
            return jsonify({"error": f"too many changes ({len(changes)}); the limit is {MAX_CHANGES_PER_BATCH}"}), 400

        results: List[Dict[str, Any]] = []
        for index, change in enumerate(changes):
            identifier = change.get("commit_sha", f"change_{index}") if isinstance(change, dict) else f"change_{index}"
            try:
                records = _extract_tests(change)
                outcome = pipeline.predict(
                    records,
                    min_tests=payload.get("min_tests"),
                    max_tests=payload.get("max_tests"),
                )
                results.append({"commit_sha": identifier, **outcome})
            except (ValueError, PipelineError) as exc:
                results.append({"commit_sha": identifier, "error": str(exc)})

        succeeded = [r for r in results if "error" not in r]
        total_candidates = sum(r["summary"]["n_candidates"] for r in succeeded)
        total_selected = sum(r["summary"]["n_selected"] for r in succeeded)
        return jsonify(
            {
                "results": results,
                "batch_summary": {
                    "n_changes": len(changes),
                    "n_succeeded": len(succeeded),
                    "n_failed": len(results) - len(succeeded),
                    "n_candidates": total_candidates,
                    "n_selected": total_selected,
                    "cost_reduction": round(1.0 - total_selected / total_candidates, 6) if total_candidates else 0.0,
                },
            }
        ), 200

    return app


def main(argv: Optional[List[str]] = None) -> int:
    """Run the development server.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Serve the conformal test-selection API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask's debug reloader.")
    args = parser.parse_args(argv)

    app = create_app()
    LOGGER.info("Serving on http://%s:%d", args.host, args.port)
    # Flask's development server is single-threaded and not hardened; a real
    # deployment should front this with gunicorn or waitress.
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
