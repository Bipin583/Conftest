"""
Unit tests for the scoring latency benchmark harness.

These exercise the timing harness, not a published latency figure, so the model here is a
throwaway two-member ensemble over explicit arrays. The number that goes into
reports/latency_benchmark.json is measured by scripts/run_latency_benchmark.py against the
serialized 5-seed ensemble and real rows from the test split.
"""

import numpy as np
import pytest

from conftest.evaluation.benchmark_latency import LatencyBenchmarkSuite
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.policy import SelectivePredictionPolicy


@pytest.fixture(scope="module")
def bundle():
    """A minimal trained bundle plus a block of feature rows to score."""
    n_feats = len(FEATURE_NAMES)
    rows = np.linspace(-1.0, 1.0, 80 * n_feats, dtype=np.float32).reshape(80, n_feats)
    labels = np.zeros(80, dtype=int)
    labels[::7] = 1

    ensemble = EnsembleUncertaintyPredictor(seeds=[42, 101], n_estimators=10)
    ensemble.train(rows, labels)
    calibrator = ConfidenceCalibrator(method="temperature_scaling")
    calibrator.fit(ensemble.predict_with_uncertainty(rows)["mean_prob"], labels)
    return ensemble, calibrator, SelectivePredictionPolicy(), rows


def test_benchmark_reports_every_stage(bundle):
    """The report names each timed stage and separates the warm-up from the statistics."""
    ensemble, calibrator, policy, rows = bundle
    suite = LatencyBenchmarkSuite(ensemble, calibrator, policy, rows)
    report = suite.benchmark_pipeline(num_iterations=10, batch_size=20, warmup=2)

    for stage in ("array_sanitisation", "ensemble_inference", "calibration",
                  "policy_decision", "scoring_total"):
        assert stage in report["stages"]
        assert report["stages"][stage]["mean_ms"] >= 0.0

    assert report["warmup_batches"] == 2
    assert len(report["warmup_batches_ms"]) == 2
    assert report["timed_statistics_exclude_warmup"] is True
    assert report["cold_start_first_batch_ms"] == report["warmup_batches_ms"][0]
    assert report["rows_available"] == len(rows)


def test_batches_are_deterministic_and_wrap(bundle):
    """
    The same iteration index always yields the same rows, and the window wraps.

    A benchmark that samples randomly reports a different figure every run, which is an
    invitation to re-roll until an SLA passes.
    """
    ensemble, calibrator, policy, rows = bundle
    suite = LatencyBenchmarkSuite(ensemble, calibrator, policy, rows)

    first = suite._batch(0, 30)
    assert np.array_equal(first, suite._batch(0, 30))
    assert not np.array_equal(first, suite._batch(1, 30))
    # 80 rows, 30 per batch: the third batch has to wrap past the end.
    assert suite._batch(2, 30).shape == (30, len(FEATURE_NAMES))


def test_suite_refuses_to_invent_rows(bundle):
    """An empty row block is an error, not a cue to generate features."""
    ensemble, calibrator, policy, rows = bundle
    with pytest.raises(ValueError, match="does not generate them"):
        LatencyBenchmarkSuite(ensemble, calibrator, policy, np.empty((0, len(FEATURE_NAMES))))
