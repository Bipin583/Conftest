"""
Unit tests for Confidence Calibration, ECE / MCE, and Reliability Diagrams.
"""

from pathlib import Path

import numpy as np
import pytest

from conftest.models.calibration import (
    ConfidenceCalibrator,
    IsotonicCalibrator,
    TemperatureScalingCalibrator,
    compute_ece,
)


def test_ece_computation_perfect_calibration():
    """Verify ECE approaches zero when predicted probabilities match true empirical rates."""
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.array([0.10]*100)  # Average confidence exactly equals 10/100 = 0.10

    ece, mce, _bins = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-3)
    assert mce == pytest.approx(0.0, abs=1e-3)


def test_ece_computation_severe_miscalibration():
    """Verify ECE is high when model is overconfident on incorrect classes."""
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.array([0.95]*100)  # Extreme overconfidence (predicts 95% fail, but only 10% fail)

    ece, mce, _bins = compute_ece(y_true, y_prob, n_bins=10)
    assert ece > 0.70  # |0.10 - 0.95| = 0.85
    assert mce > 0.70


def test_temperature_scaling_fitting_and_calibration():
    """Verify TemperatureScalingCalibrator fits scalar temperature T and maps probabilities smoothly."""
    rng = np.random.RandomState(42)
    # Overconfident uncalibrated probabilities
    raw_probs = np.clip(rng.beta(0.5, 0.5, size=200), 0.05, 0.95)
    y_true = (raw_probs > 0.6).astype(int)

    cal = TemperatureScalingCalibrator()
    cal.fit(raw_probs, y_true)

    assert cal.temperature > 0.0

    cal_probs = cal.calibrate(raw_probs)
    assert len(cal_probs) == 200
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Monotonicity check: higher raw prob should map to higher calibrated prob
    assert cal_probs[np.argmax(raw_probs)] >= cal_probs[np.argmin(raw_probs)]


def test_isotonic_calibrator_monotonicity():
    """Verify IsotonicCalibrator preserves monotonic order."""
    raw_probs = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
    y_true = np.array([0, 0, 0, 1, 1])

    cal = IsotonicCalibrator()
    cal.fit(raw_probs, y_true)

    cal_probs = cal.calibrate(raw_probs)
    assert np.all(np.diff(cal_probs) >= 0.0)  # Monotonically non-decreasing


def test_uncalibrated_calibrator_is_an_explicit_serializable_identity(tmp_path: Path):
    """A declined calibration still produces a truthful downstream artifact."""
    raw_probs = np.array([0.1, 0.3, 0.9], dtype=np.float64)
    cal = ConfidenceCalibrator(method="uncalibrated")

    np.testing.assert_allclose(cal.calibrate(raw_probs), raw_probs, rtol=1e-6)
    save_path = tmp_path / "identity.joblib"
    cal.save(str(save_path))

    loaded = ConfidenceCalibrator.load(str(save_path))
    assert loaded.method == "uncalibrated"
    np.testing.assert_allclose(loaded.calibrate(raw_probs), raw_probs, rtol=1e-6)


def test_confidence_calibrator_save_and_load(tmp_path: Path):
    """Verify ConfidenceCalibrator serialization and deserialization."""
    raw_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_true = np.array([0, 0, 1, 1, 1])

    cal = ConfidenceCalibrator(method="isotonic")
    cal.fit(raw_probs, y_true)
    orig_calibrated = cal.calibrate(raw_probs)

    save_path = tmp_path / "calibrator.joblib"
    cal.save(str(save_path))
    assert save_path.exists()

    loaded = ConfidenceCalibrator.load(str(save_path))
    loaded_calibrated = loaded.calibrate(raw_probs)

    np.testing.assert_allclose(orig_calibrated, loaded_calibrated, rtol=1e-5)
