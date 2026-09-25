"""Layer 2: probability calibration, and the ECE that decides whether it worked.

A gradient-boosted model trained with ``scale_pos_weight`` produces scores that
rank well and lie badly: they are inflated towards the positive class by the
weighting itself. Everything downstream -- the conformal threshold, the API's
reported confidence -- reads those numbers as probabilities, so the calibration
map is load-bearing rather than cosmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.calibrate import ScoreCalibrator, _safe_logit, expected_calibration_error
from tests.conftest import make_scores


# --------------------------------------------------------------------------
# Expected calibration error
# --------------------------------------------------------------------------


def test_perfect_calibration_scores_near_zero_ece():
    """Scores that *are* the event rate must report almost no error.

    Each row is drawn with its own probability and labelled from exactly that
    probability, so the model is calibrated by construction. Any sizeable ECE
    here would be an artefact of the binning, not of the scores.
    """
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.05, 0.95, size=40_000)
    labels = (rng.random(40_000) < probabilities).astype(int)
    assert expected_calibration_error(labels, probabilities, n_bins=15)["ece"] < 0.01


def test_systematically_overconfident_scores_score_high_ece():
    """Doubling every probability must be visible as miscalibration."""
    rng = np.random.default_rng(1)
    honest = rng.uniform(0.05, 0.45, size=20_000)
    labels = (rng.random(20_000) < honest).astype(int)
    inflated = np.clip(honest * 2.0, 0.0, 1.0)

    assert expected_calibration_error(labels, inflated, n_bins=15)["ece"] > 0.15


def test_ece_reports_the_worst_bin_alongside_the_average():
    """MCE is the maximum bin gap and can never be below the mean gap.

    The project rejected isotonic regression on exactly this distinction --
    lower mean error, worse worst-bin error -- so both numbers must be
    reported, not just the headline one.
    """
    labels, probabilities = make_scores(n=5000, seed=3)[1], make_scores(n=5000, seed=3)[0]
    report = expected_calibration_error(labels, probabilities, n_bins=15)
    assert report["mce"] >= report["ece"]
    assert report["n_bins"] == 15
    assert len(report["bins"]) <= 15


def test_ece_bins_account_for_every_row():
    """No row may be dropped by the binning, or the average is over the wrong n."""
    probabilities, labels = make_scores(n=4000, seed=4)
    report = expected_calibration_error(labels, probabilities, n_bins=10)
    assert sum(b["count"] for b in report["bins"]) == labels.size


# --------------------------------------------------------------------------
# The calibration map
# --------------------------------------------------------------------------


def test_platt_scaling_reduces_calibration_error_on_inflated_scores():
    """The map must actually improve ECE; that is its entire purpose."""
    rng = np.random.default_rng(5)
    honest = rng.uniform(0.02, 0.5, size=30_000)
    labels = (rng.random(30_000) < honest).astype(int)
    inflated = np.clip(honest * 1.9, 1e-6, 1 - 1e-6)

    before = expected_calibration_error(labels, inflated)["ece"]
    calibrator = ScoreCalibrator("platt").fit(inflated, labels)
    after = expected_calibration_error(labels, calibrator.transform(inflated))["ece"]

    assert after < before
    assert after < 0.08, "post-calibration ECE must clear the configured gate"


def test_calibration_preserves_the_ranking():
    """Platt scaling is monotone, so it cannot change which test looks riskier.

    This matters more than the ECE itself: the conformal layer thresholds the
    calibrated score, and a non-monotone map would silently reorder the
    ranking that the selection rule depends on.
    """
    probabilities, labels = make_scores(n=5000, seed=6)
    calibrated = ScoreCalibrator("platt").fit(probabilities, labels).transform(probabilities)

    order_before = np.argsort(probabilities)
    assert np.all(np.diff(calibrated[order_before]) >= -1e-9)


def test_calibrated_scores_stay_inside_the_unit_interval():
    """Downstream code treats the output as a probability, so it must be one."""
    probabilities, labels = make_scores(n=3000, seed=7)
    calibrated = ScoreCalibrator("platt").fit(probabilities, labels).transform(probabilities)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def test_transform_before_fit_is_refused():
    """An unfitted calibrator must not quietly pass scores through."""
    with pytest.raises(Exception):
        ScoreCalibrator("platt").transform(np.array([0.1, 0.5]))


def test_safe_logit_survives_the_endpoints():
    """Raw scores of exactly 0 or 1 must not become infinite.

    XGBoost emits hard 0.0 and 1.0 often enough that an unclipped logit would
    put a NaN into the calibration fit and take the whole pipeline down.
    """
    transformed = _safe_logit(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(transformed))
    assert transformed[1] == pytest.approx(0.0, abs=1e-6)


def test_unknown_calibration_method_is_rejected():
    """A typo in ``calibration.method`` must fail loudly at fit time."""
    with pytest.raises(Exception):
        ScoreCalibrator("magic").fit(*reversed(make_scores(n=500, seed=8)))
