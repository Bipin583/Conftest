"""
Tests for calibrator selection.

Every case here is the shape of a decision the original one-liner got wrong:

    best_cal = iso_cal if iso_ece <= temp_ece else temp_cal

It chose on ECE alone, it chose on the test split, and it could not decline.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conftest.models.calibrator_selection import (  # noqa: E402
    BRIER_TOLERANCE,
    MCE_TOLERANCE,
    CalibrationScore,
    select_calibrator,
    split_for_selection,
)


UNCAL = CalibrationScore("uncalibrated", ece=0.2222, mce=0.2222, brier=0.1800)


# --------------------------------------------------------------------------
# The measured regression
# --------------------------------------------------------------------------

def test_the_run_that_motivated_this_module():
    """
    Real numbers from this project's own calibration run. Temperature scaling had
    the best ECE of the three and an MCE of 0.8943 -- four times worse than
    leaving the model alone. The abstention policy reads confidence at a specific
    threshold, so a bin that is off by 0.89 is precisely what breaks it.
    """
    scores = [
        UNCAL,
        CalibrationScore("isotonic", ece=0.0400, mce=0.1900, brier=0.1600),
        CalibrationScore("temperature_scaling", ece=0.0192, mce=0.8943, brier=0.1750),
    ]
    outcome = select_calibrator(scores)
    assert outcome.method == "isotonic"
    assert "temperature_scaling" in outcome.disqualified
    assert "MCE" in outcome.disqualified["temperature_scaling"]


def test_lowest_ece_alone_does_not_win():
    """Direct statement of the old bug: best ECE, disqualifying MCE."""
    scores = [
        UNCAL,
        CalibrationScore("temperature_scaling", ece=0.0001, mce=0.9999, brier=0.1799),
    ]
    assert select_calibrator(scores).method == "uncalibrated"


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_a_worse_proper_score_disqualifies():
    scores = [
        UNCAL,
        CalibrationScore("isotonic", ece=0.01, mce=0.20, brier=UNCAL.brier + 0.05),
    ]
    outcome = select_calibrator(scores)
    assert outcome.method == "uncalibrated"
    assert "Brier" in outcome.disqualified["isotonic"]


def test_guards_allow_movement_inside_tolerance():
    """
    Bin boundaries shift between candidates, so an exactly-equal method must not
    be rejected on noise. Slightly worse MCE and Brier, clearly better ECE: keep.
    """
    scores = [
        UNCAL,
        CalibrationScore(
            "isotonic",
            ece=0.05,
            mce=UNCAL.mce + MCE_TOLERANCE / 2,
            brier=UNCAL.brier + BRIER_TOLERANCE / 2,
        ),
    ]
    assert select_calibrator(scores).method == "isotonic"


def test_no_ece_gain_disqualifies():
    """A method that does not improve calibration has no reason to be installed."""
    scores = [
        UNCAL,
        CalibrationScore("temperature_scaling", ece=UNCAL.ece, mce=0.10, brier=0.15),
    ]
    outcome = select_calibrator(scores)
    assert outcome.method == "uncalibrated"
    assert "no calibration gain" in outcome.disqualified["temperature_scaling"]


# --------------------------------------------------------------------------
# Declining is a real outcome
# --------------------------------------------------------------------------

def test_uncalibrated_can_win_and_says_why():
    scores = [
        UNCAL,
        CalibrationScore("isotonic", ece=0.30, mce=0.40, brier=0.20),
        CalibrationScore("temperature_scaling", ece=0.25, mce=0.99, brier=0.19),
    ]
    outcome = select_calibrator(scores)
    assert outcome.method == "uncalibrated"
    assert outcome.calibrated is False
    assert "uncalibrated" in outcome.reason
    assert len(outcome.disqualified) == 2


def test_the_baseline_must_be_supplied():
    """
    Without the uncalibrated model there is nothing to guard against, and no way
    to decline. Silently picking the least-bad candidate is how the old code
    installed a calibrator with MCE 0.8943.
    """
    with pytest.raises(ValueError, match="uncalibrated"):
        select_calibrator([CalibrationScore("isotonic", 0.01, 0.02, 0.03)])


def test_a_winner_reports_calibrated_true():
    scores = [UNCAL, CalibrationScore("isotonic", ece=0.02, mce=0.10, brier=0.15)]
    assert select_calibrator(scores).calibrated is True


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_ties_break_towards_the_safer_method():
    """Equal ECE: the lower worst-case bin wins."""
    scores = [
        UNCAL,
        CalibrationScore("isotonic", ece=0.05, mce=0.18, brier=0.16),
        CalibrationScore("temperature_scaling", ece=0.05, mce=0.09, brier=0.16),
    ]
    assert select_calibrator(scores).method == "temperature_scaling"


def test_selection_is_order_independent():
    a = CalibrationScore("isotonic", ece=0.04, mce=0.19, brier=0.16)
    b = CalibrationScore("temperature_scaling", ece=0.0192, mce=0.8943, brier=0.175)
    assert select_calibrator([UNCAL, a, b]).method == select_calibrator([b, a, UNCAL]).method


# --------------------------------------------------------------------------
# The fit/select split
# --------------------------------------------------------------------------

def test_split_leaves_both_halves_non_empty():
    for n in (2, 3, 7, 100, 4001):
        fit = split_for_selection(n)
        assert 1 <= fit <= n - 1, n


def test_split_honours_the_fraction():
    assert split_for_selection(100, holdout_fraction=0.25) == 75
    assert split_for_selection(100, holdout_fraction=0.5) == 50


def test_split_rejects_a_degenerate_fraction():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="holdout_fraction"):
            split_for_selection(100, holdout_fraction=bad)
