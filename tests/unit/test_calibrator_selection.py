"""
Tests for calibrator selection.

Every case here is the shape of a decision the original one-liner got wrong:

    best_cal = iso_cal if iso_ece <= temp_ece else temp_cal

It chose on ECE alone, it chose on the test split, and it could not decline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conftest.models.calibrator_selection import (  # noqa: E402
    BRIER_TOLERANCE,
    MCE_TOLERANCE,
    CalibrationScore,
    score_calibrators,
    select_calibrator,
    split_clusters_for_selection,
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


# --------------------------------------------------------------------------
# Choosing on measured noise rather than on fixed tolerances
# --------------------------------------------------------------------------


def _difference(point: float, lower: float, upper: float) -> dict:
    """A paired-difference block of the shape `intervals_from_replicates` returns."""
    return {
        "point": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": lower > 0.0 or upper < 0.0,
    }


def _candidate(method: str, ece: float, mce: float, brier: float, diffs: dict):
    """A candidate carrying paired intervals, so the bootstrap path is taken."""
    return CalibrationScore(
        method=method,
        ece=ece,
        mce=mce,
        brier=brier,
        ece_vs_baseline=diffs["ece"],
        mce_vs_baseline=diffs["mce"],
        brier_vs_baseline=diffs["brier"],
    )


# Measured on this project's own test split: temperature scaling came out 25.5%
# better on ECE than leaving the model alone, and its paired difference over
# resampled mutants was -0.0066 [-0.0112, +0.0110]. The report would have printed
# "ECE reduced by 25%" from a difference the evidence cannot distinguish from zero.
NOISY_GAIN = dict(
    ece=_difference(-0.00657, -0.01118, 0.01099),
    mce=_difference(0.00100, -0.00800, 0.01000),
    brier=_difference(0.00006, -0.00059, 0.00089),
)
REAL_GAIN = dict(
    ece=_difference(-0.02000, -0.03000, -0.01000),
    mce=_difference(-0.01000, -0.02000, -0.00100),
    brier=_difference(-0.00300, -0.00500, -0.00100),
)

# The same run's uncalibrated row, as the baseline every difference is taken from.
UNCAL_MEASURED = CalibrationScore("uncalibrated", ece=0.0258, mce=0.2222, brier=0.0449)


def test_a_lower_ece_whose_interval_spans_zero_is_not_a_gain():
    """
    The decisive case: the two paths disagree, and the interval path is right.

    This candidate's ECE point estimate beats the uncalibrated model and both
    guard metrics sit well inside their tolerances, so the fixed-tolerance rule
    accepts it and the report claims a 25% ECE reduction. The paired difference
    over resampled mutants spans zero, so the same measurement supports no claim
    at all. Nothing about the candidate changes between the two calls here except
    whether the intervals are available to be read.
    """
    tolerance_path = [
        UNCAL_MEASURED,
        CalibrationScore("temperature_scaling", ece=0.0192, mce=0.2300, brier=0.0449),
    ]
    assert select_calibrator(tolerance_path).method == "temperature_scaling"
    assert select_calibrator(tolerance_path).basis == "point"

    interval_path = [
        UNCAL_MEASURED,
        _candidate("temperature_scaling", 0.0192, 0.2300, 0.0449, NOISY_GAIN),
    ]
    outcome = select_calibrator(interval_path)
    assert outcome.method == "uncalibrated"
    assert outcome.basis == "bootstrap"
    assert "does not exclude zero" in outcome.disqualified["temperature_scaling"]


def test_a_gain_clear_of_zero_still_wins():
    """The interval path is a filter on noise, not a refusal to ever calibrate."""
    outcome = select_calibrator([
        UNCAL_MEASURED,
        _candidate("isotonic", 0.0058, 0.2100, 0.0419, REAL_GAIN),
    ])
    assert outcome.method == "isotonic"
    assert outcome.basis == "bootstrap"
    assert outcome.calibrated is True


def test_a_measurably_worse_worst_case_is_named_as_such():
    """
    MCE is checked before ECE, and the reason says which comparison sank it.

    The abstention policy reads confidence at a threshold, so the worst bin is the
    one that breaks it -- a candidate that improves the average while measurably
    degrading the worst case is not a candidate.
    """
    worse_mce = dict(REAL_GAIN, mce=_difference(0.20910, 0.05840, 0.30000))
    outcome = select_calibrator([
        UNCAL_MEASURED,
        _candidate("isotonic", 0.0058, 0.3129, 0.0419, worse_mce),
    ])
    assert outcome.method == "uncalibrated"
    assert "MCE 0.3129" in outcome.disqualified["isotonic"]
    assert "excludes zero" in outcome.disqualified["isotonic"]


def test_one_candidate_without_intervals_puts_the_whole_decision_on_tolerances():
    """
    A mixed comparison must not be judged by two different standards.

    Rejecting one candidate on a measured interval while accepting another on a
    fixed tolerance would compare them on incomparable evidence, and the stricter
    standard would fall on whichever candidate happened to carry intervals.
    """
    outcome = select_calibrator([
        UNCAL_MEASURED,
        _candidate("temperature_scaling", 0.0192, 0.2300, 0.0449, NOISY_GAIN),
        CalibrationScore("isotonic", ece=0.0251, mce=0.2300, brier=0.0368),
    ])
    assert outcome.basis == "point"
    # On tolerances the noisy ECE gain is a gain, and the better of the two wins.
    assert outcome.method == "temperature_scaling"


# --------------------------------------------------------------------------
# Scoring: the resampling unit is the mutant
# --------------------------------------------------------------------------


def _clustered_frame(n_mutants: int = 8, n_tests: int = 10):
    """
    Labels concentrated inside mutants, which is how injected faults behave.

    The first quarter of mutants fail every test; the rest fail none. Resampling
    mutants therefore swings the failure rate from 0 to 1, while resampling rows
    holds it near its pooled value -- the difference this fixture exists to expose.
    """
    y, clusters = [], []
    for m in range(n_mutants):
        for _ in range(n_tests):
            y.append(1 if m < n_mutants // 4 else 0)
            clusters.append(f"mutant_{m}")
    y = np.asarray(y)
    return y, clusters, np.where(y == 1, 0.75, 0.20)


def test_resampling_rows_instead_of_mutants_understates_every_interval():
    """
    The single easiest way to overstate a calibration result at this sample size.

    Both calls measure the same rows, the same labels and the same probabilities,
    and both report the same point estimate. The only difference is what a
    replicate draws: whole mutants, or rows. Rows are not independent draws -- all
    the tests of one mutant see one injected fault -- and treating them as if they
    were reports an interval several times narrower than the evidence supports.
    """
    y, clusters, probs = _clustered_frame()
    methods = {"uncalibrated": probs, "candidate": probs * 0.9}

    def ece_difference(cluster_ids):
        scores = score_calibrators(
            y, methods, cluster_ids=cluster_ids, num_bootstraps=400, random_seed=5
        )
        return {s.method: s for s in scores}["candidate"].ece_vs_baseline

    by_mutant = ece_difference(clusters)
    by_row = ece_difference([f"row_{i}" for i in range(len(y))])

    assert by_mutant["num_units"] == 8
    assert by_row["num_units"] == len(y)
    assert by_mutant["point"] == pytest.approx(by_row["point"]), (
        "the point estimate is a property of the data, not of the resampling unit"
    )
    width = lambda d: d["ci_upper"] - d["ci_lower"]  # noqa: E731
    assert width(by_mutant) > 2.0 * width(by_row)


def test_the_baseline_comes_first_and_carries_no_difference_from_itself():
    """Callers read the list positionally to report the uncalibrated row first."""
    y, clusters, probs = _clustered_frame()
    scores = score_calibrators(
        y,
        {"isotonic": probs * 0.9, "uncalibrated": probs},
        cluster_ids=clusters,
        num_bootstraps=100,
    )

    assert [s.method for s in scores] == ["uncalibrated", "isotonic"]
    assert scores[0].has_intervals is False
    assert scores[1].has_intervals is True
    assert scores[1].ece_vs_baseline["reference"] == "uncalibrated"


def test_without_cluster_labels_the_scores_carry_no_intervals():
    """
    No clusters is a degradation to be visible, not one to be papered over.

    `has_intervals` False is what routes `select_calibrator` back to its fixed
    tolerances; inventing an interval by resampling rows would be worse than
    having none, because it would look like evidence.
    """
    y, _, probs = _clustered_frame()
    scores = score_calibrators(y, {"uncalibrated": probs, "isotonic": probs * 0.9})

    assert all(s.has_intervals is False for s in scores)
    assert all(s.ece == s.ece for s in scores), "point estimates are still measured"


def test_a_single_class_resample_still_yields_a_brier_score():
    """
    Every row can share a label, and that is a normal resample here.

    The harvest targets a failure rate in the 1-15% band, so a resample of mutants
    that drew no failing one is ordinary. sklearn's `brier_score_loss` infers the
    positive label from the labels present and raises on that input, which is why
    the score is written out by hand.
    """
    y = np.zeros(40, dtype=int)
    probs = np.full(40, 0.1)
    clusters = [f"mutant_{i // 10}" for i in range(40)]

    scores = score_calibrators(
        y, {"uncalibrated": probs, "isotonic": probs / 2}, cluster_ids=clusters,
        num_bootstraps=50,
    )
    by_method = {s.method: s for s in scores}
    assert by_method["uncalibrated"].brier == pytest.approx(0.01)
    assert by_method["isotonic"].brier == pytest.approx(0.0025)


def test_scoring_needs_the_baseline_and_matching_lengths():
    """Every difference is taken against the baseline, so it has to be present."""
    y, clusters, probs = _clustered_frame()

    with pytest.raises(ValueError, match="uncalibrated"):
        score_calibrators(y, {"isotonic": probs}, cluster_ids=clusters)

    with pytest.raises(ValueError, match="probabilities for"):
        score_calibrators(y, {"uncalibrated": probs, "isotonic": probs[:-1]})

    with pytest.raises(ValueError, match="cluster_ids has"):
        score_calibrators(y, {"uncalibrated": probs}, cluster_ids=clusters[:-1])


# --------------------------------------------------------------------------
# Splitting the validation half along cluster boundaries
# --------------------------------------------------------------------------


def test_no_mutant_lands_on_both_sides_of_the_split():
    """
    The leak the positional cut leaves open, stated as the properties that close it.

    Isotonic regression is flexible enough to learn a mutant's failure pattern from
    the rows it was fitted on; if that mutant's remaining rows are then used to
    judge it, it is rewarded for the memorisation. Comparing candidates on such a
    split is how the more flexible method wins a comparison it should lose.
    """
    clusters = np.asarray([f"mutant_{i // 7}" for i in range(70)])
    fit_mask = split_clusters_for_selection(clusters)

    fitted = set(clusters[fit_mask])
    held_out = set(clusters[~fit_mask])
    assert fitted and held_out, "neither half may be empty"
    assert not fitted & held_out
    assert fitted | held_out == set(clusters), "every mutant is used somewhere"


def test_the_positional_cut_is_what_splits_a_mutant_in_two():
    """
    The contrast case: the same shape of data through the old cut, which does leak.

    Kept as a test rather than a comment because `split_for_selection` is still the
    fallback for datasets carrying no cluster column, and this records exactly what
    is given up on that path.
    """
    clusters = np.asarray([f"mutant_{i // 7}" for i in range(74)])
    cut = split_for_selection(len(clusters))

    assert set(clusters[:cut]) & set(clusters[cut:]), (
        "a row-index cut lands inside a mutant unless the sizes happen to divide"
    )
    assert not (
        set(clusters[split_clusters_for_selection(clusters)])
        & set(clusters[~split_clusters_for_selection(clusters)])
    )


def test_the_halves_are_balanced_by_rows_not_by_mutant_count():
    """
    Mutants differ widely in how many tests they carry, so counting them misleads.

    Here one mutant holds two thirds of the rows. An even split by mutant count
    would leave one half with 50 rows and the other with 10 -- and the ECE measured
    on 10 rows would decide the comparison.
    """
    clusters = ["big"] * 40 + ["mid"] * 12 + ["small_a"] * 4 + ["small_b"] * 4
    fit_mask = split_clusters_for_selection(np.asarray(clusters))

    fit_rows, held_rows = int(fit_mask.sum()), int((~fit_mask).sum())
    assert fit_rows + held_rows == 60
    assert fit_rows == 40 and held_rows == 20, (
        "the 40-row mutant alone is the closest reachable split of 60 rows"
    )


def test_a_split_of_one_mutant_is_refused():
    """
    With one cluster there is no honest split, so this fails rather than leaking.

    Silently falling back to a positional cut would reintroduce the leak at exactly
    the sample size where it does the most damage.
    """
    with pytest.raises(ValueError, match="at least 2 clusters"):
        split_clusters_for_selection(np.asarray(["only_mutant"] * 20))

    with pytest.raises(ValueError, match="holdout_fraction"):
        split_clusters_for_selection(np.asarray(["a", "a", "b", "b"]), 1.0)
