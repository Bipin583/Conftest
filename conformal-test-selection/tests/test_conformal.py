"""Layer 3: the coverage guarantee, which is the claim the project stands on.

Most of this file checks arithmetic, but the two coverage-simulation tests are
the ones that carry weight. They do not check that the *code* computes a
documented formula; they re-run the experiment the guarantee describes -- draw
a calibration set, fit a threshold, deploy it on fresh data, repeat -- and
check that realised coverage behaves as promised. That is the difference
between testing an implementation and testing a claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models.conformal import (
    ConformalError,
    ConformalSelector,
    _achieved_pac_coverage,
    _apply_budget,
    _per_commit_metrics,
    build_selector,
    marginal_rank,
    nonconformity_scores,
    pac_rank,
    selection_metrics,
)
from tests.conftest import make_scores


# --------------------------------------------------------------------------
# Quantile levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (100, 0.05, 96),    # ceil(101 * 0.95)
        (500, 0.05, 476),   # ceil(501 * 0.95)
        (1000, 0.10, 901),  # ceil(1001 * 0.90)
        (19, 0.05, 19),     # smallest n that can express 95% at all
    ],
)
def test_marginal_rank_matches_the_closed_form(n, alpha, expected):
    """``k = ceil((n + 1)(1 - alpha))``, the standard split-conformal rank."""
    assert marginal_rank(n, alpha) == expected == math.ceil((n + 1) * (1 - alpha))


def test_marginal_rank_clips_rather_than_over_promising():
    """Too small a calibration set selects everything, never under-covers.

    With ``n = 10`` no order statistic can express 95% coverage. Returning
    ``n`` means the rule keeps every test -- useless for saving CI time, but
    honest, which is the right way to fail.
    """
    assert marginal_rank(10, 0.05) == 10


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_rank_rejects_an_alpha_outside_the_unit_interval(bad_alpha):
    """A miscoverage rate outside ``(0, 1)`` is a programming error."""
    with pytest.raises(ConformalError):
        marginal_rank(100, bad_alpha)


def test_rank_rejects_an_empty_calibration_set():
    """No calibration rows means no quantile, not a silent default."""
    with pytest.raises(ConformalError):
        marginal_rank(0, 0.05)


@pytest.mark.parametrize("n", [100, 500, 4578])
def test_pac_rank_is_at_least_the_marginal_rank(n):
    """The stronger statement costs a more conservative threshold.

    A higher rank is a larger score cut-off, which selects more tests. PAC
    coverage therefore cannot be cheaper than marginal coverage; if it ever
    were, the guarantee would be free, which would mean it was not a guarantee.
    """
    assert pac_rank(n, 0.05, 0.10) >= marginal_rank(n, 0.05)


@pytest.mark.parametrize("n", [100, 500, 4578])
def test_pac_rank_certifies_the_target_but_the_marginal_rank_does_not(n):
    """The PAC rank clears 95%; the marginal rank sits below it.

    This is the quantitative gap between "on average over calibration sets"
    and "for the calibration set you actually drew".
    """
    pac_k = pac_rank(n, 0.05, 0.10)
    assert _achieved_pac_coverage(n, pac_k, 0.10) >= 0.95
    assert _achieved_pac_coverage(n, marginal_rank(n, 0.05), 0.10) < 0.95


def test_pac_rank_returns_none_when_the_level_is_unreachable():
    """Ten calibration failures cannot certify 95% at 90% confidence."""
    assert pac_rank(10, 0.05, 0.10) is None


def test_shipped_configuration_reproduces_the_committed_rank():
    """The deployed threshold is rank 4369 of 4578 calibration failures.

    Pinning this catches a change to the rank arithmetic that would silently
    move the shipped operating point.
    """
    assert pac_rank(4578, 0.05, 0.10) == 4369


# --------------------------------------------------------------------------
# Non-conformity and selection
# --------------------------------------------------------------------------


def test_nonconformity_score_is_one_minus_the_probability():
    """Low score means the model expected the failure, i.e. it conformed."""
    probabilities = np.array([0.0, 0.25, 0.9, 1.0])
    assert np.allclose(nonconformity_scores(probabilities), [1.0, 0.75, 0.1, 0.0])


def test_select_keeps_everything_at_or_above_the_probability_floor():
    """``select`` is exactly ``p >= probability_floor``, edge included."""
    rule = ConformalSelector(
        threshold=0.7,
        probability_floor=0.3,
        guarantee="pac",
        coverage=0.95,
        confidence=0.90,
        rank=1,
        n_calibration=1,
    )
    chosen = rule.select(np.array([0.29, 0.30, 0.31, 0.99]))
    assert chosen.tolist() == [False, True, True, True]


def test_selection_is_monotone_in_the_probability(selector):
    """A riskier test is never dropped while a safer one is kept."""
    chosen = selector.select(np.linspace(0.0, 1.0, 101))
    # Once the mask turns True as p increases, it must stay True.
    assert np.all(np.diff(chosen.astype(int)) >= 0)


def test_selector_round_trips_through_a_dictionary(selector):
    """``to_dict`` is the serialisation the report and API responses quote."""
    payload = selector.to_dict()
    assert payload["guarantee"] == "pac"
    assert payload["probability_floor"] == pytest.approx(1.0 - selector.threshold)
    restored = ConformalSelector(**payload)
    assert restored.select(np.array([0.5])) == selector.select(np.array([0.5]))


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def test_build_selector_is_class_conditional_by_default(scores):
    """The quantile is taken over failing rows only (Mondrian).

    Marginal-over-all-rows would spend the error budget on the 95% of rows
    that pass, guaranteeing a quantity nobody asked about.
    """
    probabilities, labels = scores
    built = build_selector(probabilities, labels, 0.95, 0.90)
    assert built["selector"].class_conditional is True
    assert built["selector"].n_calibration == int((labels == 1).sum())


def test_build_selector_reports_both_variants(scores):
    """Both thresholds are always computed, so the price of PAC is visible."""
    probabilities, labels = scores
    built = build_selector(probabilities, labels, 0.95, 0.90, guarantee="pac")
    assert {"pac", "marginal"} <= set(built["variants"])
    assert built["variants"]["pac"]["rank"] >= built["variants"]["marginal"]["rank"]


def test_build_selector_rejects_a_calibration_split_with_no_failures():
    """Without a failing row there is nothing to take a quantile over."""
    with pytest.raises(ConformalError):
        build_selector(np.linspace(0, 1, 50), np.zeros(50, dtype=int), 0.95, 0.90)


def test_build_selector_rejects_an_unknown_guarantee(scores):
    """A typo in the config must not silently fall back to a weaker rule."""
    probabilities, labels = scores
    with pytest.raises(ConformalError):
        build_selector(probabilities, labels, 0.95, 0.90, guarantee="bayesian")


def test_build_selector_rejects_misaligned_inputs():
    """Probabilities and labels describing different rows is a caller bug."""
    with pytest.raises(ConformalError):
        build_selector(np.array([0.1, 0.2, 0.3]), np.array([0, 1]), 0.95, 0.90)


# --------------------------------------------------------------------------
# The guarantee itself
# --------------------------------------------------------------------------

#: Trials for the coverage simulations. Seeds are fixed, so these tests are
#: deterministic: they either pass for everyone or fail for everyone.
N_TRIALS = 200


def _coverage_trials(guarantee: str, n_trials: int = N_TRIALS) -> np.ndarray:
    """Fit on one draw, deploy on a fresh draw, record realised coverage.

    Args:
        guarantee: ``"pac"`` or ``"marginal"``.
        n_trials: Number of independent calibration draws.

    Returns:
        Realised coverage on held-out data, one entry per trial.
    """
    realised = np.empty(n_trials)
    for trial in range(n_trials):
        cal_p, cal_y = make_scores(n=3000, seed=trial)
        rule = build_selector(cal_p, cal_y, 0.95, 0.90, guarantee=guarantee)["selector"]
        test_p, test_y = make_scores(n=6000, seed=10_000 + trial)
        realised[trial] = rule.select(test_p)[test_y == 1].mean()
    return realised


def test_pac_guarantee_holds_over_repeated_calibration_draws():
    """At least 90% of calibration draws must deliver at least 95% coverage.

    This is the project's headline claim, restated as an experiment. The PAC
    construction promises that the *proportion of calibration draws* whose
    deployed threshold covers 95% of future failures is at least
    ``1 - delta = 90%``.
    """
    realised = _coverage_trials("pac")
    success_rate = float((realised >= 0.95).mean())
    assert success_rate >= 0.90, (
        f"PAC threshold covered 95% of failures in only {success_rate:.1%} of "
        f"{N_TRIALS} calibration draws; the guarantee promises >= 90%."
    )


def test_marginal_guarantee_is_weaker_in_exactly_the_documented_way():
    """Marginal coverage is right on average and unreliable per draw.

    Mean coverage clears the target, but far fewer than 90% of individual
    draws do -- which is why ``config.yaml`` ships ``guarantee: pac``. If this
    distinction ever evaporates, the README overclaims.
    """
    marginal = _coverage_trials("marginal")
    pac = _coverage_trials("pac")

    assert marginal.mean() >= 0.95, "marginal coverage should be correct on average"
    marginal_success = float((marginal >= 0.95).mean())
    pac_success = float((pac >= 0.95).mean())
    assert marginal_success < 0.90, (
        "marginal thresholds are not supposed to reach 90% per-draw "
        f"reliability; got {marginal_success:.1%}"
    )
    assert pac_success > marginal_success


# --------------------------------------------------------------------------
# Metrics and budgets
# --------------------------------------------------------------------------


def test_selection_metrics_are_computed_the_way_the_report_reads_them():
    """Recall, selection rate and cost reduction against hand-checked counts."""
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.5, 0.05, 0.04, 0.03, 0.02])
    rule = ConformalSelector(
        threshold=0.6,
        probability_floor=0.4,
        guarantee="pac",
        coverage=0.95,
        confidence=0.90,
        rank=1,
        n_calibration=4,
    )
    # Selected: p >= 0.4 -> the 0.9, 0.8, 0.7 failures plus 0.6 and 0.5 passes.
    result = selection_metrics(labels, probabilities, rule)
    assert result["n_selected"] == 5
    assert result["failures_caught"] == 3
    assert result["failures_missed"] == 1
    assert result["empirical_coverage"] == pytest.approx(0.75)
    assert result["selection_rate"] == pytest.approx(0.5)
    assert result["cost_reduction"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.6)


def test_min_tests_tops_up_with_the_riskiest_unselected_tests():
    """A floor may only add tests, so it can never lower coverage."""
    groups = np.array(["c1"] * 6)
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    selected = np.array([True, False, False, False, False, False])
    topped = _apply_budget(groups, probabilities, selected.copy(), min_tests=3, max_tests=None)
    assert topped.sum() == 3
    # The two additions are the next-riskiest, not arbitrary rows.
    assert topped[:3].all() and not topped[3:].any()


def test_max_tests_drops_the_least_risky_selected_tests():
    """A cap removes from the bottom of the ranking, never the top."""
    groups = np.array(["c1"] * 5)
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    selected = np.ones(5, dtype=bool)
    capped = _apply_budget(groups, probabilities, selected.copy(), min_tests=0, max_tests=2)
    assert capped.tolist() == [True, True, False, False, False]


def test_per_commit_metrics_separate_full_catches_from_partial_ones():
    """A commit counts as caught only when every one of its failures is.

    Catching one of a commit's two failures still ships a broken commit, so
    the per-commit view is the one a release engineer actually cares about.
    """
    groups = np.array(["c1", "c1", "c2", "c2", "c3", "c3"])
    labels = np.array([1, 1, 1, 0, 0, 0])
    selected = np.array([True, True, False, True, False, False])
    result = _per_commit_metrics(groups, labels, selected)
    assert result["n_commits"] == 3
    assert result["n_failing_commits"] == 2       # c1 and c2
    assert result["commits_fully_caught"] == 1    # c1 only
    assert result["commit_full_catch_rate"] == pytest.approx(0.5)
