"""
Unit tests for the selective policy threshold tuner.

Two properties matter here and neither is about the search itself.

The first is that an unsatisfiable objective has no answer. Until 2026-09-03 the
tuner fell back to a silent ``(0.015, 0.50)`` whenever the zero-escape constraint
admitted nothing on the grid, so a run that had measured no acceptable operating
point still wrote a policy config -- one that no evaluation had ever scored -- and
every downstream report described a system tuned to nothing.

The second is that the two objectives are genuinely different questions. The
shipped constraint forbids any escaped failure; gate G5 asks for the largest
reduction that holds recall at a floor. On this dataset the second admits points
the first rejects, so a tuner that only knows the first cannot report G5 at all.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tune_policy import parse_grid, select_frontier_point  # noqa: E402
from conftest.features.pipeline import FEATURE_NAMES  # noqa: E402


class _StubEnsemble:
    """Confident and unanimous, so the policy takes its fast path."""

    def predict_with_uncertainty(self, X):
        n = len(X)
        return {
            "mean_prob": np.full(n, 0.9),
            "epistemic_std": np.zeros(n),
        }


class _StubCalibrator:
    def calibrate(self, probs):
        return np.asarray(probs, dtype=float)


def record(tau_a, tau_c, trr, recall, escapes, abstention=50.0):
    """One grid point, shaped like evaluate_thresholds_on_dataset returns it."""
    return {
        "tau_abstain": tau_a,
        "tau_conf": tau_c,
        "test_reduction_trr_pct": trr,
        "failure_recall_pct": recall,
        "abstention_rate_pct": abstention,
        "escaped_commits": escapes,
        "missed_failures": 0 if escapes == 0 else escapes,
    }


GRID = [
    record(0.005, 0.10, 0.0, 100.0, 0),
    record(0.020, 0.10, 2.42, 100.0, 0),
    record(0.030, 0.10, 5.18, 99.98, 1),
    record(0.044, 0.10, 32.85, 96.02, 11),
    record(0.050, 0.10, 45.29, 80.65, 23),
]


def test_zero_escape_takes_the_largest_reduction_that_lets_nothing_escape():
    result = select_frontier_point(GRID, "zero_escape", recall_floor=95.0)
    assert result["selected"]["tau_abstain"] == 0.020
    assert result["selected"]["escaped_commits"] == 0
    assert result["eligible_points"] == 2
    assert result["constraint"] == "escaped_commits == 0"


def test_recall_floor_admits_points_the_zero_escape_constraint_rejects():
    """The whole point of the second objective: 96% recall is not 100% recall."""
    result = select_frontier_point(GRID, "recall_floor", recall_floor=95.0)
    assert result["selected"]["tau_abstain"] == 0.044
    assert result["selected"]["test_reduction_trr_pct"] == 32.85
    # It buys an order of magnitude more reduction than the zero-escape point, at
    # the cost of eleven commits that lose a failing test.
    assert result["selected"]["escaped_commits"] == 11


def test_recall_floor_excludes_points_below_the_floor():
    result = select_frontier_point(GRID, "recall_floor", recall_floor=99.0)
    assert result["selected"]["tau_abstain"] == 0.030
    assert result["eligible_points"] == 3


def test_unsatisfiable_objective_raises_instead_of_inventing_a_default():
    """
    A grid where every point escapes has no zero-escape operating point. The old
    behaviour was to write (0.015, 0.50) anyway.
    """
    all_escape = [
        record(0.044, 0.10, 32.85, 96.02, 11),
        record(0.050, 0.10, 45.29, 80.65, 23),
    ]
    with pytest.raises(ValueError, match="No threshold pair"):
        select_frontier_point(all_escape, "zero_escape", recall_floor=95.0)
    for forbidden in ("0.015", "0.50"):
        try:
            select_frontier_point(all_escape, "zero_escape", recall_floor=95.0)
        except ValueError as exc:
            assert forbidden not in str(exc)


def test_unreachable_recall_floor_raises():
    with pytest.raises(ValueError, match="failure_recall_pct >= 100.5"):
        select_frontier_point(GRID, "recall_floor", recall_floor=100.5)


def test_nan_metrics_are_not_eligible():
    """
    A pair whose recall could not be computed is an absent measurement. Ranking on
    it, or reading it as a zero, would let an unmeasured pair win the search.
    """
    grid = [
        record(0.010, 0.10, float("nan"), float("nan"), 0),
        record(0.020, 0.10, 2.42, 100.0, 0),
    ]
    result = select_frontier_point(grid, "zero_escape", recall_floor=95.0)
    assert result["selected"]["tau_abstain"] == 0.020
    assert result["eligible_points"] == 1

    only_nan = [record(0.010, 0.10, float("nan"), float("nan"), 0)]
    with pytest.raises(ValueError, match="No threshold pair"):
        select_frontier_point(only_nan, "zero_escape", recall_floor=95.0)


def test_ties_break_deterministically_towards_the_lower_thresholds():
    tied = [
        record(0.030, 0.30, 10.0, 100.0, 0),
        record(0.020, 0.10, 10.0, 100.0, 0),
        record(0.020, 0.50, 10.0, 100.0, 0),
    ]
    first = select_frontier_point(tied, "zero_escape", recall_floor=95.0)["selected"]
    shuffled = select_frontier_point(list(reversed(tied)), "zero_escape", 95.0)["selected"]
    assert (first["tau_abstain"], first["tau_conf"]) == (0.020, 0.10)
    assert first == shuffled


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="Unknown objective"):
        select_frontier_point(GRID, "maximise_trr", recall_floor=95.0)


def test_parse_grid_reads_and_sorts_a_comma_separated_sweep():
    assert parse_grid("0.05,0.01,0.03", "--tau-abstain-grid") == [0.01, 0.03, 0.05]
    assert parse_grid(" 0.1 , 0.2 ", "--tau-conf-grid") == [0.1, 0.2]


def test_parse_grid_rejects_an_empty_or_unparseable_sweep():
    with pytest.raises(ValueError, match="is empty"):
        parse_grid(" , ", "--tau-abstain-grid")
    with pytest.raises(ValueError, match="comma-separated numbers"):
        parse_grid("0.01,high", "--tau-abstain-grid")


def test_evaluate_reports_nan_not_zero_when_a_split_has_no_failure():
    """
    The denominator guard used to be max(1, n), which turned "nothing to find" into a
    hard 0.0% recall. A commit whose mutant no test detects has no recall to report,
    and calling that zero understates every threshold pair on the grid.
    """
    from tune_policy import evaluate_thresholds_on_dataset

    df = pd.DataFrame(
        {
            "commit_sha": ["c1", "c1", "c2", "c2"],
            "test_id": ["t1", "t2", "t1", "t2"],
            "label_failed": [0, 0, 0, 0],
            "diff_num_files_changed": [1, 1, 1, 1],
            "diff_total_churn": [10, 10, 10, 10],
            **{name: [0.0, 0.0, 0.0, 0.0] for name in FEATURE_NAMES},
        }
    )
    result = evaluate_thresholds_on_dataset(
        df=df,
        ensemble=_StubEnsemble(),
        calibrator=_StubCalibrator(),
        tau_abstain=0.5,
        tau_conf=0.1,
        budget_ratio=0.25,
    )
    assert math.isnan(result["failure_recall_pct"])
    assert result["failures_available"] == 0
    # Reduction is still defined: there were tests to skip even with nothing to find.
    assert not math.isnan(result["test_reduction_trr_pct"])
    assert result["escaped_commits"] == 0
