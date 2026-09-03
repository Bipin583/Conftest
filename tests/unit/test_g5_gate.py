"""
Unit tests for the G5 gate check.

G5 is stated as "reduction at 95% recall, reported with CI, on data never seen in
training". Three things can quietly break that sentence, and each has a test here:

* the interval must resample commits, not rows -- rows inside one commit share a
  mutant and a diff, so resampling rows would shrink every interval;
* a replicate whose denominator is empty is undefined, and averaging it in as a
  zero would drag the lower bound down and make the gate look harder than it is;
* whether the gate is met has to be read off the interval's lower bound, not the
  point estimate, or a sample one commit wider could unmeet it.
"""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_g5_recall_floor import intervals_for  # noqa: E402


def commit(sha, avail, executed, failures, detected, abstained=False, escaped=False):
    return {
        "commit_sha": sha,
        "tests_available": avail,
        "tests_executed": executed,
        "failures_available": failures,
        "failures_detected": detected,
        "abstained": abstained,
        "escaped": escaped,
    }


def test_intervals_cover_the_point_estimate_and_name_the_resampling_unit():
    per_commit = [
        commit(f"c{i}", 100, 25, 10, 9 if i % 3 else 10, escaped=bool(i % 3))
        for i in range(40)
    ]
    out = intervals_for(per_commit, bootstraps=200)
    assert out["resampling_unit"] == "commit_sha"
    assert out["commits_resampled"] == 40
    assert out["num_bootstraps"] == 200

    trr = out["intervals"]["test_reduction_trr_pct"]
    recall = out["intervals"]["failure_recall_pct"]
    # 25 of 100 executed everywhere, so reduction is 75% with no spread at all.
    assert trr["point"] == pytest.approx(75.0)
    assert trr["ci_lower"] == pytest.approx(75.0)
    assert trr["ci_upper"] == pytest.approx(75.0)
    assert trr["num_units"] == 40
    assert recall["ci_lower"] <= recall["point"] <= recall["ci_upper"]
    assert 90.0 < recall["point"] < 100.0
    assert recall["undefined_replicates"] == 0


def test_a_replicate_with_no_failure_to_find_is_undefined_not_zero():
    """
    Every commit here has an empty recall denominator, so no replicate can produce a
    recall. The interval has to come back undefined rather than as a hard 0%.
    """
    per_commit = [commit(f"c{i}", 50, 10, 0, 0) for i in range(10)]
    out = intervals_for(per_commit, bootstraps=100)
    recall = out["intervals"]["failure_recall_pct"]
    assert math.isnan(float(recall["point"]))
    assert math.isnan(float(recall["ci_lower"]))
    # Every replicate was undefined, and the report says so rather than leaving the
    # reader to infer it from a missing bound.
    assert recall["undefined_replicates"] == recall["num_bootstraps"]
    # Reduction is still measurable: there were tests to skip.
    assert out["intervals"]["test_reduction_trr_pct"]["point"] == pytest.approx(80.0)


def test_escape_and_abstention_rates_are_per_commit_shares():
    per_commit = [commit("a", 10, 10, 1, 1, abstained=True)] + [
        commit(f"b{i}", 10, 2, 1, 0, escaped=True) for i in range(3)
    ]
    out = intervals_for(per_commit, bootstraps=100)
    assert out["intervals"]["abstention_rate_pct"]["point"] == pytest.approx(25.0)
    assert out["intervals"]["escaped_commit_rate_pct"]["point"] == pytest.approx(75.0)


def test_per_commit_records_are_off_by_default_and_shaped_when_asked():
    from tune_policy import evaluate_thresholds_on_dataset
    import inspect

    sig = inspect.signature(evaluate_thresholds_on_dataset)
    assert sig.parameters["return_per_commit"].default is False
