"""
Unit tests for the 8 RTS Baselines and Comparative Benchmark Engine.
"""

import numpy as np
import pandas as pd
import pytest

from conftest.models.baselines import (
    FullSuiteSelector,
    RandomKSelector,
    ChangedFileSelector,
    DependencyGraphSelector,
    HistoricalFailureSelector,
    UncalibratedMLSelector,
    CalibratedNoAbstentionSelector,
    ConfTestSelectiveSelector,
)
from conftest.evaluation.benchmark import BaselineBenchmarkRunner, METRIC_LABELS


def _commit_frame(n_commits: int = 12, n_tests: int = 8, failures: bool = True) -> pd.DataFrame:
    """
    A deterministic multi-commit frame for the benchmark: no sampled labels.

    A test is 'affected' by a commit when its module index matches the commit's,
    which gives the changed-file and dependency baselines something real to find.
    Labels come from that structure, never from a draw.
    """
    rows = []
    for c in range(n_commits):
        for t in range(n_tests):
            affected = (t % 4) == (c % 4)
            rows.append({
                "commit_sha": f"sha_{c}",
                "test_id": f"tests/test_mod{t}.py::test_case",
                "changed_file_path": f"src/mod{c % 4}.py",
                "label_failed": 1 if (failures and affected and c % 3 == 0) else 0,
                "dep_is_direct_import": 1.0 if affected else 0.0,
                "hist_lifetime_failure_rate": 0.4 if affected else 0.02,
                "hist_avg_duration": 0.5 + t * 0.25,
                "raw_score": 0.9 if affected else 0.1,
                "calibrated_confidence": 0.93 if affected else 0.06,
                "uncertainty": 0.05 if affected else 0.03,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_candidates():
    """Fixture providing candidate test cases with diffs and features."""
    return [
        {
            "test_id": "tests/test_auth.py::test_login",
            "test_path": "tests/test_auth.py",
            "features": {"dep_is_direct_import": 1.0, "dep_shortest_path_depth": 1.0, "hist_lifetime_failure_rate": 0.3},
            "raw_score": 0.95,
            "calibrated_confidence": 0.96,
            "uncertainty": 0.04,
        },
        {
            "test_id": "tests/test_db.py::test_pool",
            "test_path": "tests/test_db.py",
            "features": {"dep_is_direct_import": 0.0, "dep_shortest_path_depth": 3.0, "hist_lifetime_failure_rate": 0.1},
            "raw_score": 0.60,
            "calibrated_confidence": 0.65,
            "uncertainty": 0.08,
        },
        {
            "test_id": "tests/test_payment.py::test_stripe",
            "test_path": "tests/test_payment.py",
            "features": {"dep_is_direct_import": 0.0, "dep_shortest_path_depth": 10.0, "hist_lifetime_failure_rate": 0.0},
            "raw_score": 0.10,
            "calibrated_confidence": 0.05,
            "uncertainty": 0.02,
        },
        {
            "test_id": "tests/test_utils.py::test_format",
            "test_path": "tests/test_utils.py",
            "features": {"dep_is_direct_import": 0.0, "dep_shortest_path_depth": 10.0, "hist_lifetime_failure_rate": 0.0},
            "raw_score": 0.05,
            "calibrated_confidence": 0.02,
            "uncertainty": 0.01,
        },
    ]


@pytest.fixture
def sample_diff():
    return [{"file_path": "src_app/auth.py", "lines_added": 20, "lines_deleted": 5}]


def test_full_suite_selector(sample_candidates, sample_diff):
    selector = FullSuiteSelector()
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.25)
    assert res.selected_count == 4
    assert res.test_reduction_ratio == 0.0
    assert res.mode == "SAFE_FULL_SUITE"


def test_random_k_selector(sample_candidates, sample_diff):
    selector = RandomKSelector(random_seed=42)
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.50)
    assert res.selected_count == 2
    assert res.test_reduction_ratio == 0.50


def test_changed_file_selector(sample_candidates, sample_diff):
    selector = ChangedFileSelector()
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.25)
    assert "tests/test_auth.py::test_login" in res.selected_tests


def test_dependency_graph_selector(sample_candidates, sample_diff):
    selector = DependencyGraphSelector()
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.25)
    assert res.selected_tests[0] == "tests/test_auth.py::test_login"


def test_conftest_selective_abstention_on_high_uncertainty(sample_candidates, sample_diff):
    # Inject high uncertainty
    sample_candidates[0]["uncertainty"] = 0.25
    selector = ConfTestSelectiveSelector(abstention_threshold=0.15)
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.25)

    assert res.abstained is True
    assert res.mode == "SAFE_FULL_SUITE"
    assert res.selected_count == 4  # Full suite fallback


def test_conftest_selective_fast_mode_on_low_uncertainty(sample_candidates, sample_diff):
    # Low uncertainty
    selector = ConfTestSelectiveSelector(abstention_threshold=0.15)
    res = selector.select(sample_candidates, sample_diff, budget_ratio=0.25)

    assert res.abstained is False
    assert res.mode == "FAST_SELECTED"
    assert res.selected_count == 1
    assert res.selected_tests[0] == "tests/test_auth.py::test_login"


def test_baseline_benchmark_runner():
    """Verify BaselineBenchmarkRunner evaluates all 8 baselines and produces summary metrics."""
    df = pd.DataFrame([
        {
            "commit_sha": "sha_1",
            "test_id": "tests/test_auth.py::test_login",
            # Changed-file provenance is mandatory: the benchmark refuses to
            # run without it rather than defaulting to a placeholder path.
            "changed_file_path": "src/auth.py",
            "label_failed": 1,
            "dep_is_direct_import": 1.0,
            "hist_lifetime_failure_rate": 0.5,
            "raw_score": 0.90,
            "calibrated_confidence": 0.92,
            "uncertainty": 0.05,
        },
        {
            "commit_sha": "sha_1",
            "test_id": "tests/test_other.py::test_other",
            "changed_file_path": "src/auth.py",
            "label_failed": 0,
            "dep_is_direct_import": 0.0,
            "hist_lifetime_failure_rate": 0.0,
            "raw_score": 0.10,
            "calibrated_confidence": 0.05,
            "uncertainty": 0.02,
        },
    ])

    runner = BaselineBenchmarkRunner(budget_ratio=0.50)
    summary_df = runner.evaluate_dataset(df)

    assert len(summary_df) == 8

    # The Changed-File baseline must now actually select the matching test.
    # It previously reported 0% recall on every commit because the harness fed
    # it a hardcoded placeholder diff instead of real changed-file data.
    changed_file_row = summary_df[
        summary_df["Strategy / Baseline"].str.contains("Changed-File", na=False)
    ]
    assert not changed_file_row.empty, (
        f"missing baseline in {list(summary_df['Strategy / Baseline'])}"
    )
    recall = float(changed_file_row.iloc[0]["Failure Recall (FR %)"].rstrip("%"))
    assert recall > 0.0, (
        "Changed-File baseline selected nothing despite src/auth.py matching "
        "tests/test_auth.py -- the placeholder-diff bug has regressed"
    )

    assert "Strategy / Baseline" in summary_df.columns
    assert "Test Reduction (TRR %)" in summary_df.columns
    assert "Failure Recall (FR %)" in summary_df.columns


def test_benchmark_rejects_dataset_without_changed_file_provenance():
    """
    A dataset with no changed-file column must fail loudly.

    Silently defaulting to a placeholder path is what made the Changed-File and
    dependency baselines meaningless in the original evaluation.
    """
    df = pd.DataFrame([
        {
            "commit_sha": "sha_1",
            "test_id": "tests/test_auth.py::test_login",
            "label_failed": 1,
            "dep_is_direct_import": 1.0,
            "hist_lifetime_failure_rate": 0.5,
            "raw_score": 0.9,
            "calibrated_confidence": 0.9,
            "uncertainty": 0.05,
        },
    ])

    runner = BaselineBenchmarkRunner(budget_ratio=0.50)
    with pytest.raises(ValueError, match="changed_file_path"):
        runner.evaluate_dataset(df)


def test_the_resampling_unit_is_the_commit_not_the_test_row():
    """
    The interval must be built from commits, not from the 96 rows they contain.

    Rows inside a commit share a diff and an injected fault, so they are not
    independent draws. Resampling them would report an interval several times
    narrower than the evidence supports -- the single easiest way to overstate a
    result at this sample size.
    """
    df = _commit_frame(n_commits=12, n_tests=8)
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    result = runner.evaluate_dataset_with_intervals(df, num_bootstraps=200)

    assert result["resampling_unit"] == "commit"
    assert result["num_commits"] == 12
    assert len(df) == 96, "fixture sanity: there are far more rows than commits"
    for block in result["metrics"].values():
        for interval in block["intervals"].values():
            assert interval["num_units"] == 12


def test_every_reported_number_carries_an_interval():
    """A metric without an interval is not a result: all 8 strategies, all 5 metrics."""
    df = _commit_frame()
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    result = runner.evaluate_dataset_with_intervals(df, num_bootstraps=200)

    assert set(result["metrics"]) == set(METRIC_LABELS)
    assert len(result["strategies"]) == 8
    for metric, block in result["metrics"].items():
        assert set(block["intervals"]) == set(result["strategies"]), metric
        for strategy, interval in block["intervals"].items():
            assert interval["ci_lower"] <= interval["point"] <= interval["ci_upper"], (
                f"{metric} / {strategy} point estimate falls outside its own interval"
            )

    table = runner.interval_table(result)
    assert len(table) == 8
    for label in METRIC_LABELS.values():
        assert label in table.columns
        assert table[label].str.contains(r"\[").all(), f"{label} has a cell with no bounds"


def test_the_interval_describes_the_number_in_the_summary_table():
    """
    The bootstrap must describe the same point estimate the report prints.

    They come from separate sweeps, so this also pins the reproducibility that
    Random-k's rewind provides: without it the two tables disagree on one row.
    """
    df = _commit_frame()
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    summary = runner.evaluate_dataset(df).set_index("Strategy / Baseline")
    result = runner.evaluate_dataset_with_intervals(df, num_bootstraps=100)

    for metric, label in METRIC_LABELS.items():
        for strategy, interval in result["metrics"][metric]["intervals"].items():
            printed = summary.loc[strategy, label]
            if printed.startswith("n/a"):
                assert np.isnan(interval["point"])
            else:
                assert interval["point"] == pytest.approx(float(printed.rstrip("%")), abs=0.05)


def test_repeated_sweeps_report_the_same_numbers():
    """
    Evaluating one dataset twice in one process must not move the result.

    Random-k carried its RNG across sweeps, so the second evaluation selected a
    different sample and reported a different row.
    """
    df = _commit_frame()
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    first = runner.evaluate_dataset(df)
    second = runner.evaluate_dataset(df)

    assert first.equals(second)


def test_differences_are_paired_against_the_named_reference():
    """
    The headline claim is a difference, so the difference is what carries an interval.

    Each difference is the strategy minus the reference on a shared resample, and
    excludes_zero answers the only question asked of it.
    """
    df = _commit_frame()
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)
    reference = "8. ConfTest (Calibrated + Selective Abstention)"

    result = runner.evaluate_dataset_with_intervals(
        df, num_bootstraps=200, reference=reference
    )

    recall = result["metrics"]["fr"]
    assert reference not in recall["differences"]
    assert len(recall["differences"]) == 7
    for strategy, difference in recall["differences"].items():
        assert difference["reference"] == reference
        assert isinstance(difference["excludes_zero"], bool)
        expected = (
            recall["intervals"][strategy]["point"] - recall["intervals"][reference]["point"]
        )
        assert difference["point"] == pytest.approx(expected)

    table = runner.interval_table(result)
    assert (table.set_index("Strategy / Baseline").loc[reference, "FR vs reference"]) == "reference"


def test_a_missing_reference_strategy_fails_loudly():
    """A misspelled reference must not silently degrade to unpaired intervals."""
    df = _commit_frame()
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    with pytest.raises(ValueError, match="reference strategy"):
        runner.evaluate_dataset_with_intervals(df, num_bootstraps=10, reference="ConfTest")


def test_recall_over_a_dataset_with_no_failures_is_not_reported_as_zero():
    """
    "No failing test was available" is not the same as "no failing test was found".

    The old max(1, denominator) guard printed 0.0% recall for a dataset that
    offered nothing to recall, which reads as a total miss rather than as an
    undefined measurement.
    """
    df = _commit_frame(failures=False)
    runner = BaselineBenchmarkRunner(budget_ratio=0.35)

    summary = runner.evaluate_dataset(df)

    assert (summary["Failure Recall (FR %)"] == "n/a").all()
    assert (summary["Missed-Failure (MFR %)"] == "n/a").all()
    # Test reduction is still perfectly well defined: tests were available.
    assert not summary["Test Reduction (TRR %)"].str.startswith("n/a").any()
