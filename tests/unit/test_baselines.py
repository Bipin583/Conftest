"""
Unit tests for the 8 RTS Baselines and Comparative Benchmark Engine.
"""

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
from conftest.evaluation.benchmark import BaselineBenchmarkRunner


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
