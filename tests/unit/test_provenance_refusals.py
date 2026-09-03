"""
Tests for the refusals that keep a missing input from turning into a plausible number.

Each case here is a place where the code used to return something usable -- a zero, an
empty interval, a report over rows it had drawn itself -- and now raises or returns NaN
instead. The value of a refusal is entirely in its being reachable, so it is pinned.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest.evaluation.benchmark import COMMIT_STAT_FIELDS, BaselineBenchmarkRunner
from conftest.evaluation.budget import recall_at_budget_per_commit
from conftest.evaluation.cross_repo import _split_by_group
from conftest.evaluation.headline import MissingArtifact
from conftest.evaluation.statistics import bootstrap_confidence_interval

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    """Import a script by path; scripts/ is a CLI directory, not an installed package."""
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPerCommitFrame:
    def test_length_mismatch_raises_instead_of_mislabelling_rows(self):
        """
        Zipping 3 numbers onto 2 commit ids silently attaches each to the wrong commit.

        A paired test over mislabelled pairs still produces a p-value, which is the
        failure mode worth refusing.
        """
        runner = BaselineBenchmarkRunner()
        runner.last_commit_order = ["aaa", "bbb"]
        per_commit = {
            "conftest": {f: np.zeros(3) for f in COMMIT_STAT_FIELDS},
        }
        with pytest.raises(ValueError, match="per-commit entries but"):
            runner.per_commit_frame(per_commit)

    def test_empty_denominator_gives_nan_not_zero_recall(self):
        """A commit with no failure available did not achieve 0% recall."""
        runner = BaselineBenchmarkRunner()
        runner.last_commit_order = ["aaa"]
        stats = {f: np.zeros(1) for f in COMMIT_STAT_FIELDS}
        stats["tests_available"] = np.array([10.0])
        stats["tests_selected"] = np.array([3.0])
        frame = runner.per_commit_frame({"conftest": stats})
        assert np.isnan(frame.loc[0, "failure_recall"])


class TestSplitByGroup:
    def _rows(self, n_groups: int, per_group: int = 4):
        groups = np.repeat([f"c{i}" for i in range(n_groups)], per_group)
        X = np.arange(len(groups) * 2, dtype=np.float64).reshape(len(groups), 2)
        return X, groups

    def test_single_group_cannot_be_split_two_ways(self):
        X, groups = self._rows(1)
        y = np.array([1, 0, 0, 0])
        with pytest.raises(ValueError, match="left one half empty"):
            _split_by_group(X, y, groups, fit_fraction=0.85, random_seed=42)

    def test_calibration_half_without_a_failure_is_refused(self):
        """A temperature fitted on one class is not a calibration."""
        X, groups = self._rows(4)
        y = np.zeros(len(groups), dtype=int)
        y[0] = 1  # the only failure; whichever half holds it, the other has none
        with pytest.raises(ValueError, match="no failing row|left one half empty"):
            _split_by_group(X, y, groups, fit_fraction=0.85, random_seed=42)


class TestRecallAtBudgetPerCommit:
    def test_missing_commit_ids_give_nan_and_say_why(self):
        y = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        value, note = recall_at_budget_per_commit(y, scores, None, 0.25)
        assert np.isnan(value)
        assert "no commit ids" in note

    def test_no_failure_anywhere_gives_nan_not_zero(self):
        y = np.zeros(4, dtype=int)
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        groups = np.array(["a", "a", "b", "b"])
        value, note = recall_at_budget_per_commit(y, scores, groups, 0.25)
        assert np.isnan(value)
        assert "no commit" in note

    def test_budget_is_spent_within_a_commit_not_across_the_dataset(self):
        """
        The defect this helper exists to prevent: a global top-k lets one commit
        borrow another's budget. Commit b's failure is ranked last globally but
        first inside its own commit, so a per-commit budget must find it.
        """
        y = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        scores = np.array([0.99, 0.98, 0.97, 0.96, 0.20, 0.10, 0.05, 0.01])
        groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        value, note = recall_at_budget_per_commit(y, scores, groups, 0.25)
        assert value == 1.0
        assert "1 of 1" in note


class TestBootstrapOnEmptySample:
    def test_empty_sample_returns_nan_and_n_zero(self):
        """An interval of [0, 0] reads as a perfectly precise measurement."""
        out = bootstrap_confidence_interval(np.array([]), num_bootstraps=10)
        assert out["n"] == 0
        assert np.isnan(out["mean"])
        assert np.isnan(out["ci_lower"])
        assert np.isnan(out["ci_upper"])


class TestLoadersRaiseMissingArtifact:
    """
    Every experiment script names the command that builds its input rather than
    falling back to something it can generate.
    """

    def test_load_per_commit_names_its_producer(self, tmp_path):
        module = _load_script("run_statistical_tests.py")
        with pytest.raises(MissingArtifact) as exc:
            module.load_per_commit(tmp_path / "absent.csv")
        assert module.PRODUCED_BY in str(exc.value)

    def test_load_repo_datasets_names_its_producer(self, tmp_path):
        module = _load_script("run_cross_repo_eval.py")
        with pytest.raises(MissingArtifact) as exc:
            module.load_repo_datasets(tmp_path / "absent.csv")
        assert module.PRODUCED_BY in str(exc.value)

    def test_load_stream_names_its_producer(self, tmp_path):
        module = _load_script("run_continuous_learning.py")
        with pytest.raises(MissingArtifact) as exc:
            module.load_stream(tmp_path / "absent.csv")
        assert module.PRODUCED_BY in str(exc.value)

    def test_load_stream_refuses_a_dataset_missing_the_stream_order(self, tmp_path):
        """
        Without a timestamp there is no stream, only a bag of rows in file order.

        Continual learning measured over an arbitrary order is not a measurement of
        anything, so the column has to be present rather than assumed.
        """
        module = _load_script("run_continuous_learning.py")
        path = tmp_path / "features.csv"
        pd.DataFrame(
            {"repo": ["tabulate"], "commit_sha": ["a"], "label_failed": [0]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="commit_timestamp|mutant_index|column"):
            module.load_stream(path)
