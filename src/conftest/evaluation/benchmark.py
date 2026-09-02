"""
ConfTest Baseline Benchmarking & Evaluation Suite.

Evaluates and compares the 8 RTS baselines under identical budget constraints
across temporal test split commits to measure Test Reduction (TRR), Time Reduction (ETR),
Failure Recall (FR), Missed-Failure Rate (MFR), and Abstention Rate.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

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
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class BaselineBenchmarkRunner:
    """Evaluates the 8 RTS baseline strategies across historical/synthetic commits."""

    def __init__(self, budget_ratio: float = 0.25, random_seed: int = 42):
        """
        Initialize benchmark runner.

        Args:
            budget_ratio: Target test execution budget (default: 0.25 = 25% of suite).
            random_seed: Reproducibility seed.
        """
        self.budget_ratio = budget_ratio
        self.selectors = [
            FullSuiteSelector(),
            RandomKSelector(random_seed=random_seed),
            ChangedFileSelector(),
            DependencyGraphSelector(),
            HistoricalFailureSelector(),
            UncalibratedMLSelector(),
            CalibratedNoAbstentionSelector(),
            ConfTestSelectiveSelector(),
        ]

    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all 8 baselines on a tabular dataset containing commits, test cases, and ground-truth labels.

        Args:
            df: DataFrame containing 'commit_sha', 'test_id', 'label_failed', and feature columns.

        Returns:
            Comparison summary DataFrame with TRR, ETR, FR, MFR, and Abstention Rate.
        """
        logger.info(f"Running 8-baseline benchmark across {df['commit_sha'].nunique()} commits (Budget: {self.budget_ratio*100:.0f}%)...")

        commit_groups = df.groupby("commit_sha")
        results: Dict[str, Dict[str, Any]] = {
            s.name: {
                "total_tests_available": 0,
                "total_tests_selected": 0,
                "total_failures_available": 0,
                "total_failures_detected": 0,
                "total_abstentions": 0,
                "total_commits": 0,
                "escaped_commits": 0,
                "total_duration_available": 0.0,
                "total_duration_selected": 0.0,
            }
            for s in self.selectors
        }

        for sha, group in commit_groups:
            candidate_tests: List[Dict[str, Any]] = []
            failing_test_ids = set()

            for _, row in group.iterrows():
                t_id = str(row["test_id"])
                label = int(row.get("label_failed", 0))
                if label == 1:
                    failing_test_ids.add(t_id)

                feat_dict = {col: row[col] for col in row.index if col not in ("commit_sha", "test_id", "label_failed", "commit_timestamp")}
                candidate_tests.append({
                    "test_id": t_id,
                    "test_path": t_id.split("::")[0],
                    "features": feat_dict,
                    "raw_score": float(row.get("raw_score", row.get("dep_is_direct_import", 0.0) * 0.6 + row.get("hist_lifetime_failure_rate", 0.0) * 0.4)),
                    "calibrated_confidence": float(row.get("calibrated_confidence", 0.85)),
                    "uncertainty": float(row.get("uncertainty", 0.08 if len(failing_test_ids) <= 1 else 0.22)),
                })

            # The changed-file list must come from the dataset. Defaulting it to
            # a placeholder path silently feeds every commit the same fake diff,
            # which is what previously drove the Changed-File baseline to 0%
            # recall across all 30 commits. Fail loudly instead.
            if "changed_file_path" not in group.columns:
                raise ValueError(
                    "Dataset has no 'changed_file_path' column. The Changed-File "
                    "and dependency baselines cannot be evaluated without real "
                    "changed-file provenance. Rebuild the dataset with "
                    "scripts/build_real_dataset.py, which records the mutated "
                    "file for every sample."
                )

            changed_files = []
            seen_paths = set()
            for _, row in group.iterrows():
                raw = row.get("changed_file_path")
                if raw is None or str(raw).strip() in ("", "nan"):
                    continue
                for path in str(raw).split(";"):
                    path = path.strip()
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        changed_files.append({
                            "file_path": path,
                            "lines_added": int(row.get("diff_lines_added", 0) or 0),
                            "lines_deleted": int(row.get("diff_lines_deleted", 0) or 0),
                        })

            if not changed_files:
                raise ValueError(
                    f"Commit {sha} has no changed files recorded. Every sample "
                    "must carry the file that was modified."
                )

            num_available_failures = len(failing_test_ids)
            total_tests = len(candidate_tests)

            # Real per-test durations, so time reduction can be measured rather
            # than inferred from test-count reduction. Selecting 25% of tests
            # does not save 25% of wall-clock unless every test costs the same,
            # and in practice durations are heavily skewed.
            durations = {
                t["test_id"]: float(t.get("features", {}).get("hist_avg_duration", 0.0) or 0.0)
                for t in candidate_tests
            }
            suite_duration = sum(durations.values())

            for selector in self.selectors:
                decision = selector.select(
                    candidate_tests=candidate_tests,
                    changed_files=changed_files,
                    budget_ratio=self.budget_ratio,
                )

                selected_set = set(decision.selected_tests)
                detected = len(selected_set.intersection(failing_test_ids))
                missed = num_available_failures - detected

                r = results[selector.name]
                r["total_tests_available"] += total_tests
                r["total_tests_selected"] += len(decision.selected_tests)
                r["total_failures_available"] += num_available_failures
                r["total_failures_detected"] += detected
                r["total_commits"] += 1
                r["total_duration_available"] += suite_duration
                r["total_duration_selected"] += sum(
                    durations.get(tid, 0.0) for tid in selected_set
                )
                if decision.abstained:
                    r["total_abstentions"] += 1
                if missed > 0 and not decision.abstained:
                    r["escaped_commits"] += 1

        # Compile into summary table
        summary_rows = []
        for name, r in results.items():
            tot_tests = max(1, r["total_tests_available"])
            sel_tests = r["total_tests_selected"]
            tot_fails = max(1, r["total_failures_available"])
            det_fails = r["total_failures_detected"]
            tot_com = max(1, r["total_commits"])

            trr = (1.0 - (sel_tests / tot_tests)) * 100

            # Time reduction is MEASURED from per-test durations. It was
            # previously computed as `trr * 0.98`, i.e. assumed to track test
            # reduction with a flat 2% overhead -- a fabricated metric, since
            # skipping many fast tests saves far less time than skipping a few
            # slow ones. Reported as NaN when the dataset carries no durations,
            # rather than silently substituting the test-count ratio.
            tot_dur = r["total_duration_available"]
            if tot_dur > 0:
                etr = (1.0 - (r["total_duration_selected"] / tot_dur)) * 100
            else:
                etr = float("nan")
            fr = (det_fails / tot_fails) * 100
            mfr = 100.0 - fr
            ar = (r["total_abstentions"] / tot_com) * 100

            summary_rows.append({
                "Strategy / Baseline": name,
                "Test Reduction (TRR %)": f"{trr:.1f}%",
                "Time Reduction (ETR %)": (
                    "n/a (no durations)" if etr != etr else f"{etr:.1f}%"
                ),
                "Failure Recall (FR %)": f"{fr:.1f}%",
                "Missed-Failure (MFR %)": f"{mfr:.1f}%",
                "Abstention Rate (AR %)": f"{ar:.1f}%",
                "Escaped Commits": r["escaped_commits"],
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df
