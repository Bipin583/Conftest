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
from conftest.evaluation.statistics import (
    bootstrap_counts,
    format_interval,
    intervals_from_replicates,
)

logger = get_logger(__name__)

# Every reported metric is a ratio of sums over commits, so these per-commit
# numerators and denominators are the sufficient statistics for all of them.
COMMIT_STAT_FIELDS = (
    "tests_available",
    "tests_selected",
    "failures_available",
    "failures_detected",
    "duration_available",
    "duration_selected",
    "abstained",
    "escaped",
)

# What baselines 6, 7 and 8 rank by. These are model outputs, not features, so the
# caller has to score the split and pass them in; the benchmark will not synthesise
# them. raw_score is the ensemble mean probability, calibrated_confidence is that
# probability after models/calibrator.joblib, and uncertainty is the across-member
# standard deviation the abstention rule thresholds.
MODEL_SCORE_COLUMNS = ("raw_score", "calibrated_confidence", "uncertainty")

# Column headings, so the interval table and the point-estimate table name the
# same metrics the same way.
METRIC_LABELS = {
    "trr": "Test Reduction (TRR %)",
    "etr": "Time Reduction (ETR %)",
    "fr": "Failure Recall (FR %)",
    "mfr": "Missed-Failure (MFR %)",
    "ar": "Abstention Rate (AR %)",
}


def _pct(value: float) -> str:
    """Format a percentage, refusing to print nan as though it were a measurement."""
    return "n/a" if value != value else f"{value:.1f}%"


def _ratio(numerator: Any, denominator: Any, scale: float = 100.0) -> Any:
    """
    Elementwise ratio that is NaN -- not zero, not inf -- where the denominator is empty.

    Works on scalars and on whole vectors of bootstrap replicates, so the metric
    formulas below are written once and used by both paths.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe * scale, np.nan)


def _metrics_from_totals(totals: Dict[str, Any], n_commits: Any) -> Dict[str, Any]:
    """
    The six reported metrics, as functions of pooled totals.

    Time reduction is MEASURED from per-test durations. It was previously computed
    as `trr * 0.98`, i.e. assumed to track test reduction with a flat 2% overhead --
    a fabricated metric, since skipping many fast tests saves far less time than
    skipping a few slow ones. It is NaN when the dataset carries no durations,
    rather than silently substituting the test-count ratio.
    """
    fr = _ratio(totals["failures_detected"], totals["failures_available"])
    return {
        "trr": 100.0 - _ratio(totals["tests_selected"], totals["tests_available"]),
        "etr": 100.0 - _ratio(totals["duration_selected"], totals["duration_available"]),
        "fr": fr,
        "mfr": 100.0 - fr,
        "ar": _ratio(totals["abstained"], n_commits),
        "escaped_commits": totals["escaped"],
        "n_commits": n_commits,
    }


def fold_commit_metrics(
    stats: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Pool per-commit numerators into the six reported metrics.

    Args:
        stats: field name -> per-commit array, as returned by accumulate_per_commit.
        idx: Commit indices to pool over. None means all of them.

    Returns:
        trr, etr, fr, mfr, ar, escaped_commits, n_commits.

    A metric whose denominator is empty over the given commits is NaN, not zero.
    "No failing test was available" is a different observation from "no failing test
    was found", and the previous max(1, denominator) guard silently turned the first
    into 0% recall -- harmless when pooling the whole dataset once, a downward bias
    on every bootstrap replicate that happens to draw no failures.
    """
    totals = {
        field: float(arr.sum() if idx is None else arr[idx].sum())
        for field, arr in stats.items()
    }
    n_commits = float(len(stats["tests_available"]) if idx is None else len(idx))
    return {
        key: float(value)
        for key, value in _metrics_from_totals(totals, n_commits).items()
    }



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
        # Filled by accumulate_per_commit. Empty until one has run, so a caller
        # that exports per-commit rows without accumulating first gets an empty
        # list rather than a plausible-looking misalignment.
        self.last_commit_order: List[str] = []

    def accumulate_per_commit(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run all 8 baselines and keep each metric's numerators per commit, unpooled.

        Every reported metric is a ratio of sums over commits, so the per-commit
        numerators and denominators are its sufficient statistics. Keeping them is
        what lets a bootstrap resample commits and recompute all six metrics
        without re-running the selectors once per replicate.

        Args:
            df: DataFrame containing 'commit_sha', 'test_id', 'label_failed', and feature columns.

        Returns:
            strategy name -> field name -> array with one entry per commit.
        """
        logger.info(f"Running 8-baseline benchmark across {df['commit_sha'].nunique()} commits (Budget: {self.budget_ratio*100:.0f}%)...")

        # A sweep must not depend on how many sweeps preceded it in this process.
        for selector in self.selectors:
            selector.reset()

        commit_groups = df.groupby("commit_sha")
        # The order the arrays are built in, kept so a per-commit export can name
        # which commit each entry belongs to. groupby sorts, so this is stable
        # across runs, but relying on that silently is how columns get misaligned.
        self.last_commit_order: List[str] = []
        results: Dict[str, Dict[str, List[float]]] = {
            s.name: {field: [] for field in COMMIT_STAT_FIELDS} for s in self.selectors
        }

        # Baselines 6, 7 and 8 are model strategies: their ranking IS the model's
        # output. Until 2026-09-03 this loop invented that output when the columns
        # were absent -- raw_score as a hand-weighted 0.6/0.4 blend of two features,
        # calibrated_confidence as the constant 0.85, and uncertainty as 0.08 or 0.22
        # chosen by len(failing_test_ids), which reads the ground-truth labels of the
        # commit and feeds them to the abstention rule. The published comparison was
        # therefore not a measurement of this system. Refuse instead, and name the
        # script that produces the columns.
        missing_scores = [c for c in MODEL_SCORE_COLUMNS if c not in df.columns]
        if missing_scores:
            raise ValueError(
                "Dataset is missing the model score columns "
                f"{missing_scores}. Baselines 6-8 rank by model output, so a "
                "benchmark without it measures nothing. Score the split first: "
                "scripts/train_baseline.py attaches raw_score, "
                "calibrated_confidence and uncertainty from "
                "models/ensembles/5_seed_lgbm and models/calibrator.joblib."
            )

        for sha, group in commit_groups:
            self.last_commit_order.append(str(sha))
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
                    "raw_score": float(row["raw_score"]),
                    "calibrated_confidence": float(row["calibrated_confidence"]),
                    "uncertainty": float(row["uncertainty"]),
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
                r["tests_available"].append(float(total_tests))
                r["tests_selected"].append(float(len(decision.selected_tests)))
                r["failures_available"].append(float(num_available_failures))
                r["failures_detected"].append(float(detected))
                r["duration_available"].append(float(suite_duration))
                r["duration_selected"].append(
                    float(sum(durations.get(tid, 0.0) for tid in selected_set))
                )
                r["abstained"].append(1.0 if decision.abstained else 0.0)
                r["escaped"].append(1.0 if (missed > 0 and not decision.abstained) else 0.0)

        return {
            name: {field: np.asarray(values, dtype=np.float64) for field, values in fields.items()}
            for name, fields in results.items()
        }

    def per_commit_frame(
        self, per_commit: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        The unpooled per-commit statistics as one tidy row per (commit, strategy).

        Pooled metrics answer 'how do the strategies compare on this dataset'. A
        paired significance test asks a different question -- 'does the difference
        hold commit by commit' -- and it needs the per-commit numerators, not the
        pooled ratio. Exporting them is what lets the statistical tests run on
        measured observations instead of on a distribution someone chose.

        Args:
            per_commit: Return value of accumulate_per_commit, from the same run
                that populated last_commit_order.

        Returns:
            One row per (commit_sha, strategy) carrying every field in
            COMMIT_STAT_FIELDS, plus the per-commit recall and time reduction
            derived from them -- NaN where the denominator is empty, since a
            commit with no failure available did not achieve 0% recall.

        Raises:
            ValueError: If the recorded commit order does not match the array
                lengths, which would silently mislabel every row.
        """
        order = list(self.last_commit_order)
        rows: List[Dict[str, Any]] = []
        for strategy, stats in per_commit.items():
            length = len(stats[COMMIT_STAT_FIELDS[0]])
            if length != len(order):
                raise ValueError(
                    f"{strategy}: {length} per-commit entries but "
                    f"{len(order)} commit ids were recorded. The export would "
                    "attach each number to the wrong commit; re-run "
                    "accumulate_per_commit and export from that same result."
                )
            for i, sha in enumerate(order):
                row: Dict[str, Any] = {"commit_sha": sha, "strategy": strategy}
                for field in COMMIT_STAT_FIELDS:
                    row[field] = float(stats[field][i])
                row["failure_recall"] = float(
                    _ratio(row["failures_detected"], row["failures_available"])
                )
                row["time_reduction"] = float(
                    100.0 - _ratio(row["duration_selected"], row["duration_available"])
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        per_commit: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> pd.DataFrame:
        """
        Run all 8 baselines on a tabular dataset containing commits, test cases, and ground-truth labels.

        Args:
            df: DataFrame containing 'commit_sha', 'test_id', 'label_failed', and feature columns.
            per_commit: An already-accumulated sweep to fold, so a caller that also
                wants intervals or a per-commit export pays for one sweep instead
                of three. None runs the sweep here.

        Returns:
            Comparison summary DataFrame with TRR, ETR, FR, MFR, and Abstention Rate.
        """
        if per_commit is None:
            per_commit = self.accumulate_per_commit(df)

        summary_rows = []
        for name, stats in per_commit.items():
            m = fold_commit_metrics(stats)
            summary_rows.append({
                "Strategy / Baseline": name,
                "Test Reduction (TRR %)": _pct(m["trr"]),
                "Time Reduction (ETR %)": (
                    "n/a (no durations)" if m["etr"] != m["etr"] else f"{m['etr']:.1f}%"
                ),
                "Failure Recall (FR %)": _pct(m["fr"]),
                "Missed-Failure (MFR %)": _pct(m["mfr"]),
                "Abstention Rate (AR %)": _pct(m["ar"]),
                "Escaped Commits": int(m["escaped_commits"]),
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def evaluate_dataset_with_intervals(
        self,
        df: pd.DataFrame,
        num_bootstraps: int = 2000,
        ci: float = 0.95,
        reference: Optional[str] = None,
        random_seed: int = 42,
        per_commit: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        The benchmark table with a bootstrap confidence interval on every number in it.

        A point estimate pooled over a few hundred commits is not a result on its
        own: at this sample size the gap between two strategies can be smaller than
        the noise in either one. The resampling unit is the commit, because the test
        rows inside a commit share a diff and an injected fault and are therefore
        not independent draws -- resampling rows would report intervals several
        times too narrow.

        All strategies are evaluated on each resample, so with `reference` set, the
        difference between a strategy and the reference is paired and its interval
        answers the question the report actually asks.

        Args:
            df: As accumulate_per_commit.
            num_bootstraps: Resamples (default: 2000).
            ci: Confidence level (default: 0.95).
            reference: Strategy that differences are taken against, e.g. the
                ConfTest selector's name. None reports marginal intervals only.
            random_seed: Reproducibility seed.
            per_commit: An already-accumulated sweep, as evaluate_dataset.

        Returns:
            Dict with 'metrics' (metric -> intervals and differences per strategy)
            and the resampling metadata.
        """
        if per_commit is None:
            per_commit = self.accumulate_per_commit(df)
        strategies = list(per_commit)
        if reference is not None and reference not in per_commit:
            raise ValueError(
                f"reference strategy {reference!r} is not one of {strategies}"
            )

        n_commits = int(len(per_commit[strategies[0]]["tests_available"])) if strategies else 0
        logger.info(
            f"Bootstrapping {len(strategies)} strategies x {len(METRIC_LABELS)} metrics "
            f"over {n_commits} commits ({num_bootstraps} resamples, unit = commit)..."
        )

        # One resample matrix, shared by every strategy and every metric: that
        # sharing is what makes the differences paired. Multiplicities rather than
        # indices, so each replicate's totals are a matrix product.
        counts = bootstrap_counts(n_commits, num_bootstraps, random_seed)

        replicate_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        point_metrics: Dict[str, Dict[str, float]] = {}
        for name, stats in per_commit.items():
            matrix = np.column_stack([stats[field] for field in COMMIT_STAT_FIELDS])
            totals = counts @ matrix
            by_field = {field: totals[:, i] for i, field in enumerate(COMMIT_STAT_FIELDS)}
            replicate_metrics[name] = _metrics_from_totals(by_field, float(n_commits))
            point_metrics[name] = fold_commit_metrics(stats)

        metrics: Dict[str, Any] = {}
        for metric, label in METRIC_LABELS.items():
            folded = intervals_from_replicates(
                {name: replicate_metrics[name][metric] for name in strategies},
                {name: point_metrics[name][metric] for name in strategies},
                ci=ci,
                num_units=n_commits,
                reference=reference,
            )
            metrics[metric] = {
                "label": label,
                "intervals": folded["intervals"],
                "differences": folded.get("differences", {}),
            }

        return {
            "resampling_unit": "commit",
            "resampling_unit_means": (
                "commits are resampled whole; test rows within a commit are not "
                "independent, so resampling rows would understate every interval"
            ),
            "num_commits": n_commits,
            "num_bootstraps": num_bootstraps,
            "confidence_level": ci,
            "budget_ratio": self.budget_ratio,
            "reference": reference or "",
            "strategies": strategies,
            "metrics": metrics,
        }

    @staticmethod
    def interval_table(result: Dict[str, Any], precision: int = 1) -> pd.DataFrame:
        """
        Render evaluate_dataset_with_intervals as a report-ready table.

        One row per strategy, one column per metric, each cell 'point [lower, upper]'.
        When the result carries a reference strategy, a final column gives the paired
        difference in failure recall and whether its interval excludes zero -- the
        one comparison the evaluation turns on.
        """
        reference = result.get("reference") or None
        rows = []
        for strategy in result["strategies"]:
            row = {"Strategy / Baseline": strategy}
            for metric, block in result["metrics"].items():
                row[block["label"]] = format_interval(
                    block["intervals"][strategy], precision=precision, unit="%"
                )
            if reference is not None and strategy != reference:
                diff = result["metrics"]["fr"]["differences"].get(strategy, {})
                verdict = format_interval(diff, precision=precision, unit="%")
                if diff.get("excludes_zero"):
                    verdict += " *"
                row["FR vs reference"] = verdict
            elif reference is not None:
                row["FR vs reference"] = "reference"
            rows.append(row)
        return pd.DataFrame(rows)
