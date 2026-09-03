"""
ConfTest Statistical Significance & Hypothesis Testing CLI.

Runs Wilcoxon Signed-Rank tests, Cliff's Delta effect sizes, and 95% Bootstrap CIs
comparing ConfTest against the other RTS baselines, commit by commit.

The paired observations come from `reports/baseline_per_commit.csv`, the unpooled
per-commit statistics that `scripts/train_baseline.py` records while it sweeps the
selectors over the evaluation split. This script computes nothing about the world on
its own and refuses to run when that artifact is absent.

Until 2026-09-03 it did the opposite: it drew 100 commits of recall and time
reduction for ConfTest and for each baseline from hand-chosen Beta and Normal
distributions, then published Wilcoxon p-values over them. Those p-values were real
arithmetic on invented numbers -- they measured the distance between two means
someone typed, and they would have reported p < 0.001 no matter how the system
actually performed. The label guard never caught it because it polices the label
path, and this was on the reporting path.

Usage:
    python scripts/train_baseline.py --bootstraps 2000     # produces the input
    python scripts/run_statistical_tests.py --output reports/statistical_significance.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.headline import MissingArtifact
from conftest.evaluation.statistics import (
    bootstrap_confidence_interval,
    StatisticalSignificanceTester,
)
from conftest.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_REFERENCE = "8. ConfTest (Calibrated + Selective Abstention)"
PRODUCED_BY = "python scripts/train_baseline.py --bootstraps 2000"

# A per-commit metric is defined only where its denominator is non-empty. Recall over
# a commit with no failure available is 0/0, and a commit whose tests all report a
# duration of 0.000 s has no time to reduce. Both are dropped from the paired sample
# and counted, because a dropped pair is a fact about the evidence, not a detail.
METRIC_DENOMINATORS = {
    "failure_recall": "failures_available",
    "time_reduction": "duration_available",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Statistical Significance Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--per-commit",
        type=str,
        default="./reports/baseline_per_commit.csv",
        help="Per-commit benchmark statistics exported by scripts/train_baseline.py.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
        help="Strategy every comparison is made against.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/statistical_significance.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--num-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for 95%% CIs.",
    )
    return parser.parse_args()


def load_per_commit(path: Path) -> pd.DataFrame:
    """
    Read the per-commit export, or say which command produces it.

    Raises:
        MissingArtifact: If the export does not exist. There is no fallback: the
            paired tests have nothing to pair without measured per-commit outcomes.
        ValueError: If the file exists but lacks the columns the tests need.
    """
    if not path.exists():
        raise MissingArtifact(path, PRODUCED_BY)

    frame = pd.read_csv(path)
    required = {"commit_sha", "strategy", *METRIC_DENOMINATORS.values(), *METRIC_DENOMINATORS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path.as_posix()} is missing {missing}. It was probably written by an "
            f"older version of the benchmark; regenerate it with: {PRODUCED_BY}"
        )
    return frame


def paired_arrays(
    reference: pd.DataFrame, other: pd.DataFrame, metric: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Align two strategies on commit_sha and keep the pairs where the metric is defined.

    Args:
        reference: Per-commit rows for the reference strategy.
        other: Per-commit rows for the strategy being compared.
        metric: 'failure_recall' or 'time_reduction'.

    Returns:
        (reference values, other values, counts) where counts records how many
        commits were available, how many were kept, and why the rest were dropped.
        A Wilcoxon test on 12 pairs is a different claim from one on 183, so the
        count travels with the result rather than being implied by it.
    """
    denominator = METRIC_DENOMINATORS[metric]
    merged = reference.merge(
        other, on="commit_sha", suffixes=("_ref", "_other"), validate="one_to_one"
    )
    defined = (merged[f"{denominator}_ref"] > 0) & (merged[f"{denominator}_other"] > 0)
    kept = merged[defined & merged[f"{metric}_ref"].notna() & merged[f"{metric}_other"].notna()]
    counts = {
        "commits_available": int(len(merged)),
        "pairs_used": int(len(kept)),
        "dropped_undefined_metric": int(len(merged) - len(kept)),
    }
    return (
        kept[f"{metric}_ref"].to_numpy(dtype=np.float64),
        kept[f"{metric}_other"].to_numpy(dtype=np.float64),
        counts,
    )


def main():
    args = parse_args()
    per_commit_path = Path(args.per_commit)

    frame = load_per_commit(per_commit_path)
    strategies: List[str] = list(dict.fromkeys(frame["strategy"]))
    if args.reference not in strategies:
        raise ValueError(
            f"reference strategy {args.reference!r} is not in {per_commit_path.as_posix()}. "
            f"It carries: {strategies}"
        )

    reference_rows = frame[frame["strategy"] == args.reference]
    logger.info(
        f"Read {len(frame)} per-commit rows for {len(strategies)} strategies over "
        f"{reference_rows['commit_sha'].nunique()} commits from {per_commit_path}."
    )

    tester = StatisticalSignificanceTester(random_seed=42)
    reference_metrics: Dict[str, np.ndarray] = {}
    for metric, denominator in METRIC_DENOMINATORS.items():
        defined = reference_rows[reference_rows[denominator] > 0]
        reference_metrics[metric] = defined[metric].dropna().to_numpy(dtype=np.float64)

    report_data = {
        "input": per_commit_path.as_posix(),
        "produced_by": PRODUCED_BY,
        "labels_measured": True,
        "observation_unit": "commit",
        "observation_unit_means": (
            "one mutant applied to one repository checkout, with its kill set measured "
            "by a full pytest run; the arrays below are per-commit outcomes, not draws"
        ),
        "metric_units": "percent",
        "reference": args.reference,
        "evaluation_commits": int(reference_rows["commit_sha"].nunique()),
        "reference_95ci": {
            metric: bootstrap_confidence_interval(
                values, num_bootstraps=args.num_bootstraps
            )
            for metric, values in reference_metrics.items()
        },
        "reference_defined_commits": {
            metric: int(len(values)) for metric, values in reference_metrics.items()
        },
        "pairwise_significance": {},
    }

    header = (
        f"{'Baseline Strategy':<34} | {'pairs':>5} | {'Wilcoxon p':<12} | "
        f"{'Sig (p<0.05)':<12} | {'Cliff delta':<12} | {'Effect Size':<12}"
    )
    logger.info("\n" + "=" * len(header))
    logger.info(header)
    logger.info("=" * len(header))

    for name in strategies:
        if name == args.reference:
            continue
        other_rows = frame[frame["strategy"] == name]

        pair_counts: Dict[str, Dict[str, int]] = {}
        ref_metrics: Dict[str, np.ndarray] = {}
        other_metrics: Dict[str, np.ndarray] = {}
        for metric in METRIC_DENOMINATORS:
            ref_values, other_values, counts = paired_arrays(reference_rows, other_rows, metric)
            ref_metrics[metric] = ref_values
            other_metrics[metric] = other_values
            pair_counts[metric] = counts

        result = tester.evaluate_pairwise(ref_metrics, other_metrics, name)
        result["pairs"] = pair_counts
        report_data["pairwise_significance"][name] = result

        recall = result["failure_recall"]
        logger.info(
            f"{name:<34} | {pair_counts['failure_recall']['pairs_used']:>5} | "
            f"p = {recall['p_value']:<8.5f} | "
            f"{str(recall['statistically_significant_p05']):<12} | "
            f"d = {recall['cliffs_delta']:<8.4f} | {recall['effect_size']:<12}"
        )

    logger.info("=" * len(header))
    dropped = {
        metric: int(
            report_data["evaluation_commits"] - report_data["reference_defined_commits"][metric]
        )
        for metric in METRIC_DENOMINATORS
    }
    logger.info(
        "Commits with no defined value for the metric, and so absent from its test: "
        f"failure_recall {dropped['failure_recall']} "
        f"(no test in the universe detected the mutant), "
        f"time_reduction {dropped['time_reduction']} (no measured test durations)."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info(f"Statistical report exported to: {out_path}")


if __name__ == "__main__":
    main()
