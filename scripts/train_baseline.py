"""
ConfTest 8-Baseline RTS Comparison Experiment Runner.

Evaluates all 8 RTS baselines under identical test budget constraints
across temporal test splits and exports comparison tables and metrics.

Usage:
    python scripts/train_baseline.py --dataset data/splits/test.csv --budget 0.25 --output reports/baseline_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.evaluation.benchmark import BaselineBenchmarkRunner
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# The strategy every difference is measured against: the one the report is about.
DEFAULT_REFERENCE = "8. ConfTest (Calibrated + Selective Abstention)"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Baseline Comparison Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/splits/test.csv",
        help="Path to evaluation test split dataset CSV.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Test budget fraction (e.g. 0.25 = top 25% tests).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/baseline_comparison.csv",
        help="Destination path for benchmark results CSV.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
        help="Bootstrap resamples for the confidence intervals. 0 skips them.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
        help="Strategy that paired differences are taken against.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.dataset)

    # Fallback to processed features if split test.csv not present
    if not data_path.exists():
        fallback = Path("./data/processed/features.csv")
        if fallback.exists():
            logger.info(f"Test split {data_path} not found. Falling back to {fallback}...")
            data_path = fallback
        else:
            logger.error(f"Dataset file not found: {data_path}. Run extract_features.py or build_splits.py first.")
            sys.exit(1)

    logger.info(f"Loading benchmark dataset from {data_path}...")
    df = pd.read_csv(data_path)

    runner = BaselineBenchmarkRunner(budget_ratio=args.budget)
    results_df = runner.evaluate_dataset(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info(f"Benchmark results exported to: {out_path}")
    logger.info(
        f"\n=== RTS Baseline Comparison Table (Budget: {args.budget*100:.0f}%) ===\n"
        + results_df.to_string(index=False)
    )

    if args.bootstraps < 1:
        logger.warning(
            "--bootstraps 0: reporting point estimates with no intervals. At a few "
            "hundred commits the gap between two strategies can be smaller than the "
            "noise in either one, so these numbers should not be quoted on their own."
        )
        return

    intervals = runner.evaluate_dataset_with_intervals(
        df, num_bootstraps=args.bootstraps, reference=args.reference
    )
    interval_df = runner.interval_table(intervals)

    interval_csv = out_path.with_name(out_path.stem + "_intervals.csv")
    interval_json = out_path.with_name(out_path.stem + "_intervals.json")
    interval_df.to_csv(interval_csv, index=False)
    interval_json.write_text(json.dumps(intervals, indent=2, default=str))

    logger.info(f"Confidence intervals exported to: {interval_csv} and {interval_json}")
    logger.info(
        f"\n=== Same table with {intervals['confidence_level']*100:.0f}% bootstrap CIs "
        f"({intervals['num_bootstraps']} resamples over {intervals['num_commits']} commits, "
        f"unit = {intervals['resampling_unit']}) ===\n"
        + interval_df.to_string(index=False)
        + "\n\n* the paired difference in failure recall excludes zero."
    )


if __name__ == "__main__":
    main()
