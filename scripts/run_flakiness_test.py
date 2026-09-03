"""
ConfTest Flakiness Stress Testing CLI.

Evaluates model resilience to injected label noise in the TRAINING labels, scored against
clean held-out labels.

Usage:
    python scripts/run_flakiness_test.py --noise-levels 0.0,0.05,0.10,0.20,0.30 \
        --output reports/flakiness_robustness.json
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.flakiness import FlakinessStressTester
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Flakiness Stress Test CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="./data/splits/train.csv",
        help="Path to training set CSV.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="./data/splits/val.csv",
        help="Path to validation set CSV.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="./data/splits/test.csv",
        help="Path to test set CSV.",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.0,0.05,0.10,0.20,0.30",
        help="Comma-separated flakiness noise ratios.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/flakiness_robustness.json",
        help="Destination path for output JSON report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    test_path = Path(args.test_data)

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        logger.error("Dataset split files missing. Run build_splits.py first.")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = prepare_feature_arrays(df_train)
    X_val, y_val = prepare_feature_arrays(df_val)
    X_test, y_test = prepare_feature_arrays(df_test)

    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",")]
    logger.info(f"Running Flakiness Stress Test across noise levels: {noise_levels}")

    # Commit ids for the held-out rows: recall is a per-commit budget, so the tester needs
    # to know which rows compete with each other.
    groups_test = df_test["commit_sha"].to_numpy() if "commit_sha" in df_test.columns else None
    if groups_test is None:
        logger.warning(
            "Test split carries no commit_sha column, so budget recall will be reported as "
            "undefined rather than pooled over a global ranking."
        )

    tester = FlakinessStressTester(random_seed=42)
    results = tester.run_stress_grid(
        X_train, y_train, X_val, y_val, X_test, y_test, noise_levels,
        groups_test=groups_test,
    )

    logger.info("\n" + "=" * 90)
    logger.info("  ConfTest Flakiness Robustness Stress Test Results")
    logger.info("=" * 90)
    logger.info(
        f"{'Noise Level':<12} | {'TrainPos%':<9} | {'Std PR-AUC':<11} | {'Robust PR-AUC':<13} | "
        f"{'Std Recall':<11} | {'Robust Recall':<13} | Advantage"
    )
    logger.info("-" * 90)

    for item in results:
        std = item["standard_unweighted"]
        rob = item["conftest_robust"]
        adv = item["robustness_advantage"]
        logger.info(
            f"{item['noise_rate_pct']:>4.1f}% Noise | "
            f"{item['train_positive_rate'] * 100:>8.2f}% | "
            f"{std['pr_auc']:<11.4f} | "
            f"{rob['pr_auc']:<13.4f} | "
            f"{std['failure_recall']:<11.4f} | "
            f"{rob['failure_recall']:<13.4f} | "
            f"dRecall: {adv['delta_recall']:+.4f}"
        )
    logger.info("=" * 90)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_measured": True,
                "noise_is_injected_into": "training labels only",
                "evaluated_against": "clean held-out labels",
                "recall_rule": "per-commit budget; see recall_denominator_* on each row",
                "confound_to_read_first": "the dataset is about 5% positive, so flipping "
                "labels mostly turns passes into failures and raises the training "
                "prevalence. A metric that improves with the noise rate may be reporting "
                "that prevalence change and not robustness to flakiness.",
                "stress_test_grid": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Robustness report saved to: {out_path}")


if __name__ == "__main__":
    main()
