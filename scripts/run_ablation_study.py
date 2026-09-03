"""
ConfTest Feature Ablation Study CLI.

Runs Leave-One-Group-Out (LOGO) and Single-Group ablation experiments
evaluating PR-AUC, ROC-AUC, Brier score, and ECE degradation.

Usage:
    python scripts/run_ablation_study.py --output reports/ablation_study.json
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.ablation import FeatureAblationStudy
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Feature Ablation Study CLI",
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
        "--output",
        type=str,
        default="./reports/ablation_study.json",
        help="Destination path for ablation JSON report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    test_path = Path(args.test_data)

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        logger.error(
            f"One or more split datasets missing ({train_path}, {val_path}, "
            f"{test_path}). Run build_splits.py first."
        )
        sys.exit(1)

    logger.info("Loading chronological dataset splits...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = prepare_feature_arrays(df_train)
    X_val, y_val = prepare_feature_arrays(df_val)
    X_test, y_test = prepare_feature_arrays(df_test)

    logger.info(f"Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")

    if "commit_sha" not in df_test.columns:
        logger.error(
            "test split has no commit_sha column, so recall cannot be budgeted per "
            "commit. Rebuild the splits with scripts/build_splits.py."
        )
        sys.exit(1)
    groups_test = df_test["commit_sha"].to_numpy()

    study = FeatureAblationStudy(random_seed=42)
    report = study.run_study(
        X_train, y_train, X_val, y_val, X_test, y_test, groups_test=groups_test
    )
    report["provenance"] = {
        "labels_measured": True,
        "label_provenance": (
            "labels measured by executing test suites; splits built from "
            "data/processed/real_features.csv"
        ),
        "recall_rule": (
            "failure recall when each commit runs the top 25% of its own test "
            "universe, pooled over commits that had at least one failure"
        ),
        "test_commits": int(len(set(groups_test.tolist()))),
        "test_rows": int(len(df_test)),
    }

    # Print Summary Tables
    full = report["full_model"]
    logger.info("\n" + "=" * 90)
    logger.info("  ConfTest Feature Ablation Study Results")
    logger.info("=" * 90)
    logger.info(
        f"Full Model ({full['feature_count']} feats, "
        f"{full['informative_feature_count']} informative) | "
        f"PR-AUC: {full['pr_auc']:.4f} | Calibrated ECE: {full['calibrated_ece']:.4f} | "
        f"Recall@25%: {full['failure_recall_at_25budget']:.4f}"
    )
    logger.info(f"Recall denominator: {full['recall_denominator']}")
    logger.info("-" * 90)

    logger.info("--- Leave-One-Group-Out (LOGO) ---")
    for name, res in report["leave_one_group_out"].items():
        logger.info(
            f"{name:<24} | PR-AUC: {res['pr_auc']:.4f} ({res['delta_pr_auc']:+.4f}) | "
            f"ECE: {res['calibrated_ece']:.4f} ({res['delta_calibrated_ece']:+.4f}) | "
            f"Recall: {res['failure_recall_at_25budget']:.4f} ({res['delta_recall']:+.4f})"
        )

    logger.info("-" * 90)
    logger.info("--- Single-Group-Only Models ---")
    for name, res in report["single_group_only"].items():
        logger.info(
            f"{name:<24} | PR-AUC: {res['pr_auc']:.4f} ({res['delta_pr_auc']:+.4f}) | "
            f"ECE: {res['calibrated_ece']:.4f} ({res['delta_calibrated_ece']:+.4f}) | "
            f"Recall: {res['failure_recall_at_25budget']:.4f} ({res['delta_recall']:+.4f})"
        )
    logger.info("=" * 90)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Ablation study report exported to: {out_path}")


if __name__ == "__main__":
    main()
