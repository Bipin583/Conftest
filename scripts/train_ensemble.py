"""
ConfTest 5-Seed Deep Ensemble Training CLI.

Trains a 5-member gradient boosted decision tree ensemble with distinct random seeds,
computes epistemic disagreement uncertainty, and exports serialized ensemble checkpoints.

Usage:
    python scripts/train_ensemble.py --train data/splits/train.csv --val data/splits/val.csv --test data/splits/test.csv --output-dir models/ensembles/5_seed_lgbm
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.logging_config import get_logger
from conftest.models.ensemble import DEFAULT_SEEDS, EnsembleUncertaintyPredictor
from conftest.models.trainer import evaluate_predictions, prepare_feature_arrays

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest 5-Seed Ensemble Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=str,
        default="./data/splits/train.csv",
        help="Path to training dataset CSV.",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="./data/splits/val.csv",
        help="Path to validation dataset CSV.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/splits/test.csv",
        help="Path to test dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Directory to save ensemble checkpoints and metadata.",
    )
    parser.add_argument(
        "--tuning-report",
        type=str,
        default="./reports/model_tuning_report.json",
        help="Validation-only tuning report whose selected_config will be trained.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./reports/ensemble_training_report.json",
        help="Destination for fitted diagnostics and optional held-out metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    tuning_path = Path(args.tuning_report)

    if not train_path.exists():
        logger.error(f"Training data {train_path} not found. Run build_splits.py first.")
        sys.exit(1)

    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    X_train, y_train = prepare_feature_arrays(train_df)
    X_val, y_val = prepare_feature_arrays(val_df) if val_df is not None else (None, None)
    X_test, y_test = prepare_feature_arrays(test_df) if test_df is not None else (None, None)

    if not tuning_path.exists():
        raise FileNotFoundError(
            f"Validation-only tuning report not found: {tuning_path}. Run scripts/tune_model.py first."
        )
    tuning_report = json.loads(tuning_path.read_text(encoding="utf-8"))
    if tuning_report.get("selection_split") != "validation" or tuning_report.get(
        "held_out_test_used_for_selection"
    ) is not False:
        raise ValueError("Tuning report does not prove validation-only model selection.")
    selected_config = dict(tuning_report["selected_config"])

    ensemble = EnsembleUncertaintyPredictor(
        seeds=DEFAULT_SEEDS,
        **selected_config,
    )
    summary = ensemble.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    test_metrics = None
    uncertainty_metrics = None
    if X_test is not None and y_test is not None:
        preds = ensemble.predict_with_uncertainty(X_test)
        metrics = evaluate_predictions(y_test, preds["mean_prob"])

        mean_uncertainty = float(preds["epistemic_std"].mean())
        max_uncertainty = float(preds["epistemic_std"].max())
        mean_entropy = float(preds["predictive_entropy"].mean())

        logger.info("\n=== 5-Seed Ensemble Test Evaluation ===")
        logger.info(f"PR-AUC (Average Precision): {metrics['pr_auc']:.4f}")
        logger.info(f"ROC-AUC:                    {metrics['roc_auc']:.4f}")
        logger.info(f"F1-Score:                   {metrics['f1_score']:.4f}")
        logger.info(f"Brier Score:                {metrics['brier_score']:.4f}")
        logger.info(f"Mean Epistemic Uncertainty: {mean_uncertainty:.4f}")
        logger.info(f"Max Epistemic Uncertainty:  {max_uncertainty:.4f}")
        logger.info(f"Mean Predictive Entropy:    {mean_entropy:.4f}")

        test_metrics = metrics
        uncertainty_metrics = {
            "mean_epistemic_std": mean_uncertainty,
            "max_epistemic_std": max_uncertainty,
            "mean_predictive_entropy": mean_entropy,
        }

    # Save ensemble
    saved_dir = ensemble.save_ensemble(args.output_dir)
    report = {
        "selected_config": selected_config,
        "selection_evidence": args.tuning_report,
        "selection_split": "validation",
        "held_out_test_used_for_selection": False,
        "training_summary": summary,
        "held_out_test_metrics": test_metrics,
        "held_out_uncertainty_metrics": uncertainty_metrics,
        "ensemble_directory": str(Path(saved_dir)),
        "produced_by": "python scripts/train_ensemble.py",
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"\nEnsemble successfully persisted to: {saved_dir}")
    logger.info(f"Training report: {report_path}")


if __name__ == "__main__":
    main()
