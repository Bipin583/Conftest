"""
ConfTest Model Training & Performance Evaluation Orchestrator.

Loads temporal datasets, trains gradient boosted tree models, computes
multi-metric evaluations (PR-AUC, ROC-AUC, F1, Recall, Brier Score),
and exports serialized model checkpoints and training diagnostic reports.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

from conftest.config import PROJECT_ROOT
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.logging_config import get_logger

logger = get_logger(__name__)

PRODUCED_BY = "python scripts/train_model.py"


def _portable_path(path: Path) -> str:
    """
    Render a path so the report stays readable in someone else's checkout.

    Reports are read long after they are written, often on another machine.
    Recording the absolute path bakes in one filesystem: this repository
    carries a training report whose `model_file` is
    `models/...` under a `Desktop/main project/` checkout that does
    not exist here, so the one field that says which artifact the metrics
    describe points at nothing. Paths inside the project are therefore stored
    relative to the project root, in POSIX form; anything genuinely outside it
    keeps its absolute path, because a relative path to an unrelated location
    would be worse than an honest absolute one.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def prepare_feature_arrays(
    df: pd.DataFrame,
    label_col: str = "label_failed",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ordered 32-feature matrix X and binary target vector y from DataFrame."""
    missing = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required canonical features: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    # Fill any unexpected NaNs with zero
    X = np.nan_to_num(X, nan=0.0)

    y = df[label_col].values.astype(int) if label_col in df.columns else np.zeros(len(df), dtype=int)
    return X, y


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """
    Compute comprehensive scientific evaluation metrics for imbalanced test failure prediction.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted failure probabilities in range [0, 1].
        threshold: Decision threshold for discrete classification metrics.

    Returns:
        Dictionary containing Precision, Recall, F1, PR-AUC, ROC-AUC, Brier Score, and Log Loss.
        A metric that is undefined for the given inputs is None, never a
        substitute value -- callers that report it must handle null.

    Raises:
        ValueError: propagated from scikit-learn if a metric cannot be computed
            for a reason other than a single-class y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(np.sum(y_true == 1))
    single_class = n_pos == 0 or n_pos == len(y_true)

    # Metrics computation with safe division
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # The degenerate cases below are guarded by n_pos, not by try/except.
    #
    # These three used to sit under `except Exception`, each with a plausible
    # stand-in: pr_auc 0.0, roc_auc 0.5, log_loss 0.0. That swallowed real
    # failures as if they were degenerate inputs. It was not hypothetical:
    # log_loss was called with `eps=1e-7`, a keyword scikit-learn removed in
    # 1.5, so on any modern sklearn it raised TypeError on every call and every
    # training report published "log_loss": 0.0 -- a number no computation had
    # produced. A metric that cannot be computed is reported as None (JSON
    # null), never as a value; anything unexpected now propagates.
    pr_auc = 0.0 if n_pos == 0 else float(average_precision_score(y_true, y_prob))
    roc_auc = 0.5 if single_class else float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    if single_class:
        # Cross-entropy against a constant label vector is not informative, and
        # sklearn cannot infer both classes from y_true alone.
        lloss: Optional[float] = None
    else:
        # Clipping replaces the removed `eps` kwarg: it keeps log(0) out of the
        # sum for probabilities that saturate at exactly 0 or 1.
        lloss = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1.0 - 1e-7), labels=[0, 1]))

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": None if lloss is None else round(lloss, 4),
        "positive_samples": n_pos,
        "total_samples": len(y_true),
    }


class ModelTrainer:
    """Orchestrates end-to-end model training, validation, and serialization."""

    def __init__(
        self,
        output_dir: str = "./models/ensembles",
        model_version: str = "lgbm_v1.0.0",
        random_seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save model checkpoints and reports.
            model_version: Semantic version identifier.
            random_seed: Reproducibility seed.
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.random_seed = random_seed

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Train LightGBM model on temporal train split, validate on val split, and evaluate on test split.

        Returns:
            Dictionary containing model file path, feature importances, and evaluation metrics.
        """
        logger.info(f"Starting model training pipeline ({self.model_version})...")

        X_train, y_train = prepare_feature_arrays(train_df)
        X_val, y_val = prepare_feature_arrays(val_df) if val_df is not None and not val_df.empty else (None, None)
        X_test, y_test = prepare_feature_arrays(test_df) if test_df is not None and not test_df.empty else (None, None)

        predictor = LightGBMTestPredictor(
            random_seed=self.random_seed,
            model_version=self.model_version,
        )

        train_diag = predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        # Evaluate on validation set
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_probs = predictor.predict_proba(X_val)
            val_metrics = evaluate_predictions(y_val, val_probs)
            logger.info(f"Validation Metrics: PR-AUC={val_metrics['pr_auc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}, F1={val_metrics['f1_score']:.4f}")

        # Evaluate on test set
        test_metrics = {}
        if X_test is not None and y_test is not None:
            test_probs = predictor.predict_proba(X_test)
            test_metrics = evaluate_predictions(y_test, test_probs)
            logger.info(f"Test Metrics: PR-AUC={test_metrics['pr_auc']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, F1={test_metrics['f1_score']:.4f}")

        # Extract feature importances
        feature_importances = predictor.get_feature_importances()
        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

        # Save model artifact
        model_filename = f"{self.model_version}_seed{self.random_seed}.joblib"
        model_path = predictor.save(str(self.output_dir / model_filename))

        # Save training report JSON
        report = {
            "model_version": self.model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.random_seed,
            "produced_by": PRODUCED_BY,
            "model_file": _portable_path(Path(model_path)),
            "model_file_relative_to": "project_root",
            "training_diagnostics": train_diag,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_10_features_by_gain": top_features,
            "all_feature_importances": feature_importances,
        }

        report_path = self.output_dir / f"{self.model_version}_training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report exported to: {report_path}")
        return report
