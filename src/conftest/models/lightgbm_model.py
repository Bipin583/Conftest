"""
ConfTest LightGBM Model Wrapper.

Implements gradient-boosted decision tree training with class-imbalance weighting,
early stopping, model serialization, and feature importance analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.task_definition import compute_class_weights
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class LightGBMTestPredictor:
    """LightGBM gradient boosted tree model for regression test failure prediction."""

    def __init__(
        self,
        random_seed: int = 42,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        model_version: str = "lgbm_v1.0.0",
    ):
        """
        Initialize LightGBM model configuration.

        Args:
            random_seed: Reproducibility seed.
            n_estimators: Maximum number of boosting trees.
            learning_rate: Boosting shrinkage parameter.
            max_depth: Maximum tree depth.
            num_leaves: Maximum tree leaves.
            min_child_samples: Minimum samples per leaf. LightGBM's own default
                (20) proved too permissive on the mutation corpus: with 392k
                training rows the booster reached its best validation iteration
                within the first ~20 rounds and then only overfit. 100-200 keeps
                trees honest at this scale and was the single largest validation
                PR-AUC gain in the re-tune.
            subsample: Row subsampling (bagging) fraction.
            subsample_freq: Bag every k iterations. LightGBM ignores `subsample`
                entirely while this is 0, which is its own default -- so passing
                a subsample fraction alone silently trains on every row. Set to
                1 so the configured fraction actually takes effect; pass 0 to
                deliberately disable bagging.
            colsample_bytree: Feature column subsampling fraction.
            model_version: Semantic model version tag.
        """
        self.random_seed = random_seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.model_version = model_version

        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = list(FEATURE_NAMES)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 25,
        eval_metric: str = "average_precision",
        use_class_weight: bool = True,
    ) -> Dict[str, Any]:
        """
        Train LightGBM binary classifier with class weighting, sample weighting, and early stopping.

        Args:
            X_train: Training feature matrix of shape (N, 32).
            y_train: Binary ground-truth labels (N,).
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation labels.
            sample_weight: Optional 1D sample weights array for flakiness downweighting.
            early_stopping_rounds: Number of rounds without validation improvement before stopping.
            eval_metric: Validation metric that controls early stopping. Average precision
                matches the ranking objective under class imbalance.
            use_class_weight: Whether to apply the training split's negative/positive
                ratio through LightGBM's ``scale_pos_weight``.

        Returns:
            Dictionary containing training diagnostics and best iteration.
        """
        weights_dict = compute_class_weights(y_train)
        measured_scale_pos_weight = weights_dict.get("scale_pos_weight", 1.0)
        scale_pos_weight = measured_scale_pos_weight if use_class_weight else 1.0

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            # Required for `subsample` to do anything at all: LightGBM's
            # bagging_freq defaults to 0, which disables row bagging and makes
            # the fraction above dead configuration. Every model trained before
            # this line was added used 100% of rows per tree regardless of the
            # 0.8 it advertised -- which also means the deep ensemble's members
            # differed only by seed-dependent tie-breaking, not by data, so its
            # spread understated epistemic uncertainty.
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            objective="binary",
            metric="None",
            importance_type="gain",
            verbose=-1,
        )

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            )

        logger.info(
            f"Training LightGBM model (Seed: {self.random_seed}, "
            f"scale_pos_weight: {scale_pos_weight:.2f})..."
        )
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_metric=eval_metric if eval_set is not None else None,
            callbacks=callbacks if callbacks else None,
        )

        best_iter = getattr(self.model, "best_iteration_", self.n_estimators)
        actual_trees = int(self.model.booster_.num_trees())
        informative_features = int(np.count_nonzero(np.ptp(X_train, axis=0)))
        if best_iter:
            logger.info(
                f"Training complete. Early stopping kept {best_iter} "
                f"of {self.n_estimators} trees."
            )
        else:
            # best_iteration_ is 0 when no eval_set was supplied, so early stopping never ran and
            # every tree is kept. Logging that as "best iteration: 0" read like a degenerate fit.
            logger.info(
                f"Training complete. No validation set, so no early stopping: "
                f"all {self.n_estimators} trees kept."
            )

        # Record the bagging configuration that was actually in force, so a
        # report can never again claim row subsampling that did not happen.
        bagging_active = self.subsample_freq > 0 and self.subsample < 1.0
        return {
            "model_version": self.model_version,
            "configured_n_estimators": int(self.n_estimators),
            "best_iteration": int(best_iter) if best_iter else actual_trees,
            "actual_num_trees": actual_trees,
            "early_stopping_used": bool(eval_set is not None),
            "early_stopping_rounds": int(early_stopping_rounds) if eval_set is not None else None,
            "eval_metric": eval_metric if eval_set is not None else None,
            "use_class_weight": bool(use_class_weight),
            "measured_scale_pos_weight": float(measured_scale_pos_weight),
            "scale_pos_weight": float(scale_pos_weight),
            "n_features": X_train.shape[1],
            "n_informative_features": informative_features,
            "n_constant_features": int(X_train.shape[1] - informative_features),
            "n_train_samples": len(y_train),
            "subsample": float(self.subsample),
            "subsample_freq": int(self.subsample_freq),
            "effective_row_fraction_per_tree": float(self.subsample) if bagging_active else 1.0,
            "bagging_active": bool(bagging_active),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict failure probability scores P(y=1 | X).

        Args:
            X: Feature matrix of shape (N, 32).

        Returns:
            1D array of failure probabilities in range [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probas = self.model.predict_proba(X)
        return probas[:, 1].astype(np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return normalized feature importances based on total Gain."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        raw_importances = self.model.feature_importances_
        total = np.sum(raw_importances)
        norm_importances = (raw_importances / total) if total > 0 else raw_importances

        return {
            name: round(float(norm_importances[i]), 5)
            for i, name in enumerate(self.feature_names)
        }

    def save(self, filepath: str) -> str:
        """Serialize model artifact to disk using joblib."""
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out_path))
        logger.info(f"Model saved to {out_path}")
        return str(out_path)

    @classmethod
    def load(cls, filepath: str) -> "LightGBMTestPredictor":
        """Load serialized model artifact from disk."""
        in_path = Path(filepath).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found: {in_path}")
        instance = joblib.load(str(in_path))
        logger.info(f"Loaded {instance.model_version} from {in_path}")
        return instance
