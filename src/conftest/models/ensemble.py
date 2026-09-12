"""
ConfTest 5-Seed Ensemble Uncertainty Estimation Engine.

Trains an ensemble of gradient-boosted decision trees over distinct random seeds
to quantify epistemic uncertainty (model disagreement), predictive entropy,
and commit-level risk for selective prediction.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SEEDS = [42, 101, 2024, 777, 999]


class EnsembleUncertaintyPredictor:
    """5-Seed Ensemble model quantifying epistemic uncertainty and predictive entropy."""

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 25,
        eval_metric: str = "average_precision",
        use_class_weight: bool = True,
        ensemble_version: str = "ensemble_v1.0.0",
    ):
        """
        Initialize ensemble configuration.

        Args:
            seeds: List of integer random seeds (defaults to 5 distinct seeds).
            n_estimators: Trees per ensemble member.
            learning_rate: Boosting learning rate.
            max_depth: Maximum tree depth.
            num_leaves: Maximum leaves per tree.
            subsample: Row subsample (bagging) ratio per tree.
            subsample_freq: Bag every k iterations; 0 disables bagging and makes
                `subsample` inert. This ensemble's epistemic spread comes from
                seeds drawing different row bags, so 0 collapses the members
                towards a single fit -- see LightGBMTestPredictor.
            colsample_bytree: Feature subsample ratio per tree.
            early_stopping_rounds: Validation rounds without improvement before stopping.
            eval_metric: Validation metric controlling early stopping.
            use_class_weight: Whether to apply the training split's class ratio.
            ensemble_version: Semantic version tag.
        """
        self.seeds = seeds or DEFAULT_SEEDS
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.use_class_weight = use_class_weight
        self.ensemble_version = ensemble_version
        self.models: List[LightGBMTestPredictor] = []
        self.training_diagnostics: List[Dict[str, Any]] = []

    @property
    def ensemble_size(self) -> int:
        return len(self.models)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train all M ensemble member models across the specified random seeds.

        Args:
            X_train: Training feature matrix (N, 32).
            y_train: Ground truth labels (N,).
            X_val: Validation feature matrix for early stopping.
            y_val: Validation labels.

        Returns:
            Dictionary containing ensemble training summary.
        """
        self.models = []
        self.training_diagnostics = []
        logger.info(f"Training {len(self.seeds)}-seed ensemble ({self.ensemble_version})...")

        for idx, seed in enumerate(self.seeds, 1):
            logger.info(f"Training ensemble member {idx}/{len(self.seeds)} (Seed: {seed})...")
            predictor = LightGBMTestPredictor(
                random_seed=seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                subsample_freq=self.subsample_freq,
                colsample_bytree=self.colsample_bytree,
                model_version=f"{self.ensemble_version}_member_{idx}",
            )
            diagnostics = predictor.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric,
                use_class_weight=self.use_class_weight,
            )
            diagnostics["seed"] = int(seed)
            self.training_diagnostics.append(diagnostics)
            self.models.append(predictor)

        # Members differ only through the randomness the config actually enables.
        # With bagging off they all see identical rows, and the disagreement this
        # class reports as epistemic uncertainty largely disappears -- so the
        # diagnostics say plainly whether that randomness was in force.
        bagging_active = self.subsample_freq > 0 and self.subsample < 1.0
        if not bagging_active:
            logger.warning(
                "Ensemble trained with row bagging DISABLED (subsample=%s, "
                "subsample_freq=%s): every member sees all rows, so member "
                "disagreement understates epistemic uncertainty.",
                self.subsample,
                self.subsample_freq,
            )
        return {
            "ensemble_version": self.ensemble_version,
            "num_members": len(self.models),
            "seeds": self.seeds,
            "n_samples": len(y_train),
            "configured_n_estimators": int(self.n_estimators),
            "actual_num_trees": [d["actual_num_trees"] for d in self.training_diagnostics],
            "eval_metric": self.eval_metric if X_val is not None and y_val is not None else None,
            "early_stopping_rounds": (
                int(self.early_stopping_rounds)
                if X_val is not None and y_val is not None
                else None
            ),
            "use_class_weight": bool(self.use_class_weight),
            "member_diagnostics": self.training_diagnostics,
            "subsample": float(self.subsample),
            "subsample_freq": int(self.subsample_freq),
            "bagging_active": bool(bagging_active),
        }

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute mean probability, epistemic standard deviation, and predictive entropy across ensemble.

        Args:
            X: Feature matrix of shape (N, 32) or 1D shape (32,).

        Returns:
            Dictionary containing:
                - 'mean_prob': 1D array of mean failure probabilities P_bar in [0, 1].
                - 'epistemic_std': 1D array of model disagreement standard deviation sigma in [0, 0.5].
                - 'epistemic_variance': 1D array of model variance sigma^2.
                - 'predictive_entropy': 1D array of binary Shannon entropy in [0, 1].
                - 'member_predictions': 2D array of shape (M, N) containing raw member outputs.
        """
        if not self.models:
            raise RuntimeError("Ensemble models have not been trained yet.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from each ensemble member
        member_preds = np.array([m.predict_proba(X) for m in self.models])  # Shape: (M, N)

        # 1. Mean ensemble prediction: bar{p} = 1/M sum_m p_m
        mean_prob = np.mean(member_preds, axis=0)

        # 2. Epistemic uncertainty (model disagreement): sigma = std(p_m)
        epistemic_std = np.std(member_preds, axis=0)
        epistemic_var = np.var(member_preds, axis=0)

        # 3. Binary predictive entropy: H(bar{p}) = -bar{p} log2 bar{p} - (1 - bar{p}) log2 (1 - bar{p})
        eps = 1e-9
        p_clipped = np.clip(mean_prob, eps, 1.0 - eps)
        entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))

        return {
            "mean_prob": mean_prob.astype(np.float32),
            "epistemic_std": epistemic_std.astype(np.float32),
            "epistemic_variance": epistemic_var.astype(np.float32),
            "predictive_entropy": entropy.astype(np.float32),
            "member_predictions": member_preds.astype(np.float32),
        }

    def compute_commit_level_uncertainty(self, X_commit: np.ndarray) -> Dict[str, float]:
        """
        Aggregate test-level uncertainties into a single commit-level decision metric U(c).

        Args:
            X_commit: Feature matrix for all candidate tests belonging to a commit (K, 32).

        Returns:
            Dictionary containing max uncertainty, mean uncertainty, 95th percentile, and top risk score.
        """
        res = self.predict_with_uncertainty(X_commit)
        stds = res["epistemic_std"]
        probs = res["mean_prob"]

        return {
            "max_epistemic_std": float(np.max(stds)),
            "mean_epistemic_std": float(np.mean(stds)),
            "p95_epistemic_std": float(np.percentile(stds, 95)),
            "max_failure_prob": float(np.max(probs)),
            "mean_failure_prob": float(np.mean(probs)),
        }

    def save_ensemble(self, directory_path: str) -> str:
        """Save all ensemble members and metadata to disk."""
        out_dir = Path(directory_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        member_files = []
        for idx, model in enumerate(self.models, 1):
            fn = f"member_{idx}_seed_{model.random_seed}.joblib"
            model.save(str(out_dir / fn))
            # Store the bare filename, not an absolute path. Members always live
            # beside their metadata, and an absolute path bakes in one machine's
            # checkout -- this repo already carries a training report pointing at
            # a directory that does not exist here.
            member_files.append(fn)

        meta = {
            "ensemble_version": self.ensemble_version,
            "seeds": self.seeds,
            "num_members": len(self.models),
            "member_files": member_files,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric": self.eval_metric,
            "use_class_weight": self.use_class_weight,
            "member_diagnostics": self.training_diagnostics,
            # Recorded so a reader can tell whether the members were actually
            # trained on different row samples, rather than inferring it from a
            # config value that LightGBM may have ignored.
            "subsample": float(self.subsample),
            "subsample_freq": int(self.subsample_freq),
            "colsample_bytree": float(self.colsample_bytree),
            "bagging_active": bool(self.subsample_freq > 0 and self.subsample < 1.0),
        }

        meta_path = out_dir / "ensemble_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Ensemble ({len(self.models)} members) saved to {out_dir}")
        return str(out_dir)

    @classmethod
    def load_ensemble(cls, directory_path: str) -> "EnsembleUncertaintyPredictor":
        """Load serialized ensemble members from directory."""
        in_dir = Path(directory_path).resolve()
        meta_path = in_dir / "ensemble_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found in: {in_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Metadata written before bagging was recorded has no subsample_freq
        # key. Those ensembles were trained with LightGBM's default of 0, i.e.
        # no bagging at all, so default to 0 here rather than to the current
        # constructor default -- otherwise a loaded legacy ensemble would
        # describe itself as bagged when it was not.
        legacy_freq = meta.get("subsample_freq", 0)
        instance = cls(
            seeds=meta["seeds"],
            n_estimators=meta.get("n_estimators", 150),
            learning_rate=meta.get("learning_rate", 0.05),
            max_depth=meta.get("max_depth", 6),
            num_leaves=meta.get("num_leaves", 31),
            subsample=meta.get("subsample", 0.8),
            subsample_freq=legacy_freq,
            colsample_bytree=meta.get("colsample_bytree", 0.8),
            early_stopping_rounds=meta.get("early_stopping_rounds", 25),
            eval_metric=meta.get("eval_metric", "average_precision"),
            use_class_weight=meta.get("use_class_weight", True),
            ensemble_version=meta.get("ensemble_version", "ensemble_v1.0.0"),
        )
        instance.training_diagnostics = meta.get("member_diagnostics", [])
        if "subsample_freq" not in meta:
            logger.warning(
                "Ensemble at %s predates the row-bagging fix: its members were "
                "trained on all rows despite a subsample=0.8 setting, so its "
                "epistemic-uncertainty spread is narrower than a bagged "
                "ensemble's. Retrain to obtain bagged members.",
                in_dir,
            )

        instance.models = []
        for fp in meta["member_files"]:
            file_path = Path(fp)
            if not file_path.exists():
                file_path = in_dir / file_path.name
            m = LightGBMTestPredictor.load(str(file_path))
            instance.models.append(m)

        logger.info(f"Loaded ensemble ({len(instance.models)} members) from {in_dir}")
        return instance
