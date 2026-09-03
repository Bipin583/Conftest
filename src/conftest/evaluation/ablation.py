"""
ConfTest Feature Ablation & Contribution Study Engine.

Executes Leave-One-Group-Out (LOGO) and Single-Group ablation experiments across
the 32-feature pipeline to quantify individual and group contribution to RTS
accuracy and calibration.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.evaluation.budget import recall_at_budget_per_commit
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.calibration import TemperatureScalingCalibrator, compute_ece
from conftest.models.trainer import prepare_feature_arrays, evaluate_predictions
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Canonical feature group definitions exactly matching FEATURE_NAMES
FEATURE_GROUPS: Dict[str, List[str]] = {
    "diff_churn": [
        "diff_lines_added", "diff_lines_deleted", "diff_total_churn",
        "diff_num_files_changed", "diff_num_src_files", "diff_num_test_files",
        "diff_has_python", "diff_has_config", "diff_msg_length",
        "diff_msg_word_count", "diff_is_fix_commit", "diff_is_refactor_commit",
    ],
    "ast_complexity": [
        "ast_test_file_functions_count", "ast_test_file_classes_count",
        "ast_test_file_imports_count", "ast_test_file_complexity",
        "ast_test_is_parameterized", "ast_test_func_name_length",
    ],
    "dependency_graph": [
        "dep_is_direct_import", "dep_name_heuristic_coupled", "dep_shortest_path_depth",
        "dep_is_reachable", "dep_max_reverse_dependencies", "dep_test_total_out_degree",
    ],
    "history_telemetry": [
        "hist_total_prior_runs", "hist_prior_failures", "hist_lifetime_failure_rate",
        "hist_recent_10_failure_rate", "hist_avg_duration", "hist_flaky_score",
        "hist_has_ever_failed", "hist_changed_files_prior_mod_count",
    ],
}


def get_feature_indices(group_names: List[str]) -> List[int]:
    """Retrieve column indices for specified feature groups."""
    target_names = set()
    for g in group_names:
        target_names.update(FEATURE_GROUPS.get(g, []))
    return [i for i, name in enumerate(FEATURE_NAMES) if name in target_names]


def _ece_value(ece: Any) -> float:
    """compute_ece has returned both a tuple and a dict across revisions."""
    if isinstance(ece, tuple):
        return float(ece[0])
    if isinstance(ece, dict):
        return float(ece["ece"])
    return float(ece)


def _delta(value: float, reference: float) -> float:
    """A difference against an unmeasured recall is unmeasured, not zero."""
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return round(value - reference, 4)


class FeatureAblationStudy:
    """Orchestrates Leave-One-Group-Out and Single-Group feature ablation experiments."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def _train_and_evaluate_subset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_indices: List[int],
        groups_test: Optional[np.ndarray] = None,
        budget_ratio: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Train LightGBM + Temperature Scaling on a feature subset and evaluate on test set.

        Reports how many of the subset's columns are constant in the training rows. An
        ablation that removes twelve constant columns produces an identical model, and
        without that count the reader has to guess whether a zero delta means the group
        does not matter or that the group was never there to remove.
        """
        X_tr_sub = X_train[:, feature_indices]
        X_val_sub = X_val[:, feature_indices]
        X_te_sub = X_test[:, feature_indices]

        # 1. Train LightGBM model
        model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
        model.train(X_train=X_tr_sub, y_train=y_train)

        # 2. Evaluate raw uncalibrated probabilities
        raw_val_probs = model.predict_proba(X_val_sub)
        raw_test_probs = model.predict_proba(X_te_sub)

        # 3. Fit Temperature Scaling calibrator on validation set
        calibrator = TemperatureScalingCalibrator()
        calibrator.fit(raw_val_probs, y_val)
        cal_test_probs = calibrator.calibrate(raw_test_probs)

        # 4. Compute metrics
        raw_metrics = evaluate_predictions(y_test, raw_test_probs)
        raw_ece = compute_ece(y_test, raw_test_probs)
        cal_ece = compute_ece(y_test, cal_test_probs)

        # Failure recall when each commit may run its own 25%. A global top-k lets one
        # commit borrow another's budget, which is not a selection a CI system can make,
        # and it scores higher because the easy commits absorb the budget.
        recall_at_budget, budget_note = recall_at_budget_per_commit(
            y_test, cal_test_probs, groups_test, budget_ratio
        )
        constant_columns = int(np.sum(np.ptp(X_tr_sub, axis=0) == 0.0))

        return {
            "pr_auc": round(float(raw_metrics["pr_auc"]), 4),
            "roc_auc": round(float(raw_metrics["roc_auc"]), 4),
            "brier_score": round(float(raw_metrics["brier_score"]), 4),
            "uncalibrated_ece": round(_ece_value(raw_ece), 4),
            "calibrated_ece": round(_ece_value(cal_ece), 4),
            "failure_recall_at_25budget": round(recall_at_budget, 4),
            "recall_denominator": budget_note,
            "feature_count": len(feature_indices),
            "constant_features_in_train": constant_columns,
            "informative_feature_count": len(feature_indices) - constant_columns,
        }

    def run_study(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_test: Optional[np.ndarray] = None,
        budget_ratio: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Execute full ablation grid:
        1. Full Model (All 32 features)
        2. Leave-One-Group-Out (w/o Diff, w/o AST, w/o Dep Graph, w/o History)
        3. Single-Group-Only (Diff Only, AST Only, Dep Graph Only, History Only)

        Args:
            groups_test: Commit id per test row. Without it the recall column is NaN
                and says so, because a budget is spent per commit, not per dataset.
        """
        all_indices = list(range(len(FEATURE_NAMES)))
        all_group_names = list(FEATURE_GROUPS.keys())

        # 1. Full Model Baseline
        logger.info("Training Full Model Baseline (32 features)...")
        full_res = self._train_and_evaluate_subset(
            X_train, y_train, X_val, y_val, X_test, y_test, all_indices,
            groups_test, budget_ratio,
        )

        # 2. Leave-One-Group-Out (LOGO)
        logo_results = {}
        for omit_group in all_group_names:
            remaining_groups = [g for g in all_group_names if g != omit_group]
            sub_indices = get_feature_indices(remaining_groups)
            logger.info(f"Evaluating LOGO: w/o {omit_group} ({len(sub_indices)} features)...")
            res = self._train_and_evaluate_subset(
                X_train, y_train, X_val, y_val, X_test, y_test, sub_indices,
                groups_test, budget_ratio,
            )
            res["delta_pr_auc"] = round(res["pr_auc"] - full_res["pr_auc"], 4)
            res["delta_calibrated_ece"] = _delta(
                res["calibrated_ece"], full_res["calibrated_ece"]
            )
            res["delta_recall"] = _delta(
                res["failure_recall_at_25budget"], full_res["failure_recall_at_25budget"]
            )
            logo_results[f"without_{omit_group}"] = res

        # 3. Single-Group-Only
        single_results = {}
        for single_group in all_group_names:
            sub_indices = get_feature_indices([single_group])
            logger.info(
                f"Evaluating Single-Group: {single_group} only "
                f"({len(sub_indices)} features)..."
            )
            res = self._train_and_evaluate_subset(
                X_train, y_train, X_val, y_val, X_test, y_test, sub_indices,
                groups_test, budget_ratio,
            )
            res["delta_pr_auc"] = round(res["pr_auc"] - full_res["pr_auc"], 4)
            res["delta_calibrated_ece"] = _delta(
                res["calibrated_ece"], full_res["calibrated_ece"]
            )
            res["delta_recall"] = _delta(
                res["failure_recall_at_25budget"], full_res["failure_recall_at_25budget"]
            )
            single_results[f"{single_group}_only"] = res

        return {
            "full_model": full_res,
            "leave_one_group_out": logo_results,
            "single_group_only": single_results,
        }
