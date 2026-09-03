"""
ConfTest Multi-Repository Cross-Project Generalization Engine.

Evaluates Zero-Shot transferability and Leave-One-Project-Out (LOPO) cross-validation
to benchmark RTS model generalization across heterogeneous codebases.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.calibration import TemperatureScalingCalibrator, compute_ece
from conftest.models.trainer import evaluate_predictions
from conftest.evaluation.budget import (
    recall_at_budget_per_commit as _recall_at_budget_per_commit,
)
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def _qualified_groups(
    repo: str, dataset: Dict[str, np.ndarray], n_rows: int
) -> np.ndarray:
    """
    Commit ids prefixed with their repository, so ids cannot collide across repos.

    Mutant ids are assigned per repository and restart at zero, so pooling five
    repositories without qualifying the ids would silently merge five different
    commits into one group and let a commit's rows land in both halves of the
    calibration split. Falls back to one group per row when no ids are supplied,
    which reproduces a row-wise split rather than pretending to a grouped one.
    """
    groups = dataset.get("groups")
    if groups is None:
        return np.array([f"{repo}::row{i}" for i in range(n_rows)], dtype=object)
    return np.array([f"{repo}::{g}" for g in np.asarray(groups).ravel()], dtype=object)


def _split_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fraction: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Split rows into a fit half and a calibration half along whole groups.

    Returns:
        (X_fit, y_fit, X_cal, y_cal, note) where note records how many groups and
        rows landed on each side, so the report can state it rather than imply it.

    Raises:
        ValueError: If either half would be empty, or if the calibration half
            carries no failure at all -- a temperature fitted on all-negative
            labels is not a calibration, and returning one silently would make
            every downstream ECE meaningless.
    """
    unique = np.unique(groups)
    rng = np.random.RandomState(random_seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_fit_groups = int(len(shuffled) * fit_fraction)
    n_fit_groups = min(max(n_fit_groups, 1), len(shuffled) - 1) if len(shuffled) > 1 else 1
    fit_groups = set(shuffled[:n_fit_groups].tolist())

    is_fit = np.array([g in fit_groups for g in groups], dtype=bool)
    if not is_fit.any() or is_fit.all():
        raise ValueError(
            f"grouped 85/15 split left one half empty: {len(unique)} groups over "
            f"{len(y)} rows. Cross-repo transfer needs a calibration half that is "
            "disjoint from the fitting half."
        )
    if int(y[~is_fit].sum()) == 0:
        raise ValueError(
            "the calibration half holds no failing row, so a temperature fitted on "
            "it would be fitted on one class. Widen the split or pool more source "
            "repositories."
        )

    note = (
        f"{n_fit_groups} of {len(unique)} source commits fit the model "
        f"({int(is_fit.sum())} rows); the remaining {len(unique) - n_fit_groups} "
        f"({int((~is_fit).sum())} rows) fit the calibrator. Split on shuffled "
        "commit ids, so no commit is in both halves and every source repository "
        "is in each."
    )
    return X[is_fit], y[is_fit], X[~is_fit], y[~is_fit], note

class CrossRepoEvaluator:
    """
    Evaluates RTS generalization across distinct repositories in Leave-One-Project-Out fashion.
    """

    def __init__(self, random_seed: int = 42, n_estimators: int = 150):
        """
        Args:
            random_seed: Reproducibility seed.
            n_estimators: Trees per transfer model. Defaults to the ensemble's own
                setting, so a transfer result is about the model the study
                reports rather than a smaller stand-in trained for the occasion.
        """
        self.random_seed = random_seed
        self.n_estimators = n_estimators

    def evaluate_lopo_transfer(
        self,
        repo_datasets: Dict[str, Dict[str, np.ndarray]],
        budget_ratio: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Execute Leave-One-Project-Out (LOPO) transfer evaluation.

        For each target repo R:
            Train on all repos != R
            Calibrate on held-out validation of training repos
            Evaluate zero-shot on target repo R

        Args:
            repo_datasets: repo_name -> {'X': ndarray, 'y': ndarray, 'groups':
                ndarray}. 'groups' names the commit each row belongs to -- one
                mutant applied to one checkout. It is what makes recall@budget a
                selection metric: a policy chooses a fraction of the tests for the
                commit in front of it, so recall has to be computed per commit and
                pooled. Ranking every row of a repository in one list and taking
                the top quarter is not a policy anyone can deploy, and it scores
                differently. Omitting 'groups' therefore yields a NaN recall rather
                than the pooled-ranking number.
            budget_ratio: Fraction of each commit's own test universe a policy may
                run (default: 0.25).

        Returns:
            Dictionary containing individual repo transfer results and macro-averaged metrics.
        """
        repo_names = list(repo_datasets.keys())
        if len(repo_names) < 2:
            raise ValueError("Cross-repo evaluation requires at least 2 repositories.")

        results: Dict[str, Any] = {"per_repository": {}, "macro_average": {}}
        macro_pr_aucs: List[float] = []
        macro_roc_aucs: List[float] = []
        macro_recalls: List[float] = []
        macro_eces: List[float] = []

        for target_repo in repo_names:
            logger.info(f"Evaluating LOPO transfer to target repository: {target_repo}...")

            # 1. Split training (all other repos) and target
            train_X_list, train_y_list, train_group_list = [], [], []
            for source_repo in repo_names:
                if source_repo != target_repo:
                    source = repo_datasets[source_repo]
                    train_X_list.append(source["X"])
                    train_y_list.append(source["y"])
                    train_group_list.append(
                        _qualified_groups(source_repo, source, len(source["y"]))
                    )

            X_train_combined = np.vstack(train_X_list)
            y_train_combined = np.concatenate(train_y_list)
            groups_combined = np.concatenate(train_group_list)

            X_target = repo_datasets[target_repo]["X"]
            y_target = repo_datasets[target_repo]["y"]
            target_groups = repo_datasets[target_repo].get("groups")

            # Sub-split the source repos for calibration, 85/15 by commit rather
            # than by position. Taking the first 85% of vertically stacked repos
            # puts the calibration half inside whichever repo happened to be last,
            # so the temperature would be fitted on one project and applied to a
            # model trained on the others. Splitting on shuffled commit ids keeps
            # every source repo in both halves and keeps a commit's rows together.
            X_tr, y_tr, X_cal, y_cal, split_note = _split_by_group(
                X_train_combined,
                y_train_combined,
                groups_combined,
                fit_fraction=0.85,
                random_seed=self.random_seed,
            )

            # 2. Train model
            model = LightGBMTestPredictor(
                random_seed=self.random_seed, n_estimators=self.n_estimators
            )
            model.train(X_train=X_tr, y_train=y_tr)

            # 3. Fit Temperature Scaling calibrator
            calibrator = TemperatureScalingCalibrator()
            val_raw = model.predict_proba(X_cal)
            calibrator.fit(val_raw, y_cal)

            # 4. Zero-shot prediction on target repo
            target_raw = model.predict_proba(X_target)
            target_cal = calibrator.calibrate(target_raw)

            # Metrics
            metrics = evaluate_predictions(y_target, target_cal)
            ece_res = compute_ece(y_target, target_cal)

            recall_budget, recall_detail = _recall_at_budget_per_commit(
                y_target, target_cal, target_groups, budget_ratio
            )

            repo_res = {
                "train_samples": len(X_tr),
                "calibration_samples": len(X_cal),
                "calibration_split": split_note,
                "target_samples": len(X_target),
                "target_failure_rate": round(float(np.mean(y_target)), 4),
                "n_estimators": self.n_estimators,
                "zero_shot_pr_auc": round(float(metrics["pr_auc"]), 4),
                "zero_shot_roc_auc": round(float(metrics["roc_auc"]), 4),
                "zero_shot_brier_score": round(float(metrics["brier_score"]), 4),
                "zero_shot_calibrated_ece": round(float(ece_res[0]), 4),
                "zero_shot_recall_at_budget": (
                    round(recall_budget, 4) if recall_budget == recall_budget else None
                ),
                "budget_ratio": budget_ratio,
                "recall_is_measured_over": recall_detail,
            }

            results["per_repository"][target_repo] = repo_res
            macro_pr_aucs.append(repo_res["zero_shot_pr_auc"])
            macro_roc_aucs.append(repo_res["zero_shot_roc_auc"])
            if recall_budget == recall_budget:
                macro_recalls.append(recall_budget)
            macro_eces.append(repo_res["zero_shot_calibrated_ece"])

        results["macro_average"] = {
            "mean_pr_auc": round(float(np.mean(macro_pr_aucs)), 4),
            "mean_roc_auc": round(float(np.mean(macro_roc_aucs)), 4),
            "mean_recall_at_budget": (
                round(float(np.mean(macro_recalls)), 4) if macro_recalls else None
            ),
            "repositories_with_defined_recall": len(macro_recalls),
            "mean_calibrated_ece": round(float(np.mean(macro_eces)), 4),
            "total_repositories_evaluated": len(repo_names),
            "budget_ratio": budget_ratio,
            "macro_average_means": (
                "each repository counts once, so a repository with a large test "
                "universe does not dominate; the per-repository rows carry the "
                "sample sizes the average hides"
            ),
        }

        return results
