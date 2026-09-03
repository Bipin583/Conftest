"""
ConfTest Flakiness Stress Testing & Robustness Evaluation Subsystem.

Simulates intermittent non-deterministic test failures (label noise injection)
and evaluates model robustness with sample downweighting and selective abstention.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from conftest.evaluation.budget import recall_at_budget_per_commit
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.calibration import TemperatureScalingCalibrator, compute_ece
from conftest.models.trainer import evaluate_predictions
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def inject_flakiness_noise(
    y: np.ndarray,
    noise_rate: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic non-deterministic flakiness into training labels.

    Args:
        y: Original binary ground-truth labels (0 = Pass, 1 = Fail).
        noise_rate: Fraction of samples to flip (0.0 to 0.50).
        random_seed: Reproducibility seed.

    Returns:
        Tuple of (noisy_y, flakiness_scores).
    """
    y = np.asarray(y).copy()
    n = len(y)
    if noise_rate <= 0.0:
        return y, np.zeros(n, dtype=np.float32)

    rng = np.random.RandomState(random_seed)
    # Generate continuous flakiness scores in [0, 1]
    flakiness_scores = rng.beta(0.5, 3.0, size=n).astype(np.float32)

    # Flip labels for the top-k most flaky tests
    num_to_flip = int(n * noise_rate)
    flip_indices = np.argsort(flakiness_scores)[-num_to_flip:]

    noisy_y = y.copy()
    noisy_y[flip_indices] = 1 - noisy_y[flip_indices]

    # Ensure at least 1 positive and 1 negative remains
    if np.sum(noisy_y) == 0:
        noisy_y[0] = 1
    elif np.sum(noisy_y) == n:
        noisy_y[0] = 0

    return noisy_y, flakiness_scores


class FlakinessStressTester:
    """Evaluates RTS model degradation across increasing degrees of test flakiness noise."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def evaluate_noise_level(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_rate: float,
        groups_test: Optional[np.ndarray] = None,
        budget_ratio: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate Standard Unweighted ML vs. ConfTest Robust Downweighted ML at a noise rate.

        Noise is injected into the training labels only. Every metric is scored against the
        clean held-out labels, so a model cannot score well by learning the corruption.

        `groups_test` carries the commit id per test row. Without it the budget recall is
        undefined rather than pooled over a global ranking, because a global ranking lets
        one commit spend another commit's budget.
        """
        # Inject noise into training set
        noisy_y_train, flakiness_scores = inject_flakiness_noise(
            y_train, noise_rate=noise_rate, random_seed=self.random_seed
        )

        # 1. Baseline: Standard Unweighted Model
        std_model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
        std_model.train(X_train=X_train, y_train=noisy_y_train, sample_weight=None)
        std_test_probs = std_model.predict_proba(X_test)
        std_metrics = evaluate_predictions(y_test, std_test_probs)

        # 2. ConfTest Robust: Flakiness Downweighted Model
        # Sample weight w_i = (1.0 - 0.7 * flakiness_score_i)
        robust_weights = np.clip(1.0 - (0.7 * flakiness_scores), 0.1, 1.0)
        robust_model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
        robust_model.train(X_train=X_train, y_train=noisy_y_train, sample_weight=robust_weights)

        # Temperature calibration
        raw_val_probs = robust_model.predict_proba(X_val)
        calibrator = TemperatureScalingCalibrator()
        calibrator.fit(raw_val_probs, y_val)
        robust_cal_probs = calibrator.calibrate(robust_model.predict_proba(X_test))

        robust_metrics = evaluate_predictions(y_test, robust_cal_probs)
        cal_ece = compute_ece(y_test, robust_cal_probs)

        # Failure recall under a per-commit budget. The previous version ranked the whole
        # test set at once and divided by max(1, failures), which turns "no failure to
        # find" into a denominator of one and reports a miss as 0.0.
        std_recall, std_note = recall_at_budget_per_commit(
            y_test, std_test_probs, groups_test, budget_ratio
        )
        rob_recall, rob_note = recall_at_budget_per_commit(
            y_test, robust_cal_probs, groups_test, budget_ratio
        )

        # Flipping labels in a 5%-positive dataset overwhelmingly turns passes into
        # failures, so the training prevalence climbs with the noise rate. Published so a
        # reader can see that a metric improving under noise may be reporting the
        # prevalence change rather than robustness.
        train_prevalence = float(np.mean(noisy_y_train))

        return {
            "noise_rate_pct": round(noise_rate * 100, 1),
            "train_positive_rate": round(train_prevalence, 4),
            "train_positive_rate_clean": round(float(np.mean(y_train)), 4),
            "labels_flipped": int(np.sum(noisy_y_train != np.asarray(y_train))),
            "evaluation_labels": "clean held-out labels; noise is injected into training only",
            "recall_denominator_standard": std_note,
            "recall_denominator_robust": rob_note,
            "standard_unweighted": {
                "pr_auc": round(float(std_metrics["pr_auc"]), 4),
                "roc_auc": round(float(std_metrics["roc_auc"]), 4),
                "brier_score": round(float(std_metrics["brier_score"]), 4),
                "failure_recall": round(std_recall, 4),
            },
            "conftest_robust": {
                "pr_auc": round(float(robust_metrics["pr_auc"]), 4),
                "roc_auc": round(float(robust_metrics["roc_auc"]), 4),
                "brier_score": round(float(robust_metrics["brier_score"]), 4),
                "calibrated_ece": round(float(cal_ece[0]), 4),
                "failure_recall": round(rob_recall, 4),
                "temperature": round(calibrator.temperature, 4),
            },
            "robustness_advantage": {
                "delta_recall": round(rob_recall - std_recall, 4),
                "delta_pr_auc": round(float(robust_metrics["pr_auc"] - std_metrics["pr_auc"]), 4),
            },
        }

    def run_stress_grid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_levels: List[float],
        groups_test: Optional[np.ndarray] = None,
        budget_ratio: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Run stress grid across multiple noise levels."""
        results = []
        for noise in noise_levels:
            logger.info(f"Evaluating flakiness noise level: {noise*100:.1f}%...")
            res = self.evaluate_noise_level(
                X_train, y_train, X_val, y_val, X_test, y_test, noise,
                groups_test=groups_test, budget_ratio=budget_ratio,
            )
            results.append(res)
        return results
