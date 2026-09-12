"""
ConfTest LightGBM Failure Scoring Model
High-throughput Gradient Boosted Decision Tree classifier for test failure scoring.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **conftest.models.lightgbm_model** (LightGBMTestPredictor, including the row-bagging fix (subsample_freq)).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "conftest.models.lightgbm_model"

import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional
import os
import pickle

class TestFailureScorer:
    """
    Trains and predicts failure probabilities for (Commit, Test) pairs using LightGBM.
    """
    FEATURE_NAMES = [
        "lines_added",
        "lines_deleted",
        "total_churn",
        "modified_files_count",
        "ast_node_delta",
        "has_interface_change",
        "has_import_change",
        "direct_dependency_match",
        "dependency_overlap_score",
        "historical_failure_rate",
        "avg_test_duration",
        "flakiness_score"
    ]

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 5):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbosity=-1
        )
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Trains the LightGBM model on tabular feature matrix X and binary labels y (1 = Test Failed, 0 = Test Passed).
        """
        feats = feature_names or self.FEATURE_NAMES
        self.model.fit(X, y, feature_name=feats)
        self.is_fitted = True

        importances = dict(zip(feats, self.model.feature_importances_))
        return {
            "training_samples": len(X),
            "feature_importances": importances
        }

    def predict_raw_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw logit scores (before sigmoid/temperature scaling).
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before predicting.")
        # LightGBM booster raw margin
        booster = self.model.booster_
        return booster.predict(X, raw_score=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns standard uncalibrated probability of test failure.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before predicting.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
            self.is_fitted = True
