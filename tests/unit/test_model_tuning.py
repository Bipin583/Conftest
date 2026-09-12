"""Tests for validation-only LightGBM configuration selection."""

from __future__ import annotations

import numpy as np

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from scripts.tune_model import tune_candidates


def _split(seed: int, rows: int = 120):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    logits = 1.8 * X[:, 0] - X[:, 1] + 0.4 * X[:, 2]
    y = (logits + rng.normal(scale=1.1, size=rows) > 1.2).astype(int)
    y[:2] = [0, 1]
    return X, y


def _candidates():
    return [
        {
            "n_estimators": 30, "learning_rate": 0.05, "max_depth": 3,
            "num_leaves": 7, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "use_class_weight": True,
            "early_stopping_rounds": 5,
        },
        {
            "n_estimators": 40, "learning_rate": 0.05, "max_depth": 4,
            "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "use_class_weight": False,
            "early_stopping_rounds": 5,
        },
    ]


def test_lightgbm_diagnostics_record_actual_fit_and_controls():
    X_train, y_train = _split(1)
    X_val, y_val = _split(2, rows=60)
    X_train[:, -2:] = 0.0
    predictor = LightGBMTestPredictor(n_estimators=40, subsample=0.7, subsample_freq=1)

    diagnostics = predictor.train(
        X_train, y_train, X_val, y_val,
        early_stopping_rounds=7,
        eval_metric="average_precision",
        use_class_weight=False,
    )

    assert diagnostics["actual_num_trees"] == predictor.model.booster_.num_trees()
    assert diagnostics["actual_num_trees"] <= diagnostics["configured_n_estimators"]
    assert diagnostics["eval_metric"] == "average_precision"
    assert diagnostics["early_stopping_rounds"] == 7
    assert diagnostics["use_class_weight"] is False
    assert diagnostics["scale_pos_weight"] == 1.0
    assert diagnostics["bagging_active"] is True
    assert diagnostics["n_constant_features"] == 2


def test_tuning_is_deterministic_and_names_validation_objective():
    X_train, y_train = _split(3)
    X_val, y_val = _split(4, rows=70)

    first = tune_candidates(X_train, y_train, X_val, y_val, _candidates(), random_seed=9)
    second = tune_candidates(X_train, y_train, X_val, y_val, _candidates(), random_seed=9)

    assert first == second
    assert first["objective"] == "validation_pr_auc"
    assert first["selection_split"] == "validation"
    assert first["held_out_test_used_for_selection"] is False
    assert first["selected_config"] == first["candidates"][first["selected_candidate_index"]]["config"]


def test_tuning_api_has_no_test_split_parameter():
    import inspect

    parameters = inspect.signature(tune_candidates).parameters

    assert "X_test" not in parameters
    assert "y_test" not in parameters
    assert "test_df" not in parameters
