"""Layer 1: the gradient-boosted ranker and the metrics that describe it.

The model is the least novel part of the system and the easiest to get
quietly wrong, because a 5%-positive corpus makes accuracy meaningless: a
classifier that predicts "passes" for every row scores 95%. The tests here
pin the hyper-parameters to the config, check the imbalance handling, and
assert that the reported metrics are the ones that survive class imbalance.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.train import (
    TrainingError,
    _resolve_scale_pos_weight,
    build_model,
    classification_metrics,
)


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def test_model_hyper_parameters_come_from_the_config(config):
    """No hyper-parameter is hard-coded in the trainer.

    The specification fixes these five values, and the config is the single
    place they are allowed to live.
    """
    model = build_model(config)
    params = model.get_params()
    for name in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert params[name] == config["model"][name], f"{name} drifted from config.yaml"


def test_scale_pos_weight_auto_is_the_negative_to_positive_ratio():
    """``auto`` must mean n_negative / n_positive, the XGBoost convention."""
    labels = np.array([0] * 95 + [1] * 5)
    assert _resolve_scale_pos_weight("auto", labels) == pytest.approx(19.0)


def test_scale_pos_weight_accepts_an_explicit_override():
    """A float in the config wins over the derived value."""
    labels = np.array([0] * 95 + [1] * 5)
    assert _resolve_scale_pos_weight(3.5, labels) == pytest.approx(3.5)


def test_scale_pos_weight_can_be_switched_off():
    """``null`` disables the weighting rather than defaulting to 1.0 silently."""
    assert _resolve_scale_pos_weight(None, np.array([0, 1, 0, 1])) is None


def test_scale_pos_weight_refuses_a_split_with_no_failures():
    """A label-free split is a data bug, and must stop the run.

    The ``auto`` ratio would divide by zero here. Raising beats returning a
    fallback weight: a model trained on a split with no failures cannot rank
    anything, and discovering that at evaluation time wastes the whole run.
    """
    with pytest.raises(TrainingError, match="no failing rows"):
        _resolve_scale_pos_weight("auto", np.zeros(100, dtype=int))


def test_unknown_scale_pos_weight_setting_is_rejected():
    """Only ``auto``, a number, or null are meaningful values."""
    with pytest.raises(TrainingError):
        _resolve_scale_pos_weight("balanced", np.array([0, 1, 0, 1]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_report_the_ones_that_survive_class_imbalance():
    """Accuracy alone is useless at a 5% base rate, so PR-AUC is reported."""
    rng = np.random.default_rng(0)
    labels = (rng.random(5000) < 0.05).astype(int)
    scores = np.clip(
        np.where(labels == 1, rng.beta(3, 1, 5000), rng.beta(1, 8, 5000)), 1e-6, 1 - 1e-6
    )
    metrics = classification_metrics(labels, scores)

    assert {"roc_auc", "pr_auc", "recall", "precision", "f1", "brier", "base_rate"} <= set(metrics)
    assert metrics["base_rate"] == pytest.approx(labels.mean())
    assert metrics["pr_auc"] > metrics["base_rate"], "ranker must beat the base rate"


def test_a_perfect_ranker_scores_perfect_auc():
    """Sanity anchor: separable scores give ROC-AUC 1.0."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    assert classification_metrics(labels, scores)["roc_auc"] == pytest.approx(1.0)


def test_a_random_ranker_scores_chance_auc():
    """An uninformative score must not look skilful."""
    rng = np.random.default_rng(1)
    labels = (rng.random(20_000) < 0.2).astype(int)
    assert classification_metrics(labels, rng.random(20_000))["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_threshold_moves_recall_and_precision_in_opposite_directions():
    """The trade-off the conformal layer exists to make principled."""
    rng = np.random.default_rng(2)
    labels = (rng.random(4000) < 0.1).astype(int)
    scores = np.clip(
        np.where(labels == 1, rng.beta(2, 1, 4000), rng.beta(1, 5, 4000)), 1e-6, 1 - 1e-6
    )
    lenient = classification_metrics(labels, scores, threshold=0.2)
    strict = classification_metrics(labels, scores, threshold=0.8)

    assert lenient["recall"] > strict["recall"]
    assert strict["precision"] > lenient["precision"]


def test_brier_score_rewards_the_better_calibrated_of_two_rankers():
    """Brier is the metric that notices confidence, not just ordering."""
    labels = np.array([1, 1, 0, 0])
    confident_and_right = np.array([0.95, 0.9, 0.1, 0.05])
    hedged = np.array([0.55, 0.52, 0.48, 0.45])
    assert (
        classification_metrics(labels, confident_and_right)["brier"]
        < classification_metrics(labels, hedged)["brier"]
    )
