"""
Unit tests for 5-Seed Ensemble Uncertainty Estimation and Risk-Coverage Quantification.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.ensemble import EnsembleUncertaintyPredictor


@pytest.fixture
def sample_feature_matrix():
    rng = np.random.RandomState(42)
    X = rng.randn(60, len(FEATURE_NAMES)).astype(np.float32)
    y = (rng.rand(60) < 0.2).astype(int)
    y[0] = 1
    y[1] = 1
    return X, y


def test_ensemble_initialization_and_training(sample_feature_matrix):
    """Verify 5-seed ensemble training across distinct seeds."""
    X, y = sample_feature_matrix
    seeds = [42, 101, 2024]  # 3 seeds for fast unit test
    ensemble = EnsembleUncertaintyPredictor(seeds=seeds, n_estimators=10)
    summary = ensemble.train(X_train=X, y_train=y)

    assert summary["num_members"] == 3
    assert ensemble.ensemble_size == 3


def test_predict_with_uncertainty_bounds(sample_feature_matrix):
    """Verify epistemic std and predictive entropy bounds."""
    X, y = sample_feature_matrix
    ensemble = EnsembleUncertaintyPredictor(seeds=[42, 101], n_estimators=10)
    ensemble.train(X_train=X, y_train=y)

    res = ensemble.predict_with_uncertainty(X[:10])

    assert "mean_prob" in res
    assert "epistemic_std" in res
    assert "predictive_entropy" in res

    probs = res["mean_prob"]
    stds = res["epistemic_std"]
    entropy = res["predictive_entropy"]

    assert len(probs) == 10
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.all(stds >= 0.0)
    assert np.all(entropy >= 0.0) and np.all(entropy <= 1.0)


def test_commit_level_uncertainty_aggregation(sample_feature_matrix):
    """Verify commit-level uncertainty aggregation U(c)."""
    X, y = sample_feature_matrix
    ensemble = EnsembleUncertaintyPredictor(seeds=[42, 101], n_estimators=10)
    ensemble.train(X_train=X, y_train=y)

    u_commit = ensemble.compute_commit_level_uncertainty(X[:15])

    assert "max_epistemic_std" in u_commit
    assert "mean_epistemic_std" in u_commit
    assert "p95_epistemic_std" in u_commit
    assert u_commit["max_epistemic_std"] >= u_commit["mean_epistemic_std"]


def test_ensemble_save_and_load_roundtrip(sample_feature_matrix, tmp_path: Path):
    """Verify saving and loading ensemble reproduces identical predictions."""
    X, y = sample_feature_matrix
    ensemble = EnsembleUncertaintyPredictor(seeds=[42, 101], n_estimators=10)
    ensemble.train(X_train=X, y_train=y)
    original_res = ensemble.predict_with_uncertainty(X[:5])

    save_dir = tmp_path / "test_ensemble"
    ensemble.save_ensemble(str(save_dir))

    loaded = EnsembleUncertaintyPredictor.load_ensemble(str(save_dir))
    loaded_res = loaded.predict_with_uncertainty(X[:5])

    np.testing.assert_allclose(original_res["mean_prob"], loaded_res["mean_prob"], rtol=1e-5)
    np.testing.assert_allclose(original_res["epistemic_std"], loaded_res["epistemic_std"], rtol=1e-5)


def test_ensemble_propagates_training_controls_and_saves_portable_diagnostics(
    sample_feature_matrix, tmp_path: Path
):
    X, y = sample_feature_matrix
    ensemble = EnsembleUncertaintyPredictor(
        seeds=[42, 101],
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
        early_stopping_rounds=4,
        eval_metric="average_precision",
        use_class_weight=False,
    )

    summary = ensemble.train(X[:40], y[:40], X[40:], y[40:])
    save_dir = tmp_path / "portable"
    ensemble.save_ensemble(str(save_dir))
    metadata = json.loads((save_dir / "ensemble_metadata.json").read_text(encoding="utf-8"))

    assert summary["actual_num_trees"] == [
        member.model.booster_.num_trees() for member in ensemble.models
    ]
    assert all(item["eval_metric"] == "average_precision" for item in summary["member_diagnostics"])
    assert all(item["use_class_weight"] is False for item in summary["member_diagnostics"])
    assert metadata["member_files"] == ["member_1_seed_42.joblib", "member_2_seed_101.joblib"]
    assert all(not Path(item).is_absolute() for item in metadata["member_files"])
    assert metadata["member_diagnostics"] == summary["member_diagnostics"]
    assert metadata["bagging_active"] is True
