"""
Unit tests for LightGBM Model, Serialization, and Evaluation Metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from conftest.config import PROJECT_ROOT
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.trainer import (
    ModelTrainer,
    _portable_path,
    evaluate_predictions,
    prepare_feature_arrays,
)


def _synthetic_frame(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """A labelled feature frame with a realistic minority-class rate."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, len(FEATURE_NAMES)).astype(np.float32)
    y = (rng.rand(n_samples) < 0.15).astype(int)
    y[0] = 1
    y[1] = 1
    frame = pd.DataFrame({name: X[:, i] for i, name in enumerate(FEATURE_NAMES)})
    frame["label_failed"] = y
    return frame


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic tabular feature matrix for model testing."""
    rng = np.random.RandomState(42)
    n_samples = 100
    X = rng.randn(n_samples, len(FEATURE_NAMES)).astype(np.float32)
    # Binary labels with 15% positive failure rate
    y = (rng.rand(n_samples) < 0.15).astype(int)
    # Ensure at least 2 positive samples
    y[0] = 1
    y[1] = 1

    df_dict = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
    df_dict["label_failed"] = y
    df = pd.DataFrame(df_dict)
    return df, X, y


def test_lightgbm_training_and_prediction(synthetic_training_data):
    """Verify LightGBM model training, probability prediction bounds, and feature importances."""
    _, X, y = synthetic_training_data
    predictor = LightGBMTestPredictor(random_seed=42, n_estimators=20)
    diag = predictor.train(X_train=X, y_train=y)

    assert diag["n_features"] == 32
    assert diag["n_train_samples"] == 100

    probs = predictor.predict_proba(X)
    assert len(probs) == 100
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    importances = predictor.get_feature_importances()
    assert len(importances) == 32
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-2)


def test_model_serialization_roundtrip(synthetic_training_data, tmp_path: Path):
    """Verify model can be saved and loaded with identical prediction outputs."""
    _, X, y = synthetic_training_data
    predictor = LightGBMTestPredictor(random_seed=42, n_estimators=20)
    predictor.train(X_train=X, y_train=y)
    original_probs = predictor.predict_proba(X[:10])

    save_path = tmp_path / "test_model.joblib"
    predictor.save(str(save_path))
    assert save_path.exists()

    loaded = LightGBMTestPredictor.load(str(save_path))
    loaded_probs = loaded.predict_proba(X[:10])

    np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)


def test_evaluation_metrics_computation():
    """Verify scientific metric calculations for PR-AUC, ROC-AUC, F1, and Brier score."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.1])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 0.1

    # Regression guard. log_loss was previously called with `eps=1e-7`, a kwarg
    # scikit-learn removed in 1.5; the TypeError was swallowed by a bare
    # `except Exception` that substituted 0.0, so every published training
    # report claimed a perfect cross-entropy. Nothing here asserted on it, so
    # nothing caught it. A well-separated-but-not-saturated forecast has
    # strictly positive log loss, which pins the value to a real computation.
    assert metrics["log_loss"] is not None
    assert metrics["log_loss"] > 0.0
    assert metrics["log_loss"] < 0.3


def test_evaluation_metrics_saturated_probabilities():
    """Probabilities at exactly 0.0/1.0 must not yield inf log loss."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])

    metrics = evaluate_predictions(y_true, y_prob)

    # Internal clipping stands in for the removed `eps` kwarg: a perfect
    # forecast scores 0.0 rather than log(0) = -inf.
    assert metrics["log_loss"] == 0.0
    assert np.isfinite(metrics["log_loss"])


def test_evaluation_metrics_report_undefined_as_none():
    """A single-class y_true makes log loss undefined; it must be None, not 0.0."""
    y_true = np.zeros(10, dtype=int)
    y_prob = np.full(10, 0.3)

    metrics = evaluate_predictions(y_true, y_prob)

    # None serialises to JSON null. Reporting 0.0 here would advertise a
    # perfect cross-entropy for a set with no positives at all.
    assert metrics["log_loss"] is None
    assert metrics["positive_samples"] == 0
    # Documented degenerate-case conventions, guarded by n_pos rather than by
    # catching an exception.
    assert metrics["roc_auc"] == 0.5
    assert metrics["pr_auc"] == 0.0


def test_model_trainer_pipeline(synthetic_training_data, tmp_path: Path):
    """Verify ModelTrainer executes full training, validation, and exports reports."""
    df, _, _ = synthetic_training_data
    train_df = df.iloc[:70]
    val_df = df.iloc[70:85]
    test_df = df.iloc[85:]

    trainer = ModelTrainer(output_dir=str(tmp_path), model_version="test_v1")
    report = trainer.train_and_evaluate(train_df=train_df, val_df=val_df, test_df=test_df)

    assert "model_file" in report
    # tmp_path lies outside the project, so the path stays absolute here; the
    # in-project case is covered by the _portable_path tests below.
    assert Path(report["model_file"]).exists()
    assert report["produced_by"] == "python scripts/train_model.py"
    assert report["model_file_relative_to"] == "project_root"
    assert "test_metrics" in report
    assert "top_10_features_by_gain" in report
    assert len(report["top_10_features_by_gain"]) <= 10


# --------------------------------------------------------------------------
# Report portability: a path only identifies an artifact where it resolves
# --------------------------------------------------------------------------

def test_portable_path_is_relative_for_artifacts_inside_the_project():
    """An in-project artifact is recorded relative to the project root."""
    inside = PROJECT_ROOT / "models" / "ensembles" / "m_seed42.joblib"

    rendered = _portable_path(inside)

    assert rendered == "models/ensembles/m_seed42.joblib"
    assert not Path(rendered).is_absolute()
    # POSIX separators, so the report reads the same on either platform.
    assert "\\" not in rendered


def test_portable_path_keeps_absolute_paths_outside_the_project(tmp_path: Path):
    """
    An artifact genuinely elsewhere keeps its absolute path.

    Rewriting it relative to a root it does not live under would produce a
    path that silently resolves to the wrong file.
    """
    outside = tmp_path / "elsewhere.joblib"

    rendered = _portable_path(outside)

    assert Path(rendered).is_absolute()
    assert Path(rendered) == outside.resolve()


def test_training_report_records_a_resolvable_model_file(tmp_path, monkeypatch):
    """
    The field naming the artifact the metrics describe must resolve here.

    The repository carries a report whose model_file points into a
    `Desktop/main project/` checkout that does not exist here, so the metrics
    in it describe a file no reader can find.
    """
    monkeypatch.chdir(PROJECT_ROOT)
    trainer = ModelTrainer(output_dir=str(tmp_path), model_version="portable_v1")
    df = _synthetic_frame(120, seed=7)
    report = trainer.train_and_evaluate(train_df=df.iloc[:90], val_df=df.iloc[90:])

    recorded = Path(report["model_file"])
    resolved = recorded if recorded.is_absolute() else PROJECT_ROOT / recorded

    assert resolved.exists(), f"{report['model_file']} does not resolve to a file"
