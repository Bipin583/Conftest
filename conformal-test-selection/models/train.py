"""Layer 1: gradient-boosted failure-risk model.

Trains an XGBoost classifier to estimate P(test fails | change, test history,
change-test interaction) from the preprocessed splits.

The corpus is heavily imbalanced -- roughly one failing row in twenty -- so the
defaults matter:

* ``scale_pos_weight`` is derived from the training split so the positive class
  is not drowned out.
* Early stopping watches the validation split, which sits strictly in the
  future of the training split, so the stopping point is chosen under the same
  temporal shift the model faces at inference.
* **PR-AUC is the headline metric, not ROC-AUC.** At a 5% positive rate, a
  model that ranks nothing usefully still posts a respectable ROC-AUC; average
  precision does not flatter it.

Raw XGBoost output is *not* a calibrated probability. :mod:`models.calibrate`
fixes that, and :mod:`models.conformal` turns the calibrated score into a
selection rule with a coverage guarantee. Nothing downstream should threshold
this model's output directly.

Example:
    >>> from models.train import train
    >>> report = train()
    >>> report["test"]["pr_auc"]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, load_config, resolve_path, save_artifact, set_seed  # noqa: E402
from data.features import LABEL_COLUMN  # noqa: E402

LOGGER = get_logger(__name__)

#: Columns carried through preprocessing for traceability, never model inputs.
NON_FEATURE_COLUMNS = {
    LABEL_COLUMN,
    "repo",
    "commit_sha",
    "test_id",
    "timestamp",
    "mutant_index",
}


class TrainingError(RuntimeError):
    """Raised when splits are missing, empty, or single-class."""


def load_split(
    name: str,
    processed_dir: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load one preprocessed split.

    Args:
        name: ``"train"``, ``"val"`` or ``"test"``.
        processed_dir: Directory holding the split CSVs. Defaults to
            ``data.processed_dir``.

    Returns:
        A ``(features, labels, identifiers)`` triple. ``identifiers`` holds the
        carried columns such as ``commit_sha`` and ``test_id``.

    Raises:
        TrainingError: If the split file is missing or empty.
    """
    cfg = load_config()
    root = resolve_path(processed_dir or cfg["data"]["processed_dir"])
    source = root / f"{name}.csv"
    if not source.is_file():
        raise TrainingError(
            f"Split not found: {source}. Run 'python cli.py preprocess' first."
        )

    frame = pd.read_csv(source, low_memory=False)
    if frame.empty:
        raise TrainingError(f"Split {source} is empty.")
    if LABEL_COLUMN not in frame:
        raise TrainingError(f"Split {source} lacks the '{LABEL_COLUMN}' column.")

    feature_columns = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]
    identifiers = frame.loc[:, [c for c in frame.columns if c in NON_FEATURE_COLUMNS and c != LABEL_COLUMN]]
    features = frame.loc[:, feature_columns].astype(np.float32)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)

    LOGGER.info(
        "Loaded %-5s: %d rows x %d features, %d failing (%.3f%%).",
        name, len(features), features.shape[1], int(labels.sum()), 100.0 * labels.mean(),
    )
    return features, labels, identifiers


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the standard binary-classification metric set.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        y_score: Predicted probability of the positive class.
        threshold: Cut-off applied to derive hard predictions.

    Returns:
        A dictionary of metrics. Ranking metrics are ``float('nan')`` when only
        one class is present.
    """
    y_pred = (y_score >= threshold).astype(np.int8)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "base_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) < 2:
        metrics.update({"roc_auc": float("nan"), "pr_auc": float("nan"), "log_loss": float("nan"), "brier": float("nan")})
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    metrics["log_loss"] = float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)))
    metrics["brier"] = float(brier_score_loss(y_true, y_score))
    return metrics


def _resolve_scale_pos_weight(setting: Any, labels: np.ndarray) -> Optional[float]:
    """Interpret the ``model.scale_pos_weight`` configuration value.

    Args:
        setting: ``"auto"``, a number, or ``None``.
        labels: Training labels, used for the ``"auto"`` derivation.

    Returns:
        The weight to pass to XGBoost, or ``None`` to leave it unset.
    """
    if setting is None:
        return None
    if isinstance(setting, str):
        if setting.lower() != "auto":
            raise TrainingError(f"model.scale_pos_weight must be 'auto', a number, or null; got {setting!r}.")
        positives = int(labels.sum())
        if positives == 0:
            raise TrainingError("Training split contains no failing rows.")
        weight = float((len(labels) - positives) / positives)
        LOGGER.info("scale_pos_weight=auto resolved to %.3f (%d neg / %d pos).", weight, len(labels) - positives, positives)
        return weight
    return float(setting)


def build_model(config: Optional[Dict[str, Any]] = None, scale_pos_weight: Optional[float] = None) -> Any:
    """Construct the XGBoost classifier from configuration.

    Args:
        config: The ``model`` config block. Defaults to ``config.yaml``.
        scale_pos_weight: Resolved imbalance weight, or ``None``.

    Returns:
        An unfitted ``xgboost.XGBClassifier``.

    Raises:
        TrainingError: If XGBoost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TrainingError("XGBoost is required: pip install xgboost==1.7.5") from exc

    cfg = config if config is not None else load_config()["model"]
    params: Dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "max_depth": int(cfg.get("max_depth", 8)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "eval_metric": cfg.get("eval_metric", "logloss"),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "random_state": int(load_config().get("project", {}).get("random_seed", 42)),
        "tree_method": "hist",
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    early = cfg.get("early_stopping_rounds")
    if early:
        # Constructor placement works on XGBoost 1.6+ and is mandatory from
        # 2.0, where the fit() keyword was removed.
        params["early_stopping_rounds"] = int(early)

    return XGBClassifier(**params)


def train(
    processed_dir: Optional[Any] = None,
    model_path: Optional[Any] = None,
    report_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train the classifier, evaluate on all splits, and persist the model.

    Args:
        processed_dir: Directory holding the split CSVs.
        model_path: Destination for the fitted model.
        report_path: Destination for the JSON metric report.

    Returns:
        The metric report, also written to ``reports/training_report.json``.

    Raises:
        TrainingError: If a split is unusable.
    """
    cfg = load_config()
    seed = set_seed()

    X_train, y_train, _ = load_split("train", processed_dir)
    X_val, y_val, _ = load_split("val", processed_dir)
    X_test, y_test, _ = load_split("test", processed_dir)

    if len(np.unique(y_train)) < 2:
        raise TrainingError("Training split contains a single class; cannot fit a classifier.")

    weight = _resolve_scale_pos_weight(cfg["model"].get("scale_pos_weight"), y_train)
    model = build_model(cfg["model"], scale_pos_weight=weight)

    LOGGER.info("Fitting XGBoost on %d rows x %d features...", len(X_train), X_train.shape[1])
    started = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.perf_counter() - started

    best_iteration = getattr(model, "best_iteration", None)
    LOGGER.info(
        "Fit complete in %.1fs (best_iteration=%s of %d).",
        elapsed, best_iteration, cfg["model"].get("n_estimators", 500),
    )

    report: Dict[str, Any] = {
        "seed": seed,
        "fit_seconds": round(elapsed, 2),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "n_features": int(X_train.shape[1]),
        "scale_pos_weight": weight,
        "params": {k: v for k, v in model.get_params().items() if v is not None and k != "missing"},
    }

    for name, features, labels in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        scores = model.predict_proba(features)[:, 1]
        report[name] = classification_metrics(labels, scores)
        LOGGER.info(
            "%-5s | PR-AUC %.4f | ROC-AUC %.4f | recall@0.5 %.4f | precision@0.5 %.4f",
            name,
            report[name]["pr_auc"],
            report[name]["roc_auc"],
            report[name]["recall"],
            report[name]["precision"],
        )

    report["feature_importance"] = _feature_importance(model, list(X_train.columns))

    destination = save_artifact(
        {"model": model, "feature_columns": list(X_train.columns), "report": report},
        model_path or cfg["artifacts"]["model_path"],
    )
    LOGGER.info("Saved model to %s", destination)

    target = resolve_path(report_path or Path(cfg["artifacts"]["reports_dir"]) / "training_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", target)
    return report


def _feature_importance(model: Any, columns: Sequence[str], top_k: int = 20) -> List[Dict[str, Any]]:
    """Extract the top-k gain-ranked features.

    Args:
        model: A fitted XGBoost classifier.
        columns: Feature names in training order.
        top_k: How many entries to return.

    Returns:
        A list of ``{"feature", "importance"}`` dictionaries, highest first.
    """
    try:
        scores = np.asarray(model.feature_importances_, dtype=float)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Could not read feature importances: %s", exc)
        return []

    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"feature": str(columns[i]), "importance": round(float(scores[i]), 6)}
        for i in order
        if scores[i] > 0
    ]


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m models.train``.

    Returns:
        ``0`` on success, ``1`` on a handled training error.
    """
    parser = argparse.ArgumentParser(description="Train the failure-risk model.")
    parser.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    parser.add_argument("--model-path", default=None, help="Destination model path.")
    parser.add_argument("--report-path", default=None, help="Destination report path.")
    args = parser.parse_args(argv)

    try:
        train(
            processed_dir=args.processed_dir,
            model_path=args.model_path,
            report_path=args.report_path,
        )
    except TrainingError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
