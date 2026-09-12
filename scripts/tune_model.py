"""Select LightGBM capacity using the validation split only.

The held-out test split is deliberately absent from this command. Candidate models
fit on train, stop on validation average precision, and are ranked by validation
PR-AUC with deterministic tie-breaking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.logging_config import get_logger
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.trainer import prepare_feature_arrays

logger = get_logger(__name__)

DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "use_class_weight": True, "early_stopping_rounds": 25},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "use_class_weight": True, "early_stopping_rounds": 40},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "use_class_weight": True, "early_stopping_rounds": 25},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "num_leaves": 63,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "use_class_weight": True, "early_stopping_rounds": 25},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
     "subsample": 1.0, "subsample_freq": 0, "colsample_bytree": 1.0,
     "use_class_weight": True, "early_stopping_rounds": 25},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "use_class_weight": False, "early_stopping_rounds": 25},
]

def _candidate_key(record: dict[str, Any]) -> tuple:
    """Rank quality first, then prefer simpler and lexically stable configs."""
    config = record["config"]
    return (
        record["validation_metrics"]["pr_auc"],
        record["validation_metrics"]["roc_auc"],
        -record["validation_metrics"]["brier_score"],
        -record["training_diagnostics"]["actual_num_trees"],
        -config["max_depth"],
        -config["num_leaves"],
        json.dumps(config, sort_keys=True),
    )


def tune_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidates: Iterable[dict[str, Any]],
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit and rank candidates without accepting or inspecting held-out data."""
    records: list[dict[str, Any]] = []
    for candidate_index, raw_config in enumerate(candidates):
        config = dict(raw_config)
        training_controls = {
            "early_stopping_rounds": int(config.pop("early_stopping_rounds", 25)),
            "use_class_weight": bool(config.pop("use_class_weight", True)),
        }
        predictor = LightGBMTestPredictor(random_seed=random_seed, **config)
        diagnostics = predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            eval_metric="average_precision",
            **training_controls,
        )
        probabilities = predictor.predict_proba(X_val)
        metrics = {
            "pr_auc": float(average_precision_score(y_val, probabilities)),
            "roc_auc": float(roc_auc_score(y_val, probabilities)),
            "brier_score": float(brier_score_loss(y_val, probabilities)),
        }
        full_config = {**config, **training_controls}
        records.append({
            "candidate_index": candidate_index,
            "config": full_config,
            "validation_metrics": metrics,
            "training_diagnostics": diagnostics,
        })

    if not records:
        raise ValueError("At least one model candidate is required.")
    selected = max(records, key=_candidate_key)
    return {
        "objective": "validation_pr_auc",
        "selection_split": "validation",
        "held_out_test_used_for_selection": False,
        "random_seed": random_seed,
        "candidate_count": len(records),
        "selected_candidate_index": selected["candidate_index"],
        "selected_config": selected["config"],
        "selected_validation_metrics": selected["validation_metrics"],
        "candidates": records,
        "labels_measured": True,
        "produced_by": "python scripts/tune_model.py",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LightGBM on train/validation only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", default="./data/splits/train.csv")
    parser.add_argument("--val", default="./data/splits/val.csv")
    parser.add_argument("--output", default="./reports/model_tuning_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidates-json",
        help="Optional JSON file containing a list of candidate configuration objects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    for split_name, split_path in (("train", train_path), ("validation", val_path)):
        if not split_path.exists():
            raise FileNotFoundError(f"{split_name} split not found: {split_path}")

    candidates = DEFAULT_CANDIDATES
    if args.candidates_json:
        candidate_path = Path(args.candidates_json)
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            raise ValueError("--candidates-json must contain a JSON list.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    X_train, y_train = prepare_feature_arrays(train_df)
    X_val, y_val = prepare_feature_arrays(val_df)
    report = tune_candidates(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        candidates=candidates,
        random_seed=args.seed,
    )
    report["inputs"] = {
        "train": train_path.as_posix(),
        "validation": val_path.as_posix(),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Selected candidate %s with validation PR-AUC %.6f; report: %s",
        report["selected_candidate_index"],
        report["selected_validation_metrics"]["pr_auc"],
        output_path,
    )


if __name__ == "__main__":
    main()
