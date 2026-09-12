"""
ConfTest Standalone Colab Training & Artifact Exporter Script
Runs model training, temperature calibration fitting, and exports weights.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **scripts/train_model.py** (training plus scripts/calibrate_model.py for calibration).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "scripts/train_model.py"

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pickle
import json
import numpy as np
from src.benchmark.dataset_generator import BenchmarkDatasetGenerator
from src.models.lightgbm_model import TestFailureScorer
from src.models.calibration import TemperatureCalibrator, UncertaintyEstimator

def train_and_export(output_model_path="conftest_model.pkl", output_meta_path="conftest_metadata.json"):
    print("[ConfTest Training Pipeline] Generating multi-commit training data...")
    generator = BenchmarkDatasetGenerator(n_commits=600, n_tests=60, random_seed=42)
    train_df, cal_df, test_df = generator.generate()

    feature_cols = TestFailureScorer.FEATURE_NAMES
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_cal, y_cal = cal_df[feature_cols].values, cal_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    print(f"[ConfTest Training Pipeline] Training LightGBM on {len(X_train)} samples...")
    scorer = TestFailureScorer(n_estimators=150, learning_rate=0.04, max_depth=6)
    scorer.train(X_train, y_train, feature_cols)

    print("[ConfTest Training Pipeline] Fitting post-hoc Temperature Calibration...")
    cal_logits = scorer.predict_raw_logits(X_cal)
    calibrator = TemperatureCalibrator()
    optimal_temp = calibrator.fit(cal_logits, y_cal)
    print(f"[ConfTest Training Pipeline] Optimal Temperature: T = {optimal_temp:.4f}")

    # Evaluate on test set
    raw_probs = scorer.predict_proba(X_test)
    test_logits = scorer.predict_raw_logits(X_test)
    cal_probs = calibrator.predict_proba(test_logits)

    raw_ece = UncertaintyEstimator.compute_ece(raw_probs, y_test)
    cal_ece = UncertaintyEstimator.compute_ece(cal_probs, y_test)

    print(f"[ConfTest Training Pipeline] Test ECE: Raw = {raw_ece:.4f} -> Calibrated = {cal_ece:.4f}")

    # Save artifacts
    with open(output_model_path, "wb") as f:
        pickle.dump(scorer.model, f)

    meta = {
        "optimal_temperature": float(optimal_temp),
        "tau_abstain": 0.18,
        "theta_select": 0.08,
        "feature_names": feature_cols,
        "raw_ece": float(raw_ece),
        "calibrated_ece": float(cal_ece),
    }
    with open(output_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Successfully exported '{output_model_path}' & '{output_meta_path}'")

if __name__ == "__main__":
    train_and_export()
