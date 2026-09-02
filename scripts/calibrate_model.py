"""
ConfTest Post-Hoc Confidence Calibration CLI.

Fits Isotonic Regression and Temperature Scaling calibrators on the validation split,
evaluates Expected Calibration Error (ECE) and Brier Score reductions on the test split,
and exports calibrated model artifacts and reliability diagram data.

Usage:
    python scripts/calibrate_model.py --val data/splits/val.csv --test data/splits/test.csv --ensemble models/ensembles/5_seed_lgbm
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.calibration import ConfidenceCalibrator, compute_ece
from conftest.models.calibrator_selection import (
    score_calibrators,
    select_calibrator,
    split_clusters_for_selection,
    split_for_selection,
)
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# The dataset builder sets commit_sha to the mutant id, so this column is the
# cluster: all the rows sharing a value saw one injected fault and are therefore
# not independent draws.
CLUSTER_COLUMN = "commit_sha"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Model Calibration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val",
        type=str,
        default="./data/splits/val.csv",
        help="Path to validation dataset CSV.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/splits/test.csv",
        help="Path to test dataset CSV.",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Directory containing trained ensemble checkpoints.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
        help="Cluster-bootstrap resamples behind every calibration interval.",
    )
    parser.add_argument(
        "--output-calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Path to save fitted calibrator artifact.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./reports/calibration_report.json",
        help="Path to export calibration diagnostics report.",
    )
    return parser.parse_args()


def score_record(score):
    """
    One calibration candidate as it appears in the report.

    The paired differences are included whenever they exist, because the point
    estimates on their own do not say whether a method helped -- and a reader who
    only ever sees three ECE values will assume the smallest one won something.
    """
    record = {
        "method": score.method,
        "ece": round(score.ece, 4),
        "mce": round(score.mce, 4),
        "brier_score": round(score.brier, 4),
    }
    differences = {
        "ece": score.ece_vs_baseline,
        "mce": score.mce_vs_baseline,
        "brier_score": score.brier_vs_baseline,
    }
    for metric, diff in differences.items():
        if diff is None:
            continue
        record[f"{metric}_vs_uncalibrated"] = {
            "point": round(diff["point"], 5),
            "ci_lower": round(diff["ci_lower"], 5),
            "ci_upper": round(diff["ci_upper"], 5),
            "excludes_zero": bool(diff["excludes_zero"]),
        }
    return record


def cluster_labels(df, split_name: str):
    """
    The mutant identifier for every row, or None when the split does not carry one.

    Without it there is no honest resampling unit: the rows of one mutant share an
    injected fault, so treating them as independent would quote intervals several
    times too narrow, and cutting the selection split between them leaks the
    mutant across both halves. Returning None is therefore a real degradation, and
    it is warned about here and recorded in the report rather than passed over.
    """
    if CLUSTER_COLUMN not in df.columns:
        logger.warning(
            f"{split_name} split has no '{CLUSTER_COLUMN}' column, so calibration "
            f"metrics get no intervals and the method is chosen against fixed "
            f"tolerances instead of measured noise. Rebuild the splits from "
            f"scripts/build_real_dataset.py, which records the mutant id."
        )
        return None
    return df[CLUSTER_COLUMN].astype(str).tolist()


def main():
    args = parse_args()

    val_path = Path(args.val)
    test_path = Path(args.test)
    ens_path = Path(args.ensemble)

    if not val_path.exists() or not test_path.exists() or not ens_path.exists():
        logger.error("Required dataset splits or ensemble directory missing.")
        sys.exit(1)

    logger.info(f"Loading ensemble from {ens_path}...")
    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ens_path))

    logger.info(f"Loading validation split from {val_path}...")
    val_df = pd.read_csv(val_path)
    X_val, y_val = prepare_feature_arrays(val_df)

    logger.info(f"Loading test split from {test_path}...")
    test_df = pd.read_csv(test_path)
    X_test, y_test = prepare_feature_arrays(test_df)

    # 1. Uncalibrated predictions on Validation and Test
    val_raw_probs = ensemble.predict_with_uncertainty(X_val)["mean_prob"]
    test_raw_probs = ensemble.predict_with_uncertainty(X_test)["mean_prob"]

    # 2. Choose the method inside the validation split.
    #
    # The calibrators are fitted on one half of validation and compared on the
    # other. Comparing them on data they were fitted to would hand the win to
    # whichever method is most flexible -- isotonic regression can drive training
    # ECE to nearly zero by memorising -- and comparing them on TEST, which is
    # what this script used to do, makes the test split part of model selection
    # and biases every number reported from it.
    val_clusters = cluster_labels(val_df, "validation")
    if val_clusters is not None:
        fit_mask = split_clusters_for_selection(np.asarray(val_clusters))
    else:
        # No cluster labels: fall back to the positional cut, which splits some
        # mutants across both halves. Recorded in the report as basis "point".
        fit_mask = np.zeros(len(val_raw_probs), dtype=bool)
        fit_mask[: split_for_selection(len(val_raw_probs))] = True

    sel_probs_fit, sel_probs_hold = val_raw_probs[fit_mask], val_raw_probs[~fit_mask]
    sel_y_fit, sel_y_hold = y_val[fit_mask], y_val[~fit_mask]
    hold_clusters = (
        list(np.asarray(val_clusters)[~fit_mask]) if val_clusters is not None else None
    )
    logger.info(
        f"Selecting calibration method on validation: {int(fit_mask.sum())} rows to "
        f"fit, {int((~fit_mask).sum())} held out to compare"
        + (
            f" ({len(set(hold_clusters))} mutants, none shared with the fit half)."
            if hold_clusters is not None
            else ", split by row index (no cluster column)."
        )
    )

    candidates = {}
    for method in ("isotonic", "temperature_scaling"):
        probe = ConfidenceCalibrator(method=method).fit(sel_probs_fit, sel_y_fit)
        candidates[method] = probe.calibrate(sel_probs_hold)

    sel_scores = score_calibrators(
        sel_y_hold,
        {"uncalibrated": sel_probs_hold, **candidates},
        cluster_ids=hold_clusters,
        num_bootstraps=args.bootstraps,
    )

    outcome = select_calibrator(sel_scores)
    logger.info(
        f"  decision basis: {outcome.basis}"
        + (
            " (paired cluster bootstrap over mutants)"
            if outcome.basis == "bootstrap"
            else " (fixed tolerances; no cluster column to bootstrap over)"
        )
    )
    for method, why in sorted(outcome.disqualified.items()):
        logger.warning(f"  disqualified {method}: {why}")
    logger.info(f"  chose {outcome.method}: {outcome.reason}")

    # 3. Refit the chosen method on the FULL validation split, then apply to test.
    logger.info("Fitting Isotonic and Temperature Scaling calibrators on validation split...")
    iso_cal = ConfidenceCalibrator(method="isotonic").fit(val_raw_probs, y_val)
    temp_cal = ConfidenceCalibrator(method="temperature_scaling").fit(val_raw_probs, y_val)

    test_iso_probs = iso_cal.calibrate(test_raw_probs)
    test_temp_probs = temp_cal.calibrate(test_raw_probs)

    # 4. Compute Calibration Metrics on Test Split
    _, _, raw_bins = compute_ece(y_test, test_raw_probs, n_bins=10)
    _, _, iso_bins = compute_ece(y_test, test_iso_probs, n_bins=10)
    _, _, temp_bins = compute_ece(y_test, test_temp_probs, n_bins=10)

    # Reported with intervals, and with the difference against the uncalibrated
    # model paired over resampled mutants. "ECE fell from 0.041 to 0.038" is not a
    # result at this sample size unless the interval on that fall clears zero.
    test_clusters = cluster_labels(test_df, "test")
    test_scores = {
        sc.method: sc
        for sc in score_calibrators(
            y_test,
            {
                "uncalibrated": test_raw_probs,
                "isotonic": test_iso_probs,
                "temperature_scaling": test_temp_probs,
            },
            cluster_ids=test_clusters,
            num_bootstraps=args.bootstraps,
        )
    }
    # Only ECE is read out here, for the reduction percentages the report has
    # always carried; every other number reaches the report through score_record,
    # which keeps the point estimate and its interval together.
    raw_ece = test_scores["uncalibrated"].ece
    iso_ece = test_scores["isotonic"].ece
    temp_ece = test_scores["temperature_scaling"].ece

    # The method was already chosen on validation. Test metrics below are
    # reported, never used to choose -- that is the whole point of step 2.
    best_method = outcome.method
    best_cal = {"isotonic": iso_cal, "temperature_scaling": temp_cal}.get(best_method)

    cal_path = Path(args.output_calibrator)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    if best_cal is not None:
        best_cal.save(str(cal_path))
    else:
        # Declining to calibrate is a real outcome. Leave no stale artifact behind
        # that a later stage would silently load as if a method had been chosen.
        if cal_path.exists():
            cal_path.unlink()
        logger.warning(
            "No calibrator saved: no method improved validation ECE without "
            "degrading worst-case calibration. Downstream stages must treat the "
            "model as uncalibrated."
        )

    report = {
        "best_method": best_method,
        "selection": {
            "chosen_on": "validation holdout",
            "reason": outcome.reason,
            "disqualified": outcome.disqualified,
            "basis": outcome.basis,
            "resampling_unit": "mutant" if val_clusters is not None else "none",
            "num_bootstraps": args.bootstraps,
            "validation_scores": [score_record(sc) for sc in sel_scores],
        },
        "test_metrics": {
            "resampling_unit": "mutant" if test_clusters is not None else "none",
            "num_bootstraps": args.bootstraps,
            "uncalibrated": score_record(test_scores["uncalibrated"]),
            "isotonic_calibration": {
                **score_record(test_scores["isotonic"]),
                "ece_reduction_pct": round(((raw_ece - iso_ece) / max(1e-5, raw_ece)) * 100, 2),
            },
            "temperature_scaling": {
                **score_record(test_scores["temperature_scaling"]),
                "ece_reduction_pct": round(((raw_ece - temp_ece) / max(1e-5, raw_ece)) * 100, 2),
            },
        },
        "reliability_diagram_bins": {
            "uncalibrated": raw_bins,
            "isotonic": iso_bins,
            "temperature_scaling": temp_bins,
        },
    }

    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n=== Confidence Calibration Results on Unseen Test Split ===")
    if test_clusters is not None:
        logger.info(
            f"  intervals: 95% cluster bootstrap, {args.bootstraps} resamples over "
            f"{len(set(test_clusters))} mutants; * marks a difference against the "
            f"uncalibrated model whose interval excludes zero"
        )
    for label, key in (
        ("Uncalibrated Model:   ", "uncalibrated"),
        ("Isotonic Calibration: ", "isotonic"),
        ("Temperature Scaling:  ", "temperature_scaling"),
    ):
        sc = test_scores[key]
        line = f"{label} ECE = {sc.ece:.4f}, MCE = {sc.mce:.4f}, Brier = {sc.brier:.4f}"
        if sc.ece_vs_baseline is not None:
            d = sc.ece_vs_baseline
            line += (
                f" | ECE vs uncalibrated {d['point']:+.4f} "
                f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
                + (" *" if d["excludes_zero"] else "")
            )
        logger.info(line)
    logger.info(f"Selected Best Calibrator: '{best_method}' (chosen on validation holdout)")
    if best_cal is not None:
        logger.info(f"  saved to: {cal_path}")
    logger.info(f"Calibration Report: {rep_path}")


if __name__ == "__main__":
    main()
