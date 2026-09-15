"""
ConfTest Selective Policy Threshold Optimizer CLI.

Performs grid search on the validation split to determine optimal (tau_abstain, tau_conf) thresholds
that maximize regression test reduction while strictly preventing escaped failures.

Usage:
    python scripts/tune_policy.py --val data/splits/val.csv --test data/splits/test.csv --ensemble models/ensembles/5_seed_lgbm --calibrator models/calibrator.joblib
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.policy import SelectivePredictionPolicy, CostBenefitModel
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Policy Threshold Optimizer",
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
        "--calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Path to fitted calibrator artifact.",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="./models/policy_config.json",
        help="Destination path for optimized policy JSON configuration.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Target fast mode test selection budget.",
    )
    parser.add_argument(
        "--tau-abstain-grid",
        type=str,
        default="0.005,0.010,0.015,0.020,0.030,0.050",
        help=(
            "Comma-separated epistemic uncertainty cut-offs to sweep. The default is "
            "the grid the shipped policy was tuned on; pass a finer one to resolve a "
            "frontier the default is too coarse to see."
        ),
    )
    parser.add_argument(
        "--tau-conf-grid",
        type=str,
        default="0.10,0.30,0.50,0.60,0.70,0.80",
        help=(
            "Comma-separated minimum top-1 confidence cut-offs to sweep. Note that "
            "no calibrated confidence in this dataset reaches 0.30, so every value "
            "at or above that abstains on every commit and buys no reduction."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./reports/policy_tuning_report.json",
        help=(
            "Where the sweep and the chosen operating point are written. Give a "
            "second path when exploring an alternative --objective so the shipped "
            "report is not overwritten by a point that is not the shipped one."
        ),
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=("zero_escape", "recall_floor"),
        default="zero_escape",
        help=(
            "Which frontier point ships in --output-config. zero_escape maximises "
            "reduction among pairs that let no failure escape; recall_floor "
            "maximises reduction among pairs holding pooled recall at or above "
            "--recall-floor, which is what gate G5 asks for. Both are always "
            "measured and reported; this flag only decides which one is written."
        ),
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=95.0,
        help="Pooled failure recall percent the recall_floor objective must hold.",
    )
    return parser.parse_args()


def evaluate_thresholds_on_dataset(
    df: pd.DataFrame,
    ensemble: EnsembleUncertaintyPredictor,
    calibrator: ConfidenceCalibrator,
    tau_abstain: float,
    tau_conf: float,
    budget_ratio: float,
    return_per_commit: bool = False,
) -> Dict[str, Any]:
    """Run selective policy simulation across a dataset and count escaped failures and time reduction."""
    policy = SelectivePredictionPolicy(
        tau_abstain=tau_abstain,
        tau_conf=tau_conf,
        budget_ratio=budget_ratio,
    )

    total_tests_available = 0
    total_tests_executed = 0
    total_failures_available = 0
    total_failures_detected = 0
    total_abstentions = 0
    escaped_commits = 0
    commits_count = df["commit_sha"].nunique()
    # Off by default: the grid search stores every point it scores, and 36 points
    # times 183 commits would bury the report. The caller that wants an interval on
    # one chosen point asks for the units the bootstrap has to resample.
    per_commit: List[Dict[str, Any]] = []

    for sha, group in df.groupby("commit_sha"):
        X, y = prepare_feature_arrays(group)
        test_ids = list(group["test_id"].astype(str))

        preds = ensemble.predict_with_uncertainty(X)
        cal_probs = calibrator.calibrate(preds["mean_prob"])
        stds = preds["epistemic_std"]

        diff_files = int(group["diff_num_files_changed"].iloc[0]) if "diff_num_files_changed" in group.columns else 1
        diff_churn = int(group["diff_total_churn"].iloc[0]) if "diff_total_churn" in group.columns else 10

        decision = policy.evaluate_commit(
            commit_sha=sha,
            candidate_test_ids=test_ids,
            calibrated_confidences=cal_probs,
            epistemic_uncertainties=stds,
            num_changed_files=diff_files,
            total_churn_lines=diff_churn,
        )

        selected_set = set(decision.selected_test_ids)
        actual_failing_set = set(group[group["label_failed"] == 1]["test_id"].astype(str))

        detected = len(selected_set.intersection(actual_failing_set))
        missed = len(actual_failing_set) - detected

        total_tests_available += len(test_ids)
        total_tests_executed += len(decision.selected_test_ids)
        total_failures_available += len(actual_failing_set)
        total_failures_detected += detected

        if decision.abstained:
            total_abstentions += 1
        if missed > 0 and not decision.abstained:
            escaped_commits += 1

        if return_per_commit:
            per_commit.append(
                {
                    "commit_sha": str(sha),
                    "tests_available": len(test_ids),
                    "tests_executed": len(decision.selected_test_ids),
                    "failures_available": len(actual_failing_set),
                    "failures_detected": detected,
                    "abstained": bool(decision.abstained),
                    "escaped": bool(missed > 0 and not decision.abstained),
                }
            )

    # An empty denominator is an absent measurement, not a zero. A split with no
    # failing test has no recall to report, and reporting 0% would understate every
    # threshold pair that was never given anything to find.
    trr = (
        max(0.0, 1.0 - (total_tests_executed / total_tests_available)) * 100.0
        if total_tests_available > 0
        else float("nan")
    )
    recall = (
        (total_failures_detected / total_failures_available) * 100.0
        if total_failures_available > 0
        else float("nan")
    )
    abstention_rate = (
        (total_abstentions / commits_count) * 100.0 if commits_count > 0 else float("nan")
    )

    result = {
        "tau_abstain": tau_abstain,
        "tau_conf": tau_conf,
        "test_reduction_trr_pct": round(trr, 2),
        "failure_recall_pct": round(recall, 2),
        "abstention_rate_pct": round(abstention_rate, 2),
        "escaped_commits": escaped_commits,
        "missed_failures": total_failures_available - total_failures_detected,
        "commits": int(commits_count),
        "failures_available": int(total_failures_available),
        "tests_available": int(total_tests_available),
        "tests_executed": int(total_tests_executed),
    }
    if return_per_commit:
        result["per_commit"] = per_commit
    return result


def parse_grid(raw: str, flag: str) -> List[float]:
    """Read a comma-separated float grid off the command line."""
    try:
        values = [float(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag} must be comma-separated numbers, got {raw!r}.") from exc
    if not values:
        raise ValueError(f"{flag} is empty; a sweep needs at least one value.")
    return sorted(values)


def select_frontier_point(
    records: List[Dict[str, Any]],
    objective: str,
    recall_floor: float,
) -> Dict[str, Any]:
    """
    Pick one point off the measured grid.

    Two objectives, because the project asks two different questions of the same
    sweep. zero_escape is the shipping constraint: let no failure through, and
    take whatever reduction is left over. recall_floor is what gate G5 states:
    hold pooled failure recall at or above a floor and take the largest reduction
    that survives it. They are not the same point and one of them is usually much
    less conservative, so both are measured and reported and only the one named by
    --objective is written into the policy config.

    A pair whose recall or reduction is NaN is not eligible: that is a threshold
    pair the split could not measure, not a pair that scored zero.
    """
    if objective == "zero_escape":
        constraint = "escaped_commits == 0"
        eligible = [r for r in records if r["escaped_commits"] == 0]
    elif objective == "recall_floor":
        constraint = f"failure_recall_pct >= {recall_floor}"
        eligible = [
            r
            for r in records
            if not math.isnan(r["failure_recall_pct"])
            and r["failure_recall_pct"] >= recall_floor
        ]
    else:
        raise ValueError(f"Unknown objective {objective!r}.")

    eligible = [r for r in eligible if not math.isnan(r["test_reduction_trr_pct"])]

    if not eligible:
        # No invented default. Until 2026-09-03 this fell back to a silent
        # (0.015, 0.50) whenever the constraint admitted nothing, so an
        # unsatisfiable objective shipped a config that no run had ever scored.
        raise ValueError(
            f"No threshold pair on the {len(records)}-point grid satisfies "
            f"{constraint}. An unsatisfiable objective has no operating point; "
            "widen the grid, lower --recall-floor, or report that the gate is "
            "not met. It does not have a default."
        )

    # Largest reduction first. Ties go to the pair that lets fewer commits escape,
    # then to the lower thresholds, so the choice is deterministic across runs.
    best = max(
        eligible,
        key=lambda r: (
            r["test_reduction_trr_pct"],
            -r["escaped_commits"],
            -r["tau_abstain"],
            -r["tau_conf"],
        ),
    )
    return {
        "objective": objective,
        "constraint": constraint,
        "grid_points": len(records),
        "eligible_points": len(eligible),
        "selected": best,
    }


def main():
    args = parse_args()

    val_path = Path(args.val)
    test_path = Path(args.test)
    ens_path = Path(args.ensemble)
    cal_path = Path(args.calibrator)

    logger.info(f"Loading ensemble ({ens_path}) and calibrator ({cal_path})...")
    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ens_path))
    calibrator = ConfidenceCalibrator.load(str(cal_path))

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Grid Search Candidates
    tau_abstain_grid = parse_grid(args.tau_abstain_grid, "--tau-abstain-grid")
    tau_conf_grid = parse_grid(args.tau_conf_grid, "--tau-conf-grid")

    logger.info(f"Starting grid search over {len(tau_abstain_grid)*len(tau_conf_grid)} threshold pairs on validation set...")

    tuning_records = []

    for tau_a in tau_abstain_grid:
        for tau_c in tau_conf_grid:
            tuning_records.append(
                evaluate_thresholds_on_dataset(
                    df=val_df,
                    ensemble=ensemble,
                    calibrator=calibrator,
                    tau_abstain=tau_a,
                    tau_conf=tau_c,
                    budget_ratio=args.budget,
                )
            )

    # Score both frontiers on the same sweep. The reduction a zero-escape
    # constraint can buy and the reduction a 95%-recall floor can buy are
    # different numbers, and the report has to carry both -- G5 is stated in
    # terms of the floor, while the shipped policy is stated in terms of escapes.
    frontiers = {}
    for name in ("zero_escape", "recall_floor"):
        try:
            frontiers[name] = select_frontier_point(tuning_records, name, args.recall_floor)
        except ValueError as exc:
            frontiers[name] = {
                "objective": name,
                "grid_points": len(tuning_records),
                "eligible_points": 0,
                "selected": None,
                "unsatisfiable": str(exc),
            }
            logger.warning(f"Objective {name} is unsatisfiable on this grid: {exc}")

    chosen = frontiers[args.objective]
    if chosen["selected"] is None:
        raise ValueError(chosen["unsatisfiable"])

    val_eval = chosen["selected"]
    best_tau_a = float(val_eval["tau_abstain"])
    best_tau_c = float(val_eval["tau_conf"])
    best_trr = float(val_eval["test_reduction_trr_pct"])
    logger.info(f"\nOptimal Thresholds Found: tau_abstain = {best_tau_a:.4f}, tau_conf = {best_tau_c:.2f} (Val TRR: {best_trr:.1f}%)")

    # Evaluate optimal thresholds on unseen Test Split
    test_eval = evaluate_thresholds_on_dataset(
        df=test_df,
        ensemble=ensemble,
        calibrator=calibrator,
        tau_abstain=best_tau_a,
        tau_conf=best_tau_c,
        budget_ratio=args.budget,
    )

    policy = SelectivePredictionPolicy(
        tau_abstain=best_tau_a,
        tau_conf=best_tau_c,
        budget_ratio=args.budget,
        # Point at the report that measured this operating point, so a loaded
        # policy can say where its thresholds came from. The constructor default
        # labels every tuned run "untuned_constructor_defaults" -- which is how
        # the shipped config came to disclaim the sweep it was chosen from.
        source=str(Path(args.output_report).resolve()),
    )
    policy.save(args.output_config)

    report = {
        "objective": args.objective,
        "recall_floor_pct": args.recall_floor,
        "optimized_policy": {
            "tau_abstain": best_tau_a,
            "tau_conf": best_tau_c,
            "budget_ratio": args.budget,
        },
        # The validation row is the grid record that won, not a fresh evaluation of
        # the same thresholds. Re-running it invited the two to drift apart.
        "validation_evaluation": val_eval,
        "unseen_test_evaluation": test_eval,
        "frontiers": frontiers,
        # The whole sweep, so a reader can see the shape of the trade-off instead of
        # taking the winner on trust. Collected since the first version of this
        # script and, until 2026-09-03, thrown away at the end of the loop.
        "grid": tuning_records,
        "grid_definition": {
            "tau_abstain": tau_abstain_grid,
            "tau_conf": tau_conf_grid,
            "budget_ratio": args.budget,
        },
        "labels_measured": True,
        "produced_by": "python scripts/tune_policy.py",
    }

    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n=== Final Selective Policy Evaluation on Unseen Test Split ===")
    logger.info(f"Test Reduction Ratio (TRR):  {test_eval['test_reduction_trr_pct']}%")
    logger.info(f"Failure Recall (FR):         {test_eval['failure_recall_pct']}%")
    logger.info(f"Abstention Fallback Rate:   {test_eval['abstention_rate_pct']}%")
    logger.info(f"Escaped Commits:             {test_eval['escaped_commits']}")
    logger.info(f"Policy Saved to:             {args.output_config}")
    logger.info(f"Tuning Report:               {rep_path}")


if __name__ == "__main__":
    main()
