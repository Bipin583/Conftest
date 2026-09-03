"""
ConfTest 8-Baseline RTS Comparison Experiment Runner.

Evaluates all 8 RTS baselines under identical test budget constraints
across temporal test splits and exports comparison tables and metrics.

Usage:
    python scripts/train_baseline.py --dataset data/splits/test.csv --budget 0.25 --output reports/baseline_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.evaluation.benchmark import BaselineBenchmarkRunner, MODEL_SCORE_COLUMNS
from conftest.evaluation.headline import MissingArtifact
from conftest.features.pipeline import FEATURE_NAMES
from conftest.logging_config import get_logger
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.ensemble import EnsembleUncertaintyPredictor

logger = get_logger(__name__)

# The strategy every difference is measured against: the one the report is about.
DEFAULT_REFERENCE = "8. ConfTest (Calibrated + Selective Abstention)"

ENSEMBLE_PRODUCED_BY = "python scripts/train_ensemble.py"
CALIBRATOR_PRODUCED_BY = "python scripts/calibrate_model.py"


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise MissingArtifact(path, produced_by)
    return path


def attach_model_scores(df, ensemble_dir: Path, calibrator_path: Path):
    """
    Score every row with the shipped ensemble and calibrator.

    Baselines 6, 7 and 8 rank by model output. The benchmark used to synthesise that
    output when the columns were absent, including an `uncertainty` derived from the
    commit's own failure labels, so the published comparison measured a hand-written
    formula rather than this system. The columns now come from the artifacts.

    Returns:
        (df with MODEL_SCORE_COLUMNS attached, provenance dict)
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} feature columns the ensemble was "
            f"trained on, first few: {missing[:5]}"
        )

    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ensemble_dir))
    calibrator = ConfidenceCalibrator.load(str(calibrator_path))

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    out = ensemble.predict_with_uncertainty(X)
    raw = np.asarray(out["mean_prob"], dtype=float)

    df = df.copy()
    df["raw_score"] = raw
    df["calibrated_confidence"] = np.asarray(calibrator.calibrate(raw), dtype=float)
    # Model disagreement, which is what the abstention rule is about: a test the five
    # members rank differently is a test this system says it cannot judge.
    df["uncertainty"] = np.asarray(out["epistemic_std"], dtype=float)

    provenance = {
        "ensemble": ensemble_dir.as_posix(),
        "ensemble_members": int(ensemble.ensemble_size),
        "calibrator": calibrator_path.as_posix(),
        "calibrator_method": getattr(calibrator, "method", type(calibrator).__name__),
        "scored_rows": int(len(df)),
        "raw_score_is": "ensemble mean failure probability",
        "calibrated_confidence_is": "raw_score after the fitted calibrator",
        "uncertainty_is": "across-member standard deviation (epistemic)",
        "raw_score_range": [round(float(raw.min()), 6), round(float(raw.max()), 6)],
        "uncertainty_range": [
            round(float(df["uncertainty"].min()), 6),
            round(float(df["uncertainty"].max()), 6),
        ],
    }
    return df, provenance


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Baseline Comparison Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/splits/test.csv",
        help="Path to evaluation test split dataset CSV.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Test budget fraction (e.g. 0.25 = top 25%% of tests).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/baseline_comparison.csv",
        help="Destination path for benchmark results CSV.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
        help="Bootstrap resamples for the confidence intervals. 0 skips them.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
        help="Strategy that paired differences are taken against.",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Ensemble directory whose mean probability ranks baselines 6-8.",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Fitted calibrator chosen on the validation split.",
    )
    parser.add_argument(
        "--per-commit-output",
        type=str,
        default="./reports/baseline_per_commit.csv",
        help=(
            "Destination for the unpooled per-commit statistics. This is the "
            "artifact scripts/run_statistical_tests.py reads; without it there "
            "are no paired observations to test."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.dataset)

    # Fallback to processed features if split test.csv not present
    if not data_path.exists():
        fallback = Path("./data/processed/features.csv")
        if fallback.exists():
            logger.info(f"Test split {data_path} not found. Falling back to {fallback}...")
            data_path = fallback
        else:
            logger.error(f"Dataset file not found: {data_path}. Run extract_features.py or build_splits.py first.")
            sys.exit(1)

    logger.info(f"Loading benchmark dataset from {data_path}...")
    df = pd.read_csv(data_path)

    ensemble_dir = _require(Path(args.ensemble), ENSEMBLE_PRODUCED_BY)
    calibrator_path = _require(Path(args.calibrator), CALIBRATOR_PRODUCED_BY)
    logger.info(
        f"Scoring {len(df)} rows with {ensemble_dir} and {calibrator_path} "
        "(baselines 6-8 rank by model output; the benchmark refuses to invent it)..."
    )
    df, score_provenance = attach_model_scores(df, ensemble_dir, calibrator_path)
    logger.info(
        f"  raw_score in {score_provenance['raw_score_range']}, "
        f"epistemic std in {score_provenance['uncertainty_range']}, "
        f"calibrator '{score_provenance['calibrator_method']}'"
    )
    assert all(c in df.columns for c in MODEL_SCORE_COLUMNS)

    runner = BaselineBenchmarkRunner(budget_ratio=args.budget)
    # One sweep, folded three ways: the pooled table, the bootstrap intervals, and
    # the per-commit export. Running the selectors once per output would be three
    # times the work and, worse, three chances for the tables to disagree.
    per_commit = runner.accumulate_per_commit(df)
    results_df = runner.evaluate_dataset(df, per_commit=per_commit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info(f"Benchmark results exported to: {out_path}")

    per_commit_path = Path(args.per_commit_output)
    per_commit_path.parent.mkdir(parents=True, exist_ok=True)
    per_commit_df = runner.per_commit_frame(per_commit)
    per_commit_df.to_csv(per_commit_path, index=False)
    reference_rows = per_commit_df[per_commit_df["strategy"] == args.reference]
    n_with_failures = int((reference_rows["failures_available"] > 0).sum())
    logger.info(
        f"Per-commit statistics exported to: {per_commit_path} "
        f"({len(per_commit_df)} rows; {n_with_failures} of "
        f"{df['commit_sha'].nunique()} commits have a failure available and so "
        f"carry a defined recall)"
    )
    logger.info(
        f"\n=== RTS Baseline Comparison Table (Budget: {args.budget*100:.0f}%) ===\n"
        + results_df.to_string(index=False)
    )

    if args.bootstraps < 1:
        logger.warning(
            "--bootstraps 0: reporting point estimates with no intervals. At a few "
            "hundred commits the gap between two strategies can be smaller than the "
            "noise in either one, so these numbers should not be quoted on their own."
        )
        return

    intervals = runner.evaluate_dataset_with_intervals(
        df,
        num_bootstraps=args.bootstraps,
        reference=args.reference,
        per_commit=per_commit,
    )
    interval_df = runner.interval_table(intervals)

    interval_csv = out_path.with_name(out_path.stem + "_intervals.csv")
    interval_json = out_path.with_name(out_path.stem + "_intervals.json")
    interval_df.to_csv(interval_csv, index=False)
    # The scores the model strategies ranked by travel with the table, so a reader can
    # tell whether row 8 is this system or something that merely shares its name.
    intervals["model_scores"] = score_provenance
    intervals["labels_measured"] = True
    interval_json.write_text(json.dumps(intervals, indent=2, default=str))

    logger.info(f"Confidence intervals exported to: {interval_csv} and {interval_json}")
    logger.info(
        f"\n=== Same table with {intervals['confidence_level']*100:.0f}% bootstrap CIs "
        f"({intervals['num_bootstraps']} resamples over {intervals['num_commits']} commits, "
        f"unit = {intervals['resampling_unit']}) ===\n"
        + interval_df.to_string(index=False)
        + "\n\n* the paired difference in failure recall excludes zero."
    )


if __name__ == "__main__":
    main()
