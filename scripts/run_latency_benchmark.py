"""
ConfTest Latency Benchmark CLI.

Times the scoring path on the artifacts the study ships -- the 5-seed ensemble under
`models/ensembles/5_seed_lgbm`, the fitted calibrator, and the tuned policy config --
over real feature rows from the held-out test split.

The figure covers scoring only. Feature mining runs once per commit upstream, needs a
repository checkout, and is not in the measurement; the report says so in its own fields
rather than leaving it to be inferred.

Every input is loaded, never generated. A missing artifact raises MissingArtifact with
the command that produces it, because a latency number for a model that was trained on
the spot to make the benchmark run is not a latency number for this system.

Usage:
    python scripts/run_latency_benchmark.py --iterations 150 --output reports/latency_benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.benchmark_latency import LatencyBenchmarkSuite
from conftest.evaluation.headline import MissingArtifact
from conftest.features.pipeline import FEATURE_NAMES
from conftest.logging_config import get_logger
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.policy import SelectivePredictionPolicy

logger = get_logger(__name__)

ENSEMBLE_PRODUCED_BY = "python scripts/train_ensemble.py"
CALIBRATOR_PRODUCED_BY = "python scripts/calibrate_model.py"
POLICY_PRODUCED_BY = "python scripts/tune_policy.py"
SPLIT_PRODUCED_BY = "python scripts/build_splits.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Scoring Latency Benchmark CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=150,
                        help="Batches to time. Each batch stands for one commit.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Candidate tests per batch.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Batches timed separately before the pooled statistics start. "
                        "The first LightGBM call pays allocation costs no later batch pays.")
    parser.add_argument("--ensemble-dir", type=str, default="./models/ensembles/5_seed_lgbm",
                        help="Directory holding the serialized ensemble members.")
    parser.add_argument("--calibrator", type=str, default="./models/calibrator.joblib",
                        help="Fitted calibrator chosen on the validation split.")
    parser.add_argument("--policy-config", type=str, default="./models/policy_config.json",
                        help="Tuned selective policy thresholds.")
    parser.add_argument("--test-data", type=str, default="./data/splits/test.csv",
                        help="Real feature rows to score.")
    parser.add_argument("--output", type=str, default="./reports/latency_benchmark.json",
                        help="Destination for the measured report.")
    return parser.parse_args()


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise MissingArtifact(str(path), produced_by)
    return path


def main():
    args = parse_args()

    ensemble_dir = _require(Path(args.ensemble_dir), ENSEMBLE_PRODUCED_BY)
    calibrator_path = _require(Path(args.calibrator), CALIBRATOR_PRODUCED_BY)
    policy_path = _require(Path(args.policy_config), POLICY_PRODUCED_BY)
    test_path = _require(Path(args.test_data), SPLIT_PRODUCED_BY)

    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ensemble_dir))
    calibrator = ConfidenceCalibrator.load(str(calibrator_path))
    policy_cfg = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = SelectivePredictionPolicy(**policy_cfg)

    df = pd.read_csv(test_path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{test_path} is missing feature columns {missing}. Rebuild it with: "
            f"{SPLIT_PRODUCED_BY}"
        )
    feature_rows = df[FEATURE_NAMES].to_numpy()

    suite = LatencyBenchmarkSuite(ensemble, calibrator, policy, feature_rows)
    report = suite.benchmark_pipeline(
        num_iterations=args.iterations, batch_size=args.batch_size, warmup=args.warmup
    )

    report["provenance"] = {
        "ensemble_dir": str(ensemble_dir.as_posix()),
        "ensemble_members": len(ensemble.models),
        "ensemble_n_estimators": getattr(ensemble, "n_estimators", None),
        "calibrator": str(calibrator_path.as_posix()),
        "calibrator_method": getattr(calibrator, "method", type(calibrator).__name__),
        "policy_config": policy_cfg,
        "rows_from": str(test_path.as_posix()),
        "inputs_measured": True,
    }

    logger.info("=" * 85)
    logger.info(
        f"  Scoring latency: {report['num_iterations']} batches x "
        f"{report['batch_size_tests']} tests | "
        f"{report['provenance']['ensemble_members']}-member ensemble | "
        f"under 100 ms on {report['sla_under_100ms_compliance_pct']}% of batches"
    )
    logger.info(f"  Excludes: {report['excludes']}")
    logger.info(f"  Cold first batch: {report['cold_start_first_batch_ms']} ms, "
                f"outside the pooled statistics below")
    logger.info("=" * 85)
    logger.info(f"{'Stage':<24} | {'Mean':<10} | {'p50':<10} | {'p90':<10} | {'p99'}")
    logger.info("-" * 85)
    for stage, s in report["stages"].items():
        logger.info(
            f"{stage:<24} | {s['mean_ms']:>7.3f}ms | {s['median_p50_ms']:>7.3f}ms | "
            f"{s['p90_ms']:>7.3f}ms | {s['p99_ms']:>7.3f}ms"
        )
    logger.info("=" * 85)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
