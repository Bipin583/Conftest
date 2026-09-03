"""
ConfTest Multi-Repository Cross-Project Generalization CLI.

Leave-One-Project-Out transfer over the five harvested repositories: train on four,
calibrate on a held-out slice of those four, and predict on the fifth, which the
model has never seen. This is the evidence G5 asks for -- a reduction figure on an
unseen repository -- so it has to be measured on the real harvest.

The input is `data/processed/real_features.csv`, whose labels are pytest outcomes
recorded per mutant. Rows are grouped by `commit_sha`, one mutant applied to one
checkout, and recall at a budget is computed per commit and pooled.

Until 2026-09-03 this script invented its own input: four `generate_mock_repo_dataset`
calls drew Gaussian feature matrices, planted signal on three column indices, and
thresholded a logistic function of them into labels. It then reported those results
under the names of four real projects -- requests, flask, fastapi, click -- none of
which had been touched. Nothing in the pipeline flagged it, because the label guard
polices the label path and this was on the reporting path.

Usage:
    python scripts/run_cross_repo_eval.py --output reports/cross_repo_generalization.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.cross_repo import CrossRepoEvaluator
from conftest.evaluation.headline import MissingArtifact
from conftest.features.pipeline import FEATURE_NAMES
from conftest.logging_config import get_logger

logger = get_logger(__name__)

PRODUCED_BY = "python scripts/build_real_dataset.py --all"
MIN_REPOS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Cross-Repository Generalization Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/processed/real_features.csv",
        help="Labelled dataset carrying a 'repo' column, one row per (mutant, test).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Fraction of each commit's own test universe a policy may run.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Trees per transfer model. Matches the reported ensemble's members.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/cross_repo_generalization.json",
        help="Path to output JSON report.",
    )
    return parser.parse_args()


def load_repo_datasets(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split the labelled dataset into one per-repository block.

    Returns:
        repo -> {'X', 'y', 'groups'}, with X restricted to FEATURE_NAMES in order.

    Raises:
        MissingArtifact: If the dataset has not been built.
        ValueError: If it lacks the columns transfer evaluation needs, or carries
            fewer than two repositories -- there is nothing to hold out from one.
    """
    if not path.exists():
        raise MissingArtifact(path, PRODUCED_BY)

    frame = pd.read_csv(path)
    required = {"repo", "commit_sha", "label_failed", *FEATURE_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path.as_posix()} is missing {len(missing)} needed columns "
            f"(first few: {missing[:5]}). Rebuild it with: {PRODUCED_BY}"
        )

    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    for repo, block in frame.groupby("repo"):
        datasets[str(repo)] = {
            "X": block[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),
            "y": block["label_failed"].to_numpy(dtype=np.int32),
            "groups": block["commit_sha"].to_numpy(dtype=object),
        }
    if len(datasets) < MIN_REPOS:
        raise ValueError(
            f"{path.as_posix()} carries {len(datasets)} repository/ies. "
            "Leave-one-project-out needs at least two, and the study claims five."
        )
    return datasets


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    repo_datasets = load_repo_datasets(dataset_path)
    logger.info(
        f"Loaded {len(repo_datasets)} repositories from {dataset_path}: "
        + ", ".join(
            f"{repo} ({len(d['y']):,} rows, {int(d['y'].sum()):,} failures, "
            f"{len(np.unique(d['groups']))} mutants)"
            for repo, d in repo_datasets.items()
        )
    )

    evaluator = CrossRepoEvaluator(random_seed=42, n_estimators=args.n_estimators)
    report: Dict[str, Any] = evaluator.evaluate_lopo_transfer(
        repo_datasets, budget_ratio=args.budget
    )
    report["dataset"] = dataset_path.as_posix()
    report["produced_by"] = PRODUCED_BY
    report["labels_measured"] = True
    report["label_provenance"] = (
        "every label is a recorded pytest outcome for one mutant against one test, "
        "harvested per repository; no feature or label in this report is generated"
    )

    header = (
        f"{'Target Repository':<14} | {'Rows':>9} | {'Fail %':>7} | {'PR-AUC':>7} | "
        f"{'ROC-AUC':>7} | {'ECE':>6} | {'Recall@budget':>13}"
    )
    logger.info("\n" + "=" * len(header))
    logger.info("  Leave-One-Project-Out transfer: trained on the other four, never on this one")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for repo, res in report["per_repository"].items():
        recall = res["zero_shot_recall_at_budget"]
        logger.info(
            f"{repo:<14} | {res['target_samples']:>9,} | "
            f"{res['target_failure_rate']*100:>6.2f}% | "
            f"{res['zero_shot_pr_auc']:>7.4f} | {res['zero_shot_roc_auc']:>7.4f} | "
            f"{res['zero_shot_calibrated_ece']:>6.4f} | "
            + (f"{recall*100:>12.2f}%" if recall is not None else f"{'n/a':>13}")
        )

    macro = report["macro_average"]
    logger.info("-" * len(header))
    mean_recall = macro["mean_recall_at_budget"]
    logger.info(
        f"{'MACRO MEAN':<14} | {'-':>9} | {'-':>7} | "
        f"{macro['mean_pr_auc']:>7.4f} | {macro['mean_roc_auc']:>7.4f} | "
        f"{macro['mean_calibrated_ece']:>6.4f} | "
        + (f"{mean_recall*100:>12.2f}%" if mean_recall is not None else f"{'n/a':>13}")
    )
    logger.info("=" * len(header))
    logger.info(
        f"Recall is measured per commit at a {args.budget:.0%} budget of that commit's "
        "own universe and pooled; a commit no test detected has no recall to measure "
        "and is excluded from it."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Cross-repo report saved to: {out_path}")


if __name__ == "__main__":
    main()
