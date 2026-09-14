"""
ConfTest Standalone Test Selection & Execution CLI.

Analyzes working tree diff or specific commit SHA, computes 32 features,
runs 5-seed uncertainty ensembling, evaluates selective abstention policy,
and optionally executes the selected regression test suite.

Usage:
    python scripts/select_tests.py --repo-path . --commit-sha HEAD --budget 0.25 --execute
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.engine.selector_engine import ConfTestEngine
from conftest.repository.git_collector import GitRepositoryMiner
from conftest.db.session import SessionLocal
from conftest.db import crud
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Test Selection Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Root path to target repository.",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default="HEAD",
        help="Commit SHA to analyze (or HEAD for latest commit/diff).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Budget fraction of tests to run in fast selective mode (e.g. 0.25 = top 25%%).",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Path to trained 5-seed ensemble directory.",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Path to fitted calibrator joblib file.",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default="./models/policy_config.json",
        help="Path to policy configuration JSON.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected tests in isolated subprocess and record outcomes.",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist predictions and decisions to SQLite database.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-run pytest timeout in seconds when executing selected tests.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to export selection outcome JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(args.repo_path).resolve()
    logger.info(f"Initializing ConfTest Engine for repository: {repo_root}...")

    engine = ConfTestEngine(
        repo_root=str(repo_root),
        ensemble_path=args.ensemble,
        calibrator_path=args.calibrator,
        policy_config_path=args.policy_config,
        default_budget=args.budget,
    )

    # Mine commit details from Git. A missing repo is a legitimate manual
    # run against a non-git directory (empty diff is honest then), but an
    # error while mining a real git repo must fail loudly: falling back to
    # a made-up diff would compute every feature against fiction.
    commit_sha = args.commit_sha
    changed_files = []
    commit_msg = "Manual test selection run"
    commit_time = datetime.utcnow()

    try:
        miner = GitRepositoryMiner(str(repo_root))
    except Exception as exc:
        logger.warning(f"Not a git repository ({exc}); running with an empty diff.")
        miner = None

    if miner and miner.repo:
        try:
            commit_obj = miner.repo.commit(args.commit_sha)
            commit_sha = commit_obj.hexsha
            commit_msg = commit_obj.message
            commit_time = datetime.utcfromtimestamp(commit_obj.committed_date)
            changed_files, _churn = miner.extract_commit_diffs(commit_obj)
        except Exception as exc:
            logger.error(f"Failed to mine commit diff for {args.commit_sha}: {exc}")
            sys.exit(2)

    # Optional DB session
    db = SessionLocal() if args.persist_db else None
    repo_id = None
    if db:
        repo_rec = crud.get_or_create_repository(db, "local/main-project", "https://github.com/local/main-project", str(repo_root))
        repo_id = repo_rec.id

    try:
        outcome = engine.analyze_and_select(
            commit_sha=commit_sha,
            changed_files=changed_files,
            commit_message=commit_msg,
            commit_timestamp=commit_time,
            budget_ratio=args.budget,
            db=db,
            repository_id=repo_id,
            execute=args.execute,
            timeout=args.timeout,
        )

        logger.info("\n=======================================================")
        logger.info(f"  ConfTest Decision Mode:   [{outcome['decision_mode']}]")
        logger.info(f"  Abstained / Fallback:     {outcome['abstained']}")
        logger.info(f"  Selected Tests:           {outcome['selected_count']} / {outcome['total_count']} ({outcome['test_reduction_pct']:.1f}% time saved)")
        logger.info(f"  Top Confidence Score:     {outcome['top_confidence']:.4f}")
        logger.info(f"  Epistemic Uncertainty:    {outcome['epistemic_uncertainty']:.4f}")
        logger.info(f"  Decision Rationale:       {' | '.join(outcome['reasons'])}")
        logger.info("=======================================================\n")

        if args.execute and outcome.get("execution_outcome"):
            exec_res = outcome["execution_outcome"]
            logger.info(f"Execution Completed in {exec_res['total_duration']}s (Passed: {exec_res['passed']}, Failed: {exec_res['failed']}, Exit Code: {exec_res['exit_code']})")

        if args.output_json:
            out_fp = Path(args.output_json)
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(outcome, f, indent=2)
            logger.info(f"Selection outcome exported to: {out_fp}")

        # Propagate the pytest exit code only after the JSON report is on
        # disk, so CI fails loudly when selected tests fail while the
        # report is still available for the PR comment step.
        if args.execute and outcome.get("execution_outcome"):
            exec_res = outcome["execution_outcome"]
            if exec_res.get("failed", 0) > 0 or exec_res.get("exit_code", 0) != 0:
                sys.exit(exec_res.get("exit_code") or 1)

    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main()
