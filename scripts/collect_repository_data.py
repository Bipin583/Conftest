"""
ConfTest Repository Data Collection CLI.

Mines Git commit history, file diffs, and CI execution traces from local
repositories or generates labeled synthetic benchmark suites for offline experimentation.

Usage Examples:
    # 1. Mine current local repository
    python scripts/collect_repository_data.py --repo-path . --name local/current-project --max-commits 50

    # 2. Generate synthetic demo suite for offline benchmarking
    python scripts/collect_repository_data.py --synthetic --name synthetic/demo-app --max-commits 100 --tests 40
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.db.session import SessionLocal
from conftest.logging_config import get_logger
from conftest.repository.collector_service import CollectorService
from conftest.repository.synthetic_generator import SyntheticRepositoryGenerator

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Repository & CI Data Collection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Path to local Git repository to mine.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="local/current-repo",
        help="Repository identifier name (e.g. 'pallets/flask').",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=50,
        help="Maximum number of commits to mine.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Target branch to traverse (defaults to default branch or HEAD).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw",
        help="Output directory for raw JSON datasets and quality reports.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run in synthetic demo mode to produce offline benchmark datasets.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        default=30,
        help="Number of test cases to synthesize (only used with --synthetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation reproducibility.",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        default=True,
        help="Persist collected/synthesized records into ConfTest SQLite database.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    db = SessionLocal() if args.persist_db else None

    try:
        if args.synthetic:
            logger.info("Running in SYNTHETIC DEMO MODE...")
            generator = SyntheticRepositoryGenerator(
                random_seed=args.seed, acknowledge_synthetic=True
            )
            dataset = generator.generate_repository_suite(
                repo_name=args.name,
                n_commits=args.max_commits,
                n_tests=args.tests,
            )

            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"synthetic_{args.name.replace('/', '_')}.json"

            serializable_dataset = copy.deepcopy(dataset)
            for c in serializable_dataset["commits"]:
                if hasattr(c["timestamp"], "isoformat"):
                    c["timestamp"] = c["timestamp"].isoformat()

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(serializable_dataset, f, indent=2)

            logger.info(f"Synthetic dataset written to: {out_file}")

            if db:
                summary = generator.persist_to_database(db, dataset)
                logger.info(f"Persisted to database: {summary}")

        else:
            logger.info("Running in REAL GIT MINING MODE...")
            collector = CollectorService(
                repo_path=args.repo_path,
                repo_name=args.name,
                output_dir=args.output,
            )
            report = collector.collect(
                db=db,
                max_commits=args.max_commits,
                branch=args.branch,
            )
            logger.info(f"Collection Report:\n{json.dumps(report, indent=2)}")

    except Exception as exc:
        logger.error(f"Data collection failed: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main()
