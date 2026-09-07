#!/usr/bin/env python
"""
Harvest BugsInPy real defects into the ground-truth schema the harness emits.

The mutation harvest answers "does the selector catch a synthetic one-line
edit". This script answers the question a reviewer asks next: does it catch a
defect a human actually shipped and later fixed. BugsInPy supplies 501 such
defects across 17 projects; what it does not supply is a test universe, which is
the quantity test selection is scored against, so this script measures it.

    python scripts/harvest_bugsinpy.py --survey
    python scripts/harvest_bugsinpy.py --clone
    python scripts/harvest_bugsinpy.py --projects tqdm --limit 3

Two facts about cost, both measured rather than assumed. Every one of the 501
bugs pins a Python between 3.6.9 and 3.8.3; a machine without those interpreters
cannot honour a single pin, so --allow-interpreter-mismatch exists and stamps the
mismatch on every record it produces. And a bug costs one clone, one virtualenv,
one install and three suite runs, so this is an overnight job per project, not a
minute -- run it per project and let it resume.

Interrupted runs resume: every record is written as it is measured, and a bug
already present in real_bugs.jsonl is skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from conftest.groundtruth.bugsinpy_adapter import (  # noqa: E402
    BUGSINPY_URL,
    PROJECT_PROFILES,
    BugsInPyHarvester,
    list_projects,
    run_command,
    survey,
)
from conftest.logging_config import configure_logging, get_logger  # noqa: E402

# Imported rather than re-implemented: the screener owns the trick that stops
# pytest walking up out of a foreign checkout and adopting THIS project's
# [tool.pytest.ini_options] -- our `pythonpath = ["src"]` and our narrower
# `python_files`. The BugsInPy workspace lives inside this repository too, so it
# needs the same marker for the same reason.
from screen_repos import isolate_workspace_from_project_config  # noqa: E402

logger = get_logger(__name__)

DEFAULT_ROOT = REPO_ROOT / "data" / "groundtruth" / "bugsinpy"
DEFAULT_METADATA = DEFAULT_ROOT / "BugsInPy"
DEFAULT_WORKSPACE = DEFAULT_ROOT / "workspace"

# Per suite run. BugsInPy suites are whole-project suites, not the single node id
# its own runner executes: youtube-dl's is minutes even when nothing is wrong.
SUITE_TIMEOUT_SECONDS = 900

# Two runs is the minimum that can observe flakiness at all. The mutation
# harvest uses three because one baseline is amortised over hundreds of mutants;
# here every bug pays for its own baseline, so the third run doubles the cost of
# the cheapest thing in the pipeline.
BASELINE_RUNS = 2


def clone_metadata(metadata_dir: Path) -> str:
    """
    Clone (or update) the BugsInPy metadata repository and return its commit.

    Shallow, because only the current metadata is read, and the commit id is
    recorded on every record so a reader can fetch that exact revision. BugsInPy
    is a live repository: bugs are added and test commands corrected, so "the
    BugsInPy dataset" without a commit id is not a reproducible input.
    """
    metadata_dir.parent.mkdir(parents=True, exist_ok=True)
    if (metadata_dir / ".git").is_dir():
        logger.info(f"Updating {metadata_dir}")
        result = run_command(
            ["git", "fetch", "--depth", "1", "origin"], cwd=metadata_dir, timeout=900
        )
        if result.ok:
            run_command(["git", "reset", "--hard", "origin/HEAD"], cwd=metadata_dir, timeout=300)
    else:
        logger.info(f"Cloning {BUGSINPY_URL} into {metadata_dir}")
        result = run_command(
            ["git", "clone", "--depth", "1", BUGSINPY_URL, str(metadata_dir)], timeout=1800
        )
        if not result.ok:
            raise SystemExit(f"git clone failed: {result.tail()}")

    head = run_command(["git", "rev-parse", "HEAD"], cwd=metadata_dir, timeout=60)
    commit = head.stdout.strip() if head.ok else "unknown"
    logger.info(f"BugsInPy metadata at {commit}")
    return commit


def resolve_projects(metadata_dir: Path, requested: Optional[Sequence[str]]) -> List[str]:
    """
    Turn --projects into a checked list, refusing anything with no run profile.

    A project without a profile has no known test directory, so a harvest of it
    would collect zero tests -- which looks exactly like a project whose suite
    passes. Refusing early is the difference between an error and a fabricated
    row.
    """
    available = list_projects(metadata_dir)
    if not requested:
        return [name for name in available if name in PROJECT_PROFILES]

    resolved: List[str] = []
    for name in requested:
        if name not in available:
            raise SystemExit(
                f"'{name}' is not in {metadata_dir}/projects. Available: "
                f"{', '.join(available[:12])}..."
            )
        if name not in PROJECT_PROFILES:
            raise SystemExit(
                f"'{name}' has no run profile, so its test layout is unknown. "
                f"Collect its suite by hand once, then add a ProjectProfile for it. "
                f"Profiled: {', '.join(sorted(PROJECT_PROFILES))}."
            )
        resolved.append(name)
    return resolved


def print_survey(payload: dict) -> None:
    """One line per project: what is there, and whether it can be run here."""
    print(f"\nBugsInPy: {payload['n_bugs']} bugs across {payload['n_projects']} projects")
    print(f"readable failing-test command: {payload['n_with_readable_test_command']}")
    print(f"pinned interpreters: {payload['pinned_python_versions']}")
    print(f"\n{'project':<16}{'bugs':>6}{'readable':>10}{'profile':>9}  pinned")
    for name, row in sorted(payload["per_project"].items()):
        pinned = ",".join(sorted(row["python_versions"]))
        print(
            f"{name:<16}{row['n_bugs']:>6}{row['n_with_readable_test_command']:>10}"
            f"{'yes' if row['has_run_profile'] else 'no':>9}  {pinned}"
        )
    print()


def print_summary(summary: dict) -> None:
    """What the run measured, and what it could not."""
    print(f"\nrecords:            {summary['n_records']}")
    print(f"usable as labels:   {summary['n_labelled']}")
    print(f"projects with labels: {summary['n_projects_with_labels']} "
          f"({', '.join(summary['projects_with_labels']) or 'none'})")
    print(f"status counts:      {summary['status_counts']}")
    print(f"interpreter:        {summary['base_python_version']} "
          f"(mismatch allowed: {summary['allow_interpreter_mismatch']})")
    for project, row in sorted(summary["per_project"].items()):
        print(
            f"  {project:<14} {row['harvested']}/{row['records']} labelled, "
            f"mean universe {row['mean_universe_size']}, mean killed {row['mean_n_killed']}"
        )
    print(f"\nrecords: {summary['records_path']}")
    if summary["n_projects_with_labels"] < 3:
        print(
            "G6 asks for external validation on at least three projects; "
            f"this run has {summary['n_projects_with_labels']}."
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure BugsInPy defects into the mutation-harvest record schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="clone of the BugsInPy metadata repository")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                        help="where per-bug checkouts and virtualenvs are built")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT,
                        help="where records and per-bug baselines are written")
    parser.add_argument("--clone", action="store_true",
                        help="clone or update the metadata repository, then exit")
    parser.add_argument("--survey", action="store_true",
                        help="read the metadata and report feasibility, running nothing")
    parser.add_argument("--projects", nargs="*", default=None,
                        help="projects to harvest (default: every profiled project)")
    parser.add_argument("--bugs", nargs="*", type=int, default=None,
                        help="bug ids within the selected project(s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="measure at most this many bugs per project")
    parser.add_argument("--baseline-runs", type=int, default=BASELINE_RUNS,
                        help="suite runs at the fixed commit, to screen out flaky tests")
    parser.add_argument("--suite-timeout", type=int, default=SUITE_TIMEOUT_SECONDS,
                        help="seconds per suite run")
    parser.add_argument("--base-python", type=str, default=None,
                        help="interpreter used to build each venv (default: this one)")
    parser.add_argument("--allow-interpreter-mismatch", action="store_true",
                        help="measure even when the bug's pinned Python is unavailable; "
                             "the mismatch is stamped on every record")
    parser.add_argument("--pinned-requirements", action="store_true",
                        help="attempt the bug's frozen requirements before installing "
                             "the checkout (usually fails off the pinned interpreter)")
    parser.add_argument("--force", action="store_true",
                        help="re-measure bugs that already have a record")
    parser.add_argument("--summary", action="store_true",
                        help="print the last run's summary and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.clone:
        clone_metadata(args.metadata)
        return 0

    if args.summary:
        summary_path = args.output / "harvest_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"No summary at {summary_path}; nothing has been harvested yet.")
        print_summary(json.loads(summary_path.read_text(encoding="utf-8")))
        return 0

    if not (args.metadata / "projects").is_dir():
        raise SystemExit(
            f"No BugsInPy metadata at {args.metadata}. Run with --clone first."
        )

    if args.survey:
        payload = survey(args.metadata, args.projects)
        print_survey(payload)
        out_path = args.output / "bugsinpy_survey.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Survey written to {out_path}")
        return 0

    projects = resolve_projects(args.metadata, args.projects)
    if not projects:
        raise SystemExit("No profiled project selected; nothing to harvest.")

    args.workspace.mkdir(parents=True, exist_ok=True)
    marker = isolate_workspace_from_project_config(args.workspace)
    logger.info(f"Neutral pytest config for the workspace: {marker}")

    harvester = BugsInPyHarvester(
        bugsinpy_root=args.metadata,
        workspace=args.workspace,
        output_dir=args.output,
        baseline_runs=args.baseline_runs,
        suite_timeout=args.suite_timeout,
        allow_interpreter_mismatch=args.allow_interpreter_mismatch,
        install_pinned_requirements=args.pinned_requirements,
        base_python=args.base_python,
    )

    logger.info(
        f"Harvesting {', '.join(projects)} with {args.baseline_runs} baseline run(s) "
        f"under Python {harvester.base_python_version()}"
    )
    summary = harvester.harvest(
        projects=projects, bug_ids=args.bugs, limit=args.limit, force=args.force
    )
    print_summary(summary)
    return 0 if summary["n_labelled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
