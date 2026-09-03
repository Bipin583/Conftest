#!/usr/bin/env python
"""
Run the mutation harness over screened subject repositories.

This is the step that produces ground truth. Every label the project trains on
comes from a pytest run recorded here, so the script is deliberately narrow:
it reads the screener's verdict, refuses anything the screener did not accept,
and writes one harvest directory per repository.

    python scripts/harvest_mutations.py --repos parse --mutants 40
    python scripts/harvest_mutations.py --all --mutants 250

Cost scales as (suite seconds x mutants). The screening report records each
suite's measured duration, so the run prints an estimate before starting and
you can split repositories across laptops by hand.

Interrupted runs resume: the harness checkpoints after every mutant, so
re-running the same command continues rather than repeating work. Pass
--no-resume to start a repository over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from conftest.groundtruth.mutation_harness import MutationHarness  # noqa: E402
from conftest.logging_config import configure_logging, get_logger  # noqa: E402

# Imported rather than re-derived on purpose. The screener owns the layout of
# the workspace it builds; if it ever renames a venv directory or changes the
# neutral-config marker, this script must follow automatically instead of
# silently pointing at a path that no longer exists.
from screen_repos import (  # noqa: E402
    isolate_workspace_from_project_config,
    venv_python,
)

logger = get_logger(__name__)

# Per-suite wall-clock cap. A mutation can turn a fast test into an infinite
# loop, so the cap has to be generous enough for an honest slow run and tight
# enough that a hang does not eat the night. The slowest accepted suite runs in
# ~12s clean; 10x that is comfortably outside normal variance.
SUITE_TIMEOUT_SECONDS = 180

# Repeat count for the baseline flakiness screen inside the harness. The
# screener already ran three identical suites, but the harness re-screens in
# its own environment: a test that is stable under the screener and flaky here
# would otherwise be recorded as mutation-killed when it simply wobbled.
BASELINE_RUNS = 3


def accepted_from_report(report_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the screener verdict, keyed by repository name."""
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No screening report at {report_path}. Run scripts/screen_repos.py --all first."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    # Stage 3 is the last gate: it rejects repositories whose dependency features
    # do not vary, where a harvest would cost real hours and produce rows that
    # cannot exercise test selection at all.
    accepted = {
        name: entry
        for name, entry in payload.get("candidates", {}).items()
        if isinstance(entry, dict) and entry.get("stage3_passed")
    }
    if not accepted:
        raise ValueError(
            f"No repository passed stage 3 in {report_path}. "
            f"Run scripts/screen_repos.py --stage3 to apply the structural gate."
        )
    return accepted


def resolve_interpreter(workspace: Path, name: str) -> Path:
    """
    The interpreter that must run this repository's suite.

    Not `sys.executable`. The harness measures whether a mutation changes test
    outcomes, which is only true if the tests import the mutated checkout. The
    screener guarantees exactly that for the per-repo venv it built with
    `pip install -e .`; the ambient interpreter guarantees nothing, and when it
    happens to hold a released copy of the same package -- sqlparse is
    installed in this project's own environment -- every mutant looks harmless
    and every label comes out 0.

    Missing is fatal rather than a fallback. A fallback to `sys.executable`
    would reintroduce exactly the failure this function exists to prevent, and
    it would do so quietly, at the point where hours of compute begin.
    """
    python = venv_python(workspace / f".venv_{name}")
    if not python.is_file():
        raise FileNotFoundError(
            f"{name}: no screened environment at {python}. Run "
            f"scripts/screen_repos.py --stage2 --repos {name} to build it. "
            f"Harvesting under the ambient interpreter would import whatever "
            f"copy of the package that environment holds, not the checkout "
            f"being mutated, and would label every mutant as harmless."
        )
    return python


def estimate_hours(entries: Sequence[Dict[str, Any]], n_mutants: int) -> float:
    """Sum of (measured clean suite duration x mutants), in hours."""
    seconds = sum(float(e.get("suite_seconds", 0.0)) for e in entries) * n_mutants
    return seconds / 3600.0


def harvest_one(
    name: str,
    entry: Dict[str, Any],
    workspace: Path,
    harvest_root: Path,
    n_mutants: int,
    resume: bool,
    python_executable: Path,
    restore_checkout: bool = False,
    reprofile_baseline: bool = False,
    break_lock: bool = False,
) -> Dict[str, Any]:
    """Harvest one repository. Raises rather than returning a partial result."""
    repo_root = (workspace / name).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"{name}: checkout missing at {repo_root}. "
            f"Re-run scripts/screen_repos.py to restore it from the pinned SHA."
        )

    source_dirs = list(entry.get("source_dirs", []))
    if not source_dirs:
        raise ValueError(
            f"{name}: the screening report records no source_dirs, so there is "
            f"nothing safe to mutate. Re-screen the repository."
        )

    logger.info(f"{name}: interpreter {python_executable}")
    harness = MutationHarness(
        repo_root=str(repo_root),
        repo_name=name,
        source_dirs=source_dirs,
        output_dir=str(harvest_root / name),
        suite_timeout=SUITE_TIMEOUT_SECONDS,
        baseline_runs=BASELINE_RUNS,
        python_executable=str(python_executable),
    )

    started = time.time()
    summary = harness.run(
        n_mutants=n_mutants,
        resume=resume,
        restore_checkout=restore_checkout,
        reprofile_baseline=reprofile_baseline,
        break_lock=break_lock,
    )
    summary["wall_clock_seconds"] = round(time.time() - started, 1)
    summary["screened_commit_sha"] = entry.get("commit_sha", "")

    # run() already wrote the summary; rewrite it so the on-disk copy carries the
    # two fields only the driver knows. The dataset builder reads this file.
    summary["summary_path"] = str(harness.write_summary(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harvest measured mutation-kill labels from screened repositories."
    )
    parser.add_argument("--repos", nargs="*", default=None, help="repository names to harvest")
    parser.add_argument("--all", action="store_true", help="harvest every screened-accepted repo")
    parser.add_argument("--mutants", type=int, default=250, help="mutants per repository")
    parser.add_argument("--workspace", default="data/repos", help="where the checkouts live")
    parser.add_argument("--harvest-root", default="data/harvest", help="where to write harvests")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore the checkpoint and re-harvest from the first mutant",
    )
    parser.add_argument(
        "--restore-checkout", action="store_true",
        help=(
            "discard modifications to tracked files in the subject checkout "
            "before harvesting; without it an unclean checkout aborts the repo"
        ),
    )
    parser.add_argument(
        "--reprofile-baseline", action="store_true",
        help=(
            "re-run the flakiness screen instead of reusing the cached one; "
            "required after repairing a checkout the cached screen ran against"
        ),
    )
    parser.add_argument(
        "--break-lock", action="store_true",
        help=(
            "take over the harvest lock on the output directory; only safe once "
            "the process named in harvest.lock is confirmed dead"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and the cost estimate, then stop",
    )
    args = parser.parse_args()

    configure_logging()

    workspace = (REPO_ROOT / args.workspace).resolve()
    harvest_root = (REPO_ROOT / args.harvest_root).resolve()
    accepted = accepted_from_report(workspace / "screening_report.json")

    if args.all:
        names = sorted(accepted)
    elif args.repos:
        names = list(args.repos)
        unknown = [n for n in names if n not in accepted]
        if unknown:
            parser.error(
                f"Not accepted by screening: {unknown}. Eligible: {sorted(accepted)}"
            )
    else:
        parser.error("choose --all or --repos NAME [NAME ...]")

    entries = [accepted[n] for n in names]
    hours = estimate_hours(entries, args.mutants)

    # Resolved for every repository before any of them starts. Discovering a
    # missing environment after the first repo has run for twenty minutes wastes
    # those minutes; discovering it now costs a stat call.
    interpreters = {name: resolve_interpreter(workspace, name) for name in names}

    # pytest searches upward for configuration, so without this marker a subject
    # repo that ships no pytest section adopts ConfTest's own -- including our
    # `pythonpath = ["src"]`. The screener writes it too; harvesting must not
    # depend on the screener having run last in this workspace.
    marker = isolate_workspace_from_project_config(workspace)

    logger.info("=" * 74)
    logger.info(f"Harvest plan: {len(names)} repo(s) x {args.mutants} mutants")
    for name, entry in zip(names, entries):
        per_repo = float(entry.get("suite_seconds", 0.0)) * args.mutants / 3600.0
        logger.info(
            f"  {name:<20} {entry.get('n_tests_collected', '?'):>5} tests  "
            f"{entry.get('suite_seconds', 0.0):>6.2f}s/suite  ~{per_repo:.2f}h"
        )
        logger.info(f"  {'':<20} {interpreters[name]}")
    logger.info(f"Estimated total: {hours:.2f}h (excludes baseline screening)")
    logger.info(f"Neutral pytest config: {marker}")
    logger.info("=" * 74)

    if args.dry_run:
        return 0

    summaries: List[Dict[str, Any]] = []
    failed: List[str] = []

    for name in names:
        logger.info("")
        logger.info(f"--- {name} ---")
        try:
            summaries.append(harvest_one(
                name=name,
                entry=accepted[name],
                workspace=workspace,
                harvest_root=harvest_root,
                n_mutants=args.mutants,
                resume=not args.no_resume,
                python_executable=interpreters[name],
                restore_checkout=args.restore_checkout,
                reprofile_baseline=args.reprofile_baseline,
                break_lock=args.break_lock,
            ))
        except Exception as exc:
            # One repository failing must not discard the others' hours of work.
            logger.error(f"{name}: harvest failed: {exc}")
            failed.append(f"{name}: {exc}")

    logger.info("")
    logger.info("=" * 74)
    for summary in summaries:
        total = summary.get("n_sampled", 0)
        logger.info(
            f"{summary['repo']:<20} harvested {summary.get('harvested', 0):>4}/{total}  "
            f"broke_suite {summary.get('broke_suite', 0):>3}  "
            f"timed_out {summary.get('timed_out', 0):>3}  "
            f"kills {summary.get('total_kills', 0):>6}  "
            f"drifted {summary.get('checkout_drifted', 0):>3}  "
            f"{summary.get('wall_clock_seconds', 0.0) / 60:.1f}min"
        )
    if failed:
        logger.error(f"{len(failed)} repository/ies failed:")
        for line in failed:
            logger.error(f"  {line}")
    logger.info("=" * 74)
    logger.info("Next: python scripts/build_real_dataset.py --repos " + " ".join(names))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
