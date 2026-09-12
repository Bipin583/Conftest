"""
Evaluation dispatcher: ``python scripts/evaluate.py``.

The documented workflow names ``scripts/evaluate.py``, but there is no single
evaluation in this project -- there are eleven, each writing its own artifact
under ``reports/`` and each with its own cost. This script does not recompute
anything itself. It maps every published artifact to the one script that
produces it, reports which artifacts are currently present, and forwards to a
producer on request.

That mapping is the point. Several modules refuse to serve a number when its
artifact is missing, and name the producing command instead of guessing; this
gives a reader the same map from the command line.

    python scripts/evaluate.py                 # show the map and what exists
    python scripts/evaluate.py --run baselines # run one producer
    python scripts/evaluate.py --all           # run all of them, in order

``--all`` is expensive (the cross-repository and continuous-learning stages
retrain per repository) and is deliberately not the default.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class Stage:
    """One evaluation, its producer script, and the artifacts it writes."""

    key: str
    producer: str
    artifacts: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def script(self) -> Path:
        return SCRIPTS_DIR / self.producer


# Ordered so that a full run produces prerequisites before consumers: the
# policy sweep needs a calibrator, and the recall-floor gate needs the sweep.
STAGES: tuple[Stage, ...] = (
    Stage(
        "calibration",
        "calibrate_model.py",
        ["calibration_report.json"],
        "fits the calibrator on validation and records which method won, and by how much",
    ),
    Stage(
        "policy",
        "tune_policy.py",
        ["policy_tuning_report.json", "policy_tuning_recall_floor.json"],
        "sweeps abstention thresholds on validation under a named objective",
    ),
    Stage(
        "baselines",
        "train_baseline.py",
        [
            "baseline_comparison.csv",
            "baseline_comparison_intervals.json",
            "baseline_per_commit.csv",
            "statistical_significance.json",
            "g4_ml_vs_baselines.json",
        ],
        "held-out head-to-head against the 7 baselines, with commit-unit bootstrap intervals",
    ),
    Stage(
        "recall-floor",
        "check_g5_recall_floor.py",
        ["g5_recall_floor.json"],
        "carries the validation-picked threshold to held-out data and reports whether the gate is met",
    ),
    Stage(
        "uncertainty",
        "uncertainty_eval.py",
        ["uncertainty_analysis.json"],
        "epistemic sigma, predictive entropy, and the risk-coverage curve",
    ),
    Stage(
        "ablation",
        "run_ablation_study.py",
        ["ablation_study.json"],
        "leave-one-group-out and single-group feature ablations",
    ),
    Stage(
        "flakiness",
        "run_flakiness_test.py",
        ["flakiness_robustness.json"],
        "label noise injected into training only, evaluated on clean held-out labels",
    ),
    Stage(
        "latency",
        "run_latency_benchmark.py",
        ["latency_benchmark.json"],
        "scoring-path latency only; excludes feature mining",
    ),
    Stage(
        "economics",
        "run_economic_analysis.py",
        ["economic_analysis.json"],
        "projects the measured reduction rate onto an assumed team; no dollar figure is observed",
    ),
    Stage(
        "cross-repo",
        "run_cross_repo_eval.py",
        ["cross_repo_generalization.json"],
        "leave-one-project-out transfer; expensive, retrains per repository",
    ),
    Stage(
        "continuous-learning",
        "run_continuous_learning.py",
        ["continuous_learning.json", "continuous_learning_timeline.csv"],
        "Page-Hinkley drift detection and replay adaptation; expensive",
    ),
)

BY_KEY = {stage.key: stage for stage in STAGES}


def _status(stage: Stage) -> str:
    missing = [name for name in stage.artifacts if not (REPORTS_DIR / name).exists()]
    if not missing:
        return "present"
    if len(missing) == len(stage.artifacts):
        return "absent"
    return f"partial ({len(stage.artifacts) - len(missing)}/{len(stage.artifacts)})"


def show_map() -> int:
    width = max(len(stage.key) for stage in STAGES)
    print("Every published number belongs to exactly one producer.\n")
    print(f"{'stage'.ljust(width)}  {'status'.ljust(9)}  producer")
    print(f"{'-' * width}  {'-' * 9}  {'-' * 32}")
    unresolved = []
    for stage in STAGES:
        if not stage.script.exists():
            unresolved.append(stage)
        print(f"{stage.key.ljust(width)}  {_status(stage).ljust(9)}  scripts/{stage.producer}")
    print("\nDetail:")
    for stage in STAGES:
        print(f"  {stage.key}: {stage.note}")
        for name in stage.artifacts:
            mark = "+" if (REPORTS_DIR / name).exists() else "-"
            print(f"      [{mark}] reports/{name}")
    if unresolved:
        print("\nBROKEN: these stages name a script that does not exist:", file=sys.stderr)
        for stage in unresolved:
            print(f"  {stage.key} -> scripts/{stage.producer}", file=sys.stderr)
        return 1
    print(
        "\nRun one with:  python scripts/evaluate.py --run <stage>"
        "\nFlags after -- are forwarded to the producer, e.g."
        "\n  python scripts/evaluate.py --run baselines -- --bootstraps 2000"
    )
    return 0


def run_stage(stage: Stage, forwarded: List[str]) -> int:
    if not stage.script.exists():
        print(f"missing producer: scripts/{stage.producer}", file=sys.stderr)
        return 1
    print(f"\n=== {stage.key}: scripts/{stage.producer} {' '.join(forwarded)} ===", flush=True)
    saved = sys.argv
    sys.argv = [str(stage.script), *forwarded]
    try:
        runpy.run_path(str(stage.script), run_name="__main__")
    except SystemExit as exc:  # a producer that exits non-zero must stop the run
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            print(f"FAILED: {stage.key} exited {code}", file=sys.stderr)
        return code
    finally:
        sys.argv = saved
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map published artifacts to their producers, and run producers on request.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", metavar="STAGE", choices=sorted(BY_KEY), help="run one stage")
    group.add_argument("--all", action="store_true", help="run every stage in dependency order")
    parser.add_argument(
        "forwarded",
        nargs="*",
        help="arguments passed through to the producer (put them after --)",
    )
    args = parser.parse_args(argv)

    if args.all:
        for stage in STAGES:
            code = run_stage(stage, args.forwarded)
            if code:
                return code
        print("\nAll stages completed.")
        return 0

    if args.run:
        return run_stage(BY_KEY[args.run], args.forwarded)

    return show_map()


if __name__ == "__main__":
    raise SystemExit(main())
