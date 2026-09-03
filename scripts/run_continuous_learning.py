"""
ConfTest Online Continual Learning CLI.

Streams the real harvest past the model one mutant at a time and asks two questions
that can be answered with measurements rather than assumptions:

  1. Is the error stationary inside a single project's history? Every drift flag
     raised while the stream stays in one repository is a false alarm, because
     nothing about the data-generating process changed.
  2. When the stream crosses from one repository to another, does Page-Hinkley
     notice, and how many mutants does it take? The crossing point is known -- the
     stream is built by concatenation -- but the rows on both sides are real
     harvested rows, and the leave-one-project-out result already established that
     the two sides differ (macro transfer ROC-AUC 0.7004, tabulate 0.4747).

The detector's threshold is not chosen here. Picking the one value that makes drift
appear on cue is how a knob gets tuned into a finding, so the whole grid is swept and
published: for each threshold, the false alarms it raises on a stationary stream and
the detection latency it buys on a real shift. A threshold that detects the shift
instantly and also fires constantly within a repo has detected nothing.

Stream unit: one mutant, contributing one row per test in the suite (191 to 895 rows
depending on the repository), ordered by commit_timestamp then mutant_index.

Until 2026-09-03 this script had no input. It drew `X_init = rng.randn(150, n_feats)`,
labelled it `rng.rand(150) < 0.08`, then streamed 40 invented commits of ten rows
each, and at commit 20 injected drift by shifting the Gaussian mean by 2.0 and the
failure rate from 0.05 to 0.40. The drift it reported detecting was drift it had
inserted itself, and reports/continuous_learning.json published the timeline as a
result. The controlled synthetic stream that exercises the adaptation mechanism is a
unit test, and lives in tests/unit/test_continuous_learning.py where it belongs.

Usage:
    python scripts/run_continuous_learning.py --output reports/continuous_learning.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.headline import MissingArtifact
from conftest.features.pipeline import FEATURE_NAMES
from conftest.logging_config import get_logger
from conftest.models.continuous_learning import OnlineContinualLearner

logger = get_logger(__name__)

PRODUCED_BY = "python scripts/build_real_dataset.py --all"
STREAM_ORDER = ["commit_timestamp", "mutant_index"]
REQUIRED_COLUMNS = {"repo", "commit_sha", "label_failed", *STREAM_ORDER}

# Swept, not selected. Page-Hinkley accumulates mean absolute error above a running
# minimum, so the useful range depends on the error scale, which is measured here for
# the first time. Reporting the grid keeps the choice out of the finding.
THRESHOLD_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Online Continual Learning and Drift Detection Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/processed/real_features.csv",
        help="Labelled dataset carrying a 'repo' column, one row per (mutant, test).",
    )
    parser.add_argument(
        "--init-mutants",
        type=int,
        default=40,
        help="Mutants used to fit the base model before streaming begins.",
    )
    parser.add_argument(
        "--segment-mutants",
        type=int,
        default=120,
        help="Mutants streamed per segment. A shift run streams one segment from the "
        "source repository and one from the target.",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=5000,
        help="Replay buffer rows. One mutant contributes a whole suite of rows, so a "
        "buffer below the suite size cannot hold a single mutant.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for the LightGBM fits. Does not affect the stream order.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/continuous_learning.json",
        help="Destination for the measured report.",
    )
    parser.add_argument(
        "--timeline-output",
        type=str,
        default="./reports/continuous_learning_timeline.csv",
        help="Per-mutant error traces. They are the plottable part and would be tens of "
        "thousands of rows inside the JSON, so they ship beside it.",
    )
    return parser.parse_args()


def load_stream(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Group the harvest into one ordered mutant stream per repository.

    Raises MissingArtifact rather than fabricating a stream: this script has no
    fallback input, which is the point of the rewrite.
    """
    if not path.exists():
        raise MissingArtifact(path, PRODUCED_BY)

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns {sorted(missing)}. "
            f"Rebuild it with: {PRODUCED_BY}"
        )

    df = df.sort_values(STREAM_ORDER, kind="mergesort").reset_index(drop=True)

    streams: Dict[str, Dict[str, Any]] = {}
    for repo, sub in df.groupby("repo", sort=True):
        mutants: List[Dict[str, Any]] = []
        for sha, rows in sub.groupby("commit_sha", sort=False):
            mutants.append(
                {
                    "commit_sha": str(sha),
                    "X": rows[FEATURE_NAMES].to_numpy(dtype=np.float32),
                    "y": rows["label_failed"].to_numpy(dtype=int),
                }
            )
        streams[str(repo)] = {
            "mutants": mutants,
            "suite_size": int(sub.groupby("commit_sha").size().median()),
            "n_mutants": len(mutants),
        }
        logger.info(
            f"  {repo}: {len(mutants)} mutants in stream order, "
            f"{streams[str(repo)]['suite_size']} rows each"
        )
    return streams


def _fit_base(stream: Dict[str, Any], init_mutants: int, threshold: float, args):
    """Fit the base model on the first `init_mutants` mutants of a repository."""
    seed_mutants = stream["mutants"][:init_mutants]
    X_init = np.vstack([m["X"] for m in seed_mutants])
    y_init = np.concatenate([m["y"] for m in seed_mutants])
    if y_init.sum() == 0:
        raise ValueError(
            f"The first {init_mutants} mutants carry no failing row, so the base model "
            "would have one class. Raise --init-mutants."
        )
    learner = OnlineContinualLearner(
        buffer_capacity=args.buffer_capacity,
        drift_threshold=threshold,
        random_seed=args.random_seed,
    )
    learner.initialize_base_model(X_init, y_init)
    return learner, {"init_mutants": len(seed_mutants), "init_rows": int(len(y_init)),
                     "init_failing_rows": int(y_init.sum())}


def _stream_segment(learner, mutants, label, start_step, timeline) -> int:
    """Push one segment through the learner, recording a row per mutant."""
    for offset, mutant in enumerate(mutants):
        result = learner.process_streaming_commit(mutant["X"], mutant["y"])
        result["step"] = start_step + offset
        result["segment"] = label
        result["commit_sha"] = mutant["commit_sha"]
        timeline.append(result)
    return start_step + len(mutants)


def _segment_errors(timeline, label) -> List[float]:
    return [row["mean_commit_error"] for row in timeline if row["segment"] == label]


def run_stationary(repo: str, stream: Dict[str, Any], threshold: float, args) -> Dict[str, Any]:
    """
    Stream one repository past a model trained on that same repository.

    Every flag here is a false alarm by construction: the stream never leaves the
    project, so there is no shift to find.
    """
    learner, init = _fit_base(stream, args.init_mutants, threshold, args)
    tail = stream["mutants"][args.init_mutants : args.init_mutants + args.segment_mutants]
    timeline: List[Dict[str, Any]] = []
    _stream_segment(learner, tail, "stationary", 0, timeline)

    errors = _segment_errors(timeline, "stationary")
    flags = [row["step"] for row in timeline if row["drift_detected"]]
    return {
        "repo": repo,
        "threshold": threshold,
        **init,
        "streamed_mutants": len(timeline),
        "false_alarms": len(flags),
        "false_alarm_steps": flags,
        "adaptations": int(timeline[-1]["total_adaptations"]) if timeline else 0,
        "mean_error": float(np.mean(errors)) if errors else float("nan"),
        "max_error": float(np.max(errors)) if errors else float("nan"),
    }


def run_shift(source: str, target: str, streams, threshold: float, args) -> Dict[str, Any]:
    """
    Stream the source repository, then hand the same model the target repository.

    The crossing step is known because the stream is concatenated, so detection
    latency is measured against a real shift whose location is not in question.
    """
    learner, init = _fit_base(streams[source], args.init_mutants, threshold, args)
    src_tail = streams[source]["mutants"][
        args.init_mutants : args.init_mutants + args.segment_mutants
    ]
    tgt_head = streams[target]["mutants"][: args.segment_mutants]

    timeline: List[Dict[str, Any]] = []
    shift_step = _stream_segment(learner, src_tail, "source", 0, timeline)
    _stream_segment(learner, tgt_head, "target", shift_step, timeline)

    before = [row["step"] for row in timeline
              if row["drift_detected"] and row["segment"] == "source"]
    after = [row["step"] for row in timeline
             if row["drift_detected"] and row["segment"] == "target"]
    src_errors = _segment_errors(timeline, "source")
    tgt_errors = _segment_errors(timeline, "target")

    latency: Optional[int] = (after[0] - shift_step + 1) if after else None
    return {
        "source_repo": source,
        "target_repo": target,
        "threshold": threshold,
        **init,
        "shift_at_step": shift_step,
        "source_mutants": len(src_errors),
        "target_mutants": len(tgt_errors),
        "false_alarms_before_shift": len(before),
        "flags_after_shift": len(after),
        "detection_latency_mutants": latency,
        "detected": latency is not None,
        "mean_error_source": float(np.mean(src_errors)) if src_errors else float("nan"),
        "mean_error_target": float(np.mean(tgt_errors)) if tgt_errors else float("nan"),
        "adaptations": int(timeline[-1]["total_adaptations"]) if timeline else 0,
        "timeline": timeline,
    }



def rotation_pairs(repos: List[str]) -> List[tuple]:
    """
    Each repository is the source exactly once and the target exactly once.

    A fixed cyclic rotation over the sorted names, so no pair is chosen for the
    result it gives.
    """
    return [(repos[i], repos[(i + 1) % len(repos)]) for i in range(len(repos))]


def adaptation_effect(stationary: List[Dict[str, Any]], repos: List[str]) -> List[Dict[str, Any]]:
    """
    Per repository, the streamed error when adaptation never fired against the error
    when it fired most often -- on a stream that never left the project.

    Retraining on a stationary stream should be neutral at worst. If the error is
    higher in the column where the model was rebuilt, the adaptation step is not
    paying for itself, and that is a result about the mechanism rather than about
    the threshold that triggered it.
    """
    rows = []
    for repo in repos:
        runs = [r for r in stationary if r["repo"] == repo]
        quiet = [r for r in runs if r["adaptations"] == 0]
        busy = max(runs, key=lambda r: r["adaptations"])
        if not quiet or busy["adaptations"] == 0:
            rows.append({"repo": repo, "comparable": False,
                         "reason": "adaptation either never fired or always fired"})
            continue
        base = min(quiet, key=lambda r: r["threshold"])
        rows.append({
            "repo": repo,
            "comparable": True,
            "no_adaptation": {"threshold": base["threshold"], "mean_error": base["mean_error"]},
            "most_adaptation": {"threshold": busy["threshold"], "mean_error": busy["mean_error"],
                                "adaptations": busy["adaptations"]},
            "error_delta": busy["mean_error"] - base["mean_error"],
        })
    return rows


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("ConfTest Online Continual Learning -- real harvest stream")
    logger.info("=" * 70)

    streams = load_stream(Path(args.dataset))
    repos = sorted(streams)
    needed = args.init_mutants + args.segment_mutants
    short = {r: streams[r]["n_mutants"] for r in repos if streams[r]["n_mutants"] < needed}
    if short:
        raise ValueError(
            f"Repositories {short} have fewer than {needed} mutants "
            f"(--init-mutants + --segment-mutants). Lower one of them."
        )

    pairs = rotation_pairs(repos)
    logger.info(f"Rotation: {', '.join(f'{s} -> {t}' for s, t in pairs)}")

    stationary: List[Dict[str, Any]] = []
    shifts: List[Dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        logger.info(f"--- threshold {threshold} ---")
        for repo in repos:
            row = run_stationary(repo, streams[repo], threshold, args)
            stationary.append(row)
            logger.info(
                f"  stationary {repo:12s} false alarms {row['false_alarms']:3d} "
                f"mean error {row['mean_error']:.4f}"
            )
        for source, target in pairs:
            row = run_shift(source, target, streams, threshold, args)
            shifts.append(row)
            logger.info(
                f"  shift {source:12s} -> {target:12s} "
                f"latency {row['detection_latency_mutants']} "
                f"pre-shift alarms {row['false_alarms_before_shift']} "
                f"error {row['mean_error_source']:.4f} -> {row['mean_error_target']:.4f}"
            )

    # One row per threshold: what it costs on a stationary stream against what it buys
    # on a real one. Read the two columns together or not at all.
    summary = []
    for threshold in THRESHOLD_GRID:
        st = [r for r in stationary if r["threshold"] == threshold]
        sh = [r for r in shifts if r["threshold"] == threshold]
        latencies = [r["detection_latency_mutants"] for r in sh if r["detected"]]
        summary.append(
            {
                "threshold": threshold,
                "false_alarms_total": int(sum(r["false_alarms"] for r in st)),
                "stationary_mutants_streamed": int(sum(r["streamed_mutants"] for r in st)),
                "shifts_detected": len(latencies),
                "shifts_attempted": len(sh),
                "median_detection_latency_mutants": (
                    float(np.median(latencies)) if latencies else None
                ),
                "pre_shift_false_alarms_total": int(
                    sum(r["false_alarms_before_shift"] for r in sh)
                ),
            }
        )

    all_stationary_errors = [r["mean_error"] for r in stationary]
    report = {
        "labels_measured": True,
        "dataset": str(Path(args.dataset).as_posix()),
        "produced_by": PRODUCED_BY,
        "observation_unit": "mutant",
        "stream_order": STREAM_ORDER,
        "error_metric": "mean absolute error between label and predicted probability, "
        "averaged over the rows of one mutant",
        "threshold_grid": THRESHOLD_GRID,
        "threshold_selected": None,
        "threshold_selection_note": "Not selected here. The grid is published so the "
        "false-alarm cost of a sensitive threshold is visible next to the detection "
        "latency it buys.",
        "shift_construction": "Segments are concatenated, so the crossing step is known "
        "by construction. The rows on both sides are real harvested rows; only their "
        "adjacency is arranged.",
        "config": {
            "init_mutants": args.init_mutants,
            "segment_mutants": args.segment_mutants,
            "buffer_capacity": args.buffer_capacity,
            "random_seed": args.random_seed,
            "n_estimators": 30,
        },
        "repositories": {
            r: {"mutants": streams[r]["n_mutants"], "suite_size": streams[r]["suite_size"]}
            for r in repos
        },
        "rotation": [{"source": s, "target": t} for s, t in pairs],
        "error_scale_observed": {
            "min_stationary_mean_error": float(np.min(all_stationary_errors)),
            "max_stationary_mean_error": float(np.max(all_stationary_errors)),
        },
        "stationary_adaptation_effect": adaptation_effect(stationary, repos),
        "detector_is_one_sided": "Page-Hinkley accumulates error above a running "
        "minimum, so it can only see a shift that makes the model worse. A crossing "
        "into an easier project lowers the error and is invisible by construction, "
        "not by misconfiguration.",
        "threshold_summary": summary,
        "stationary_runs": stationary,
        "shift_runs": shifts,
    }

    timeline_rows = []
    for run in shifts:
        for row in run["timeline"]:
            timeline_rows.append(
                {
                    "threshold": run["threshold"],
                    "source_repo": run["source_repo"],
                    "target_repo": run["target_repo"],
                    "shift_at_step": run["shift_at_step"],
                    **row,
                }
            )
    timeline_path = Path(args.timeline_output)
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timeline_rows).to_csv(timeline_path, index=False)

    # The traces are in the CSV now. Leaving a copy in the JSON would make the report a
    # two-megabyte file that every dashboard page reparses to read six summary rows.
    for run in report["shift_runs"]:
        run.pop("timeline", None)
    report["timeline_export"] = str(timeline_path.as_posix())
    report["timeline_rows"] = len(timeline_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    for row in summary:
        logger.info(
            f"threshold {row['threshold']:5.1f}: "
            f"{row['false_alarms_total']:3d} false alarms over "
            f"{row['stationary_mutants_streamed']} stationary mutants, "
            f"{row['shifts_detected']}/{row['shifts_attempted']} shifts detected, "
            f"median latency {row['median_detection_latency_mutants']}"
        )
    logger.info(f"Report written to {out_path}; {len(timeline_rows)} trace rows to {timeline_path}")


if __name__ == "__main__":
    main()
