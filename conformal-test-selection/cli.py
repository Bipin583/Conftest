"""Command-line interface for the conformal test-selection pipeline.

One entry point for every stage, in dependency order::

    python cli.py collect      # harvest CI history from GitHub Actions
    python cli.py features     # build the 30-feature matrix
    python cli.py preprocess   # impute, scale, encode, split by time
    python cli.py train        # fit the gradient-boosted model
    python cli.py calibrate    # Platt-scale the scores, report ECE
    python cli.py conformal    # fit the coverage-guaranteed threshold
    python cli.py predict      # select tests for a change
    python cli.py evaluate     # score the pipeline against config targets

``evaluate`` is the one that matters for the write-up: it compares the
conformal rule against fixed-threshold and cost-matched baselines on the held-out
split and writes ``reports/evaluation.json``, which is where the README's
measured column comes from. Nothing in this project writes a target number into
a results table.

Example:
    $ python cli.py evaluate --update-readme
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_logger, load_config, resolve_path, setup_logging  # noqa: E402

LOGGER = get_logger("cli")

#: Rows of the acceptance table, as (label, report key, target key, direction).
TARGET_ROWS = [
    ("Recall (failures caught)", "empirical_coverage", "recall", "ge"),
    ("Selection rate", "selection_rate", "selection_rate", "le"),
    ("Calibration error (ECE)", "ece", "ece", "le"),
    ("Conformal coverage", "empirical_coverage", "coverage", "ge"),
    ("Cost reduction", "cost_reduction", "cost_reduction", "ge"),
]


# --------------------------------------------------------------------------
# Stage commands
# --------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> int:
    """Harvest test outcomes from GitHub Actions.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from data.collect import main as collect_main

    forwarded: List[str] = []
    if args.repos:
        forwarded += ["--repos", args.repos]
    if args.max_repos is not None:
        forwarded += ["--max-repos", str(args.max_repos)]
    if args.output:
        forwarded += ["--output", args.output]
    return collect_main(forwarded)


def cmd_features(args: argparse.Namespace) -> int:
    """Build the feature matrix from harvested history.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from data.features import main as features_main

    forwarded: List[str] = []
    if args.source:
        forwarded += ["--source", args.source]
    if args.output:
        forwarded += ["--output", args.output]
    return features_main(forwarded)


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Impute, scale, encode and split into train/val/test.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from data.preprocess import main as preprocess_main

    forwarded: List[str] = []
    if args.source:
        forwarded += ["--source", args.source]
    if args.output_dir:
        forwarded += ["--output-dir", args.output_dir]
    if args.external is True:
        forwarded.append("--external")
    elif args.external is False:
        forwarded.append("--no-external")
    return preprocess_main(forwarded)


def cmd_train(args: argparse.Namespace) -> int:
    """Fit the gradient-boosted failure-risk model.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from models.train import main as train_main

    forwarded: List[str] = []
    if args.processed_dir:
        forwarded += ["--processed-dir", args.processed_dir]
    if args.model_path:
        forwarded += ["--model-path", args.model_path]
    return train_main(forwarded)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit Platt scaling and report ECE before and after.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from models.calibrate import main as calibrate_main

    forwarded: List[str] = []
    if args.processed_dir:
        forwarded += ["--processed-dir", args.processed_dir]
    if args.cv_crosscheck:
        forwarded.append("--cv-crosscheck")
    return calibrate_main(forwarded)


def cmd_conformal(args: argparse.Namespace) -> int:
    """Fit the conformal selection threshold.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from models.conformal import main as conformal_main

    forwarded: List[str] = []
    if args.processed_dir:
        forwarded += ["--processed-dir", args.processed_dir]
    if args.calibration_split:
        forwarded += ["--calibration-split", args.calibration_split]
    return conformal_main(forwarded)


def cmd_predict(args: argparse.Namespace) -> int:
    """Select tests for a change described in a JSON or CSV file.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    import pandas as pd

    from models.pipeline import PipelineError, SelectionPipeline, records_from_frame

    source = resolve_path(args.input)
    if not source.is_file():
        LOGGER.error("Input not found: %s", source)
        return 1

    try:
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            records = payload["tests"] if isinstance(payload, dict) and "tests" in payload else payload
            if isinstance(records, dict):
                records = [records]
        else:
            records = records_from_frame(pd.read_csv(source, low_memory=False))
    except (OSError, ValueError, KeyError) as exc:
        LOGGER.error("Could not read %s: %s", source, exc)
        return 1

    try:
        pipeline = SelectionPipeline.load()
        result = pipeline.predict(records, min_tests=args.min_tests, max_tests=args.max_tests)
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.output:
        target = resolve_path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOGGER.info("Wrote %s", target)
    else:
        summary = result["summary"]
        print(f"\nSelected {summary['n_selected']} of {summary['n_candidates']} tests "
              f"({summary['selection_rate']:.1%}), saving {summary['cost_reduction']:.1%} of CI cost.")
        print(f"Guarantee: {result['guarantee']['claim']}")
        if summary["degraded"]:
            print(f"WARNING: only {summary['feature_completeness']:.0%} of features supplied; "
                  "the rest were imputed.")
        print()
        for row in sorted(result["predictions"], key=lambda r: -r["failure_probability"])[: args.top]:
            mark = "RUN " if row["selected"] else "skip"
            print(f"  [{mark}] p={row['failure_probability']:.4f}  {row['test_id']}")
    return 0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _baselines(
    labels: np.ndarray,
    probabilities: np.ndarray,
    conformal_rate: float,
    val_labels: Optional[np.ndarray] = None,
    val_probabilities: Optional[np.ndarray] = None,
    target_recall: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Score reference strategies on the same split.

    Three comparisons, each answering a different objection:

    * ``run_all`` -- the status quo. Perfect recall, zero saving.
    * ``fixed_threshold_0.5`` -- what a practitioner does without conformal
      prediction: threshold the calibrated probability at one half.
    * ``cost_matched_topk`` -- selects the same *number* of tests as the
      conformal rule, ranked by probability. It reproduces the conformal result
      exactly, and that is not a null finding: both cut the same ranking, so it
      confirms conformal prediction is not a better ranker. What it cannot do is
      choose ``k`` without reading the test labels, which is precisely the
      information a deployed system does not have.

    * ``val_tuned_threshold`` -- the honest competitor, and the one the claim
      rests on. A practitioner without conformal prediction picks the lowest
      threshold meeting the target recall *on validation data*, then ships it.
      The gap between the recall it promised on validation and the recall it
      delivers on the future test split is the failure mode this project exists
      to remove.

    Args:
        labels: Ground-truth failure labels on the evaluation split.
        probabilities: Calibrated failure probabilities on that split.
        conformal_rate: Selection rate achieved by the conformal rule.
        val_labels: Validation labels, enabling the val-tuned baseline.
        val_probabilities: Validation probabilities.
        target_recall: Recall the val-tuned baseline aims for.

    Returns:
        A mapping of strategy name to its metrics.
    """
    n = labels.size
    n_failures = int((labels == 1).sum())

    def score(mask: np.ndarray) -> Dict[str, float]:
        chosen = int(mask.sum())
        caught = int(((labels == 1) & mask).sum())
        return {
            "recall": float(caught / n_failures) if n_failures else float("nan"),
            "selection_rate": float(chosen / n) if n else 0.0,
            "cost_reduction": float(1.0 - chosen / n) if n else 0.0,
            "precision": float(caught / chosen) if chosen else 0.0,
            "failures_missed": n_failures - caught,
        }

    k = int(round(conformal_rate * n))
    top_k = np.zeros(n, dtype=bool)
    if k > 0:
        top_k[np.argsort(probabilities)[::-1][:k]] = True

    results = {
        "run_all": score(np.ones(n, dtype=bool)),
        "fixed_threshold_0.5": score(probabilities >= 0.5),
        "cost_matched_topk": score(top_k),
    }

    if val_labels is not None and val_probabilities is not None:
        val_positive = np.asarray(val_probabilities)[np.asarray(val_labels) == 1]
        if val_positive.size:
            # Lowest threshold that still catches `target_recall` of validation
            # failures: the empirical quantile, with no finite-sample correction.
            # That missing correction is exactly what conformal prediction adds.
            cut = float(np.quantile(val_positive, 1.0 - target_recall))
            entry = score(probabilities >= cut)
            entry["threshold"] = round(cut, 6)
            entry["promised_recall_on_val"] = float(target_recall)
            entry["recall_shortfall"] = round(float(target_recall - entry["recall"]), 6)
            results["val_tuned_threshold"] = entry

    return results


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the fitted pipeline against the configured acceptance targets.

    Args:
        args: Parsed arguments.

    Returns:
        ``0`` if every target passed, ``1`` otherwise.
    """
    from models.calibrate import CalibrationError, load_calibrated_scorer
    from models.conformal import ConformalError, load_selector, selection_metrics
    from models.train import load_split

    cfg = load_config()
    targets = cfg.get("targets", {})

    try:
        base_model, calibrator, feature_columns = load_calibrated_scorer()
        selector = load_selector()
    except (CalibrationError, ConformalError) as exc:
        LOGGER.error("%s", exc)
        return 1

    X_test, y_test, ids_test = load_split("test", args.processed_dir)
    probabilities = calibrator.transform(
        base_model.predict_proba(X_test.loc[:, feature_columns])[:, 1]
    )

    group_column = cfg["data"].get("group_column", "commit_sha")
    groups = ids_test[group_column].to_numpy() if group_column in ids_test else None
    metrics = selection_metrics(y_test, probabilities, selector, groups=groups)

    # ECE comes from the calibration stage rather than being recomputed, so the
    # README quotes one number produced by one definition.
    ece_path = resolve_path(Path(cfg["artifacts"]["reports_dir"]) / "calibration_report.json")
    if ece_path.is_file():
        metrics["ece"] = json.loads(ece_path.read_text(encoding="utf-8"))["ece_after"]
    else:
        LOGGER.warning("No calibration report found; ECE will be reported as null.")
        metrics["ece"] = None

    X_val, y_val, _ = load_split("val", args.processed_dir)
    val_probabilities = calibrator.transform(
        base_model.predict_proba(X_val.loc[:, feature_columns])[:, 1]
    )
    baselines = _baselines(
        np.asarray(y_test),
        np.asarray(probabilities),
        metrics["selection_rate"],
        val_labels=np.asarray(y_val),
        val_probabilities=np.asarray(val_probabilities),
        target_recall=float(selector.coverage),
    )

    rows: List[Dict[str, Any]] = []
    all_passed = True
    for label, metric_key, target_key, direction in TARGET_ROWS:
        measured = metrics.get(metric_key)
        target = targets.get(target_key)
        if measured is None or target is None:
            rows.append({"metric": label, "target": target, "measured": measured, "passed": None})
            continue
        passed = measured >= target if direction == "ge" else measured <= target
        all_passed = all_passed and passed
        rows.append(
            {
                "metric": label,
                "target": float(target),
                "measured": float(measured),
                "direction": direction,
                "passed": bool(passed),
            }
        )

    report = {
        "split": "test",
        "n_rows": metrics["n_rows"],
        "n_failures": metrics["n_failures"],
        "guarantee": {
            "type": selector.guarantee,
            "target_coverage": selector.coverage,
            "confidence": selector.confidence,
            "certified_coverage": selector.certified_coverage,
            "probability_floor": selector.probability_floor,
            "calibration_size": selector.n_calibration,
        },
        "metrics": metrics,
        "acceptance": rows,
        "all_targets_passed": all_passed,
        "baselines": baselines,
    }

    destination = resolve_path(args.output or Path(cfg["artifacts"]["reports_dir"]) / "evaluation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Wrote %s", destination)

    _print_evaluation(report)

    if args.update_readme:
        _update_readme(report)
    return 0 if all_passed else 1


def _print_evaluation(report: Dict[str, Any]) -> None:
    """Render the evaluation report as a terminal table.

    Args:
        report: The report produced by :func:`cmd_evaluate`.
    """
    metrics = report["metrics"]
    print("\n" + "=" * 72)
    print(f"  Held-out evaluation: {report['n_rows']:,} rows, {report['n_failures']:,} failing")
    print("=" * 72)
    print(f"  {'Metric':<26}{'Target':>10}{'Measured':>12}{'Status':>10}")
    print("  " + "-" * 58)
    for row in report["acceptance"]:
        if row["measured"] is None:
            print(f"  {row['metric']:<26}{'-':>10}{'n/a':>12}{'SKIP':>10}")
            continue
        arrow = ">=" if row.get("direction") == "ge" else "<="
        status = "PASS" if row["passed"] else "FAIL"
        target = f"{arrow}{row['target']:.2f}"
        print(f"  {row['metric']:<26}{target:>10}{row['measured']:>12.4f}{status:>10}")
    print("  " + "-" * 58)

    guarantee = report["guarantee"]
    if guarantee["certified_coverage"] is not None:
        print(f"  Guarantee: {guarantee['type'].upper()} -- at least "
              f"{guarantee['certified_coverage']:.2%} coverage with "
              f"{guarantee['confidence']:.0%} confidence")
    print(f"  Per-commit: {metrics.get('commit_full_catch_rate', float('nan')):.2%} of failing commits "
          f"fully caught, {metrics.get('commit_any_catch_rate', float('nan')):.2%} partially")

    print("\n  Baselines on the same split:")
    print(f"  {'Strategy':<24}{'Recall':>10}{'Selected':>11}{'Saving':>10}")
    print("  " + "-" * 55)
    print(f"  {'conformal (this work)':<24}{metrics['empirical_coverage']:>10.4f}"
          f"{metrics['selection_rate']:>11.4f}{metrics['cost_reduction']:>10.4f}")
    for name, values in report["baselines"].items():
        print(f"  {name:<24}{values['recall']:>10.4f}{values['selection_rate']:>11.4f}"
              f"{values['cost_reduction']:>10.4f}")
    print("=" * 72)
    verdict = "ALL TARGETS MET" if report["all_targets_passed"] else "SOME TARGETS NOT MET"
    print(f"  {verdict}\n")


def _update_readme(report: Dict[str, Any], readme_path: Optional[Any] = None) -> None:
    """Write measured numbers into the README results table.

    The table lives between ``<!-- RESULTS:START -->`` and
    ``<!-- RESULTS:END -->``. Only that block is rewritten; if the markers are
    absent nothing is touched, because silently reformatting someone's README is
    worse than doing nothing.

    Args:
        report: The evaluation report.
        readme_path: Override for the README location.
    """
    path = resolve_path(readme_path or "README.md")
    if not path.is_file():
        LOGGER.warning("No README at %s; skipping update.", path)
        return

    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        LOGGER.warning("README lacks RESULTS markers; leaving it untouched.")
        return

    lines = [
        start,
        "",
        f"Measured on the held-out test split ({report['n_rows']:,} rows, "
        f"{report['n_failures']:,} failing) by `python cli.py evaluate`.",
        "",
        "| Metric | Target | Measured | Status |",
        "|---|---|---|---|",
    ]
    for row in report["acceptance"]:
        if row["measured"] is None:
            lines.append(f"| {row['metric']} | - | not measured | - |")
            continue
        arrow = "&ge;" if row.get("direction") == "ge" else "&le;"
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {row['metric']} | {arrow} {row['target']:.0%} | **{row['measured']:.2%}** | {status} |"
        )

    guarantee = report["guarantee"]
    if guarantee["certified_coverage"] is not None:
        lines += [
            "",
            f"Conformal guarantee in force: **{guarantee['type'].upper()}** -- with "
            f"{guarantee['confidence']:.0%} confidence over the calibration draw, at least "
            f"{guarantee['certified_coverage']:.2%} of failing tests are selected "
            f"(threshold: run a test when P(fail) &ge; {guarantee['probability_floor']:.4f}, "
            f"fitted on {guarantee['calibration_size']:,} calibration failures).",
        ]

    metrics = report["metrics"]
    lines += [
        "",
        "| Strategy | Recall | Selection rate | Cost saving |",
        "|---|---|---|---|",
        f"| **Conformal selection (this work)** | **{metrics['empirical_coverage']:.2%}** | "
        f"{metrics['selection_rate']:.2%} | {metrics['cost_reduction']:.2%} |",
    ]
    for name, values in report["baselines"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {values['recall']:.2%} | "
            f"{values['selection_rate']:.2%} | {values['cost_reduction']:.2%} |"
        )
    lines += ["", end]

    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    path.write_text(head + "\n".join(lines) + tail, encoding="utf-8")
    LOGGER.info("Updated the results table in %s", path)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every subcommand.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Confidence-calibrated test selection with conformal guarantees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical run:\n"
            "  python cli.py preprocess && python cli.py train && \\\n"
            "  python cli.py calibrate && python cli.py conformal && \\\n"
            "  python cli.py evaluate --update-readme\n"
        ),
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Harvest CI history from GitHub Actions.")
    p.add_argument("--repos", default=None, help="File listing owner/name per line.")
    p.add_argument("--max-repos", type=int, default=None, help="Stop after this many repositories.")
    p.add_argument("--output", default=None, help="Destination CSV.")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("features", help="Build the feature matrix.")
    p.add_argument("--source", default=None, help="Harvested history CSV.")
    p.add_argument("--output", default=None, help="Destination features CSV.")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("preprocess", help="Impute, scale, encode, and split by time.")
    p.add_argument("--source", default=None, help="Features CSV or external splits directory.")
    p.add_argument("--output-dir", default=None, help="Destination directory.")
    p.add_argument("--external", dest="external", action="store_true", help="Force the reuse path.")
    p.add_argument("--no-external", dest="external", action="store_false", help="Force the harvest path.")
    p.set_defaults(func=cmd_preprocess, external=None)

    p = sub.add_parser("train", help="Fit the gradient-boosted model.")
    p.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    p.add_argument("--model-path", default=None, help="Destination model path.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="Fit Platt scaling and report ECE.")
    p.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    p.add_argument("--cv-crosscheck", action="store_true", help="Also fit CalibratedClassifierCV.")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("conformal", help="Fit the coverage-guaranteed threshold.")
    p.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    p.add_argument("--calibration-split", default=None, choices=["val", "test"], help="Split for the quantile.")
    p.set_defaults(func=cmd_conformal)

    p = sub.add_parser("predict", help="Select tests for a change.")
    p.add_argument("input", help="JSON or CSV describing the candidate tests.")
    p.add_argument("--output", default=None, help="Write the full result as JSON.")
    p.add_argument("--min-tests", type=int, default=None, help="Override selection.min_tests.")
    p.add_argument("--max-tests", type=int, default=None, help="Override selection.max_tests.")
    p.add_argument("--top", type=int, default=20, help="Rows to print (default 20).")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="Score the pipeline against config targets.")
    p.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    p.add_argument("--output", default=None, help="Destination report path.")
    p.add_argument("--update-readme", action="store_true", help="Write measured numbers into README.md.")
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        The subcommand's exit code, or ``130`` if interrupted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        import os

        os.environ["CTS_CONFIG"] = args.config
    setup_logging(level="DEBUG" if args.verbose else None)

    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted.")
        return 130
    except Exception as exc:  # pragma: no cover - top-level safety net
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        LOGGER.info("Re-run with --verbose for the full traceback.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
