"""Advisory conformal-selection report for CI (non-gating).

Run from the ``conformal-test-selection/`` directory. Loads the shipped fitted
artefacts, runs a live ``predict`` on a small committed sample to prove the
serving path works in CI, and surfaces the committed held-out metrics and the
PAC coverage guarantee. Writes a Markdown comment to ``$COMMENT_PATH`` (and to
``$GITHUB_STEP_SUMMARY`` when present).

It **never** raises out to the caller: an advisory step must not turn a build
red, so every failure is caught and reported as text.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SAMPLE = HERE / "sample_candidates.json"
EVAL_REPORT = ROOT / "reports" / "evaluation.json"
CONFORMAL_REPORT = ROOT / "reports" / "conformal_report.json"


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_comment() -> str:
    """Compose the advisory Markdown, or an honest note if artefacts are absent."""
    lines = ["## 🔮 Conformal Test Selection — advisory (no tests were skipped)", ""]

    # Live selection on the shipped artefacts ------------------------------
    try:
        from models.pipeline import SelectionPipeline

        records = _load_json(SAMPLE)
        pipeline = SelectionPipeline.load()
        # min_tests=1 shows the pure conformal rule rather than the smoke-set floor.
        result = pipeline.predict(records, min_tests=1)
        summary = result["summary"]
        guarantee = result["guarantee"]

        lines += [
            f"**Live demo** on `examples/sample_candidates.json` "
            f"(the serving path, running here on the committed `.pkl` artefacts):",
            "",
            f"- Selected **{summary['n_selected']} / {summary['n_candidates']}** "
            f"candidate tests — projected cost saving **{_pct(summary['cost_reduction'])}**.",
            f"- Feature completeness {_pct(summary['feature_completeness'])} "
            f"(`degraded={summary['degraded']}`). Guarantee holds: `{guarantee['holds']}`.",
            "",
            "| Selected | P(fail) | Test |",
            "|:--:|--:|---|",
        ]
        for pred in sorted(result["predictions"], key=lambda p: -p["failure_probability"]):
            mark = "✅ run" if pred["selected"] else "⏭️ skip"
            lines.append(f"| {mark} | {pred['failure_probability']:.4f} | `{pred['test_id']}` |")
        lines.append("")
        lines.append(
            f"> Selection rule: run a test when P(fail) ≥ "
            f"`{guarantee['probability_floor']:.6f}` "
            f"({guarantee['type'].upper()} threshold)."
        )
        lines.append("")
    except Exception as exc:  # noqa: BLE001 - advisory must never crash the build
        lines += [f"> ⚠️ Live demo unavailable in this checkout: `{exc}`", ""]

    # Committed held-out measurement --------------------------------------
    try:
        report = _load_json(EVAL_REPORT)
        m = report["metrics"]
        g = report["guarantee"]
        lines += [
            f"**Held-out measurement** (committed `reports/evaluation.json`, "
            f"test split: {m['n_rows']:,} rows, {m['n_failures']:,} failing) — "
            f"all acceptance targets **{'PASS' if report.get('all_targets_passed') else 'FAIL'}**:",
            "",
            "| Metric | Measured |",
            "|---|--:|",
            f"| Recall (failures caught) | **{_pct(m['empirical_coverage'])}** |",
            f"| Selection rate | {_pct(m['selection_rate'])} |",
            f"| Cost reduction | **{_pct(m['cost_reduction'])}** |",
            f"| Calibration error (ECE) | {_pct(m['ece'])} |",
            f"| Conformal coverage | {_pct(m['empirical_coverage'])} |",
            "",
            f"**Guarantee (PAC):** with **{_pct(g['confidence'])}** confidence over the "
            f"calibration draw, at least **{_pct(g['certified_coverage'])}** of failing tests "
            f"are selected (threshold P(fail) ≥ `{g['probability_floor']:.6f}`, fitted on "
            f"{g['calibration_size']:,} calibration failures).",
            "",
            "**Scope, stated honestly:** the guarantee is *per failing test*. At the commit "
            f"level it fully catches **{_pct(m['commit_full_catch_rate'])}** of failing pushes "
            f"and catches at least one failing test in **{_pct(m['commit_any_catch_rate'])}**. "
            "It holds under *exchangeability* of the calibration and future rows; the split is "
            "deliberately temporal, so distribution drift can weaken it — monitor realized "
            "coverage rather than treating it as unconditional.",
            "",
        ]
    except Exception as exc:  # noqa: BLE001
        lines += [f"> ⚠️ Held-out metrics unavailable: `{exc}`", ""]

    lines += [
        "---",
        "*Advisory only — this job selects nothing and skips no tests. It reports what the "
        "conformal rule would do so its coverage can be watched on real PRs before it is "
        "trusted to gate. See `conformal-test-selection/CONFORMAL_IMPLEMENTATION_AUDIT.md`.*",
    ]
    return "\n".join(lines)


def main() -> int:
    comment = build_comment()

    comment_path = os.environ.get("COMMENT_PATH")
    if comment_path:
        try:
            Path(comment_path).write_text(comment, encoding="utf-8")
        except OSError as exc:
            print(f"Could not write comment file: {exc}", file=sys.stderr)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as handle:
                handle.write(comment + "\n")
        except OSError:
            pass

    # Echo for the CI log. Guard the encode: a non-UTF-8 console (Windows cp1252)
    # would otherwise raise on the emoji and turn an advisory step red.
    try:
        print(comment)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(comment.encode("utf-8", "replace") + b"\n")
    return 0  # advisory: always succeed


if __name__ == "__main__":
    raise SystemExit(main())
