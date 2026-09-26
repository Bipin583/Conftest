"""Advisory conformal-selection report for CI (non-gating).

Run from the ``conformal-test-selection/`` directory. Loads the shipped fitted
artefacts and runs ``predict`` on the *real PR* (features extracted live from the
diff by :mod:`data.live_extract`), falling back to a committed sample when no PR
diff/tests are available. It surfaces the committed held-out metrics and the PAC
coverage guarantee, and flags that live predictions on this repo are
out-of-distribution. Writes a Markdown comment to ``$COMMENT_PATH`` (and to
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


def _candidates() -> tuple:
    """Return ``(records, source_label, changed_files, is_live)`` for scoring.

    Prefers features extracted from the *real* PR (env ``BASE_SHA``/``HEAD_SHA``/
    ``REPO_NAME``/``GITHUB_WORKSPACE``). Falls back to the committed sample when
    there is no diff, no changed source, or no discoverable test — so the job
    runs identically in a ``workflow_dispatch`` or a bare checkout. Extraction is
    already crash-proof, but the import and env plumbing are guarded here too:
    the advisory step must never go red.
    """
    try:
        from data.live_extract import build_live_candidates

        repo_root = os.environ.get("GITHUB_WORKSPACE") or str(ROOT.parent)
        base_sha = os.environ.get("BASE_SHA") or None
        head_sha = os.environ.get("HEAD_SHA") or None
        repo_name = os.environ.get("REPO_NAME") or "final-year-project"
        records = build_live_candidates(repo_root, base_sha, head_sha, repo_name, max_tests=500)
        if records:
            changed = sorted({r.get("changed_file_path", "") for r in records if r.get("changed_file_path")})
            return records, "live", changed, True
    except Exception:  # noqa: BLE001 - fall back to the sample rather than fail
        pass
    return _load_json(SAMPLE), "sample", [], False


def build_comment() -> str:
    """Compose the advisory Markdown, or an honest note if artefacts are absent."""
    lines = ["## 🔮 Conformal Test Selection — advisory (no tests were skipped)", ""]

    # Selection on the shipped artefacts, over the real PR when one is present.
    try:
        from models.pipeline import SelectionPipeline

        records, source, changed_files, is_live = _candidates()
        pipeline = SelectionPipeline.load()
        # min_tests=1 shows the pure conformal rule rather than the smoke-set floor.
        result = pipeline.predict(records, min_tests=1)
        summary = result["summary"]
        guarantee = result["guarantee"]

        if is_live:
            header = (
                f"**Live on this PR** — features extracted from the real diff "
                f"({len(records)} discovered test{'s' if len(records) != 1 else ''} scored "
                f"against {len(changed_files)} changed source file"
                f"{'s' if len(changed_files) != 1 else ''}):"
            )
        else:
            header = (
                "**Demo** on `examples/sample_candidates.json` — no PR diff/tests found, so this "
                "shows the committed sample (the serving path, on the committed `.pkl` artefacts):"
            )
        lines += [
            header,
            "",
            f"- Selected **{summary['n_selected']} / {summary['n_candidates']}** "
            f"candidate tests — projected cost saving **{_pct(summary['cost_reduction'])}**.",
            f"- Mechanical feature completeness {_pct(summary['feature_completeness'])} "
            f"(`degraded={summary['degraded']}`). Guarantee holds: `{guarantee['holds']}`.",
        ]
        if is_live and changed_files:
            shown = ", ".join(f"`{p}`" for p in changed_files[:8])
            more = f" (+{len(changed_files) - 8} more)" if len(changed_files) > 8 else ""
            lines.append(f"- Changed source files: {shown}{more}.")
        lines += [
            "",
            "| Selected | P(fail) | Test |",
            "|:--:|--:|---|",
        ]
        ranked = sorted(result["predictions"], key=lambda p: -p["failure_probability"])
        TOP = 20
        for pred in ranked[:TOP]:
            mark = "✅ run" if pred["selected"] else "⏭️ skip"
            lines.append(f"| {mark} | {pred['failure_probability']:.4f} | `{pred['test_id']}` |")
        if len(ranked) > TOP:
            lines.append(f"| … | | _+{len(ranked) - TOP} more, ranked below the top {TOP}_ |")
        lines.append("")
        lines.append(
            f"> Selection rule: run a test when P(fail) ≥ "
            f"`{guarantee['probability_floor']:.6f}` "
            f"({guarantee['type'].upper()} threshold)."
        )
        lines.append("")
        if is_live:
            lines += [
                "> ⚠️ **Out-of-distribution — read as a *relative* ranking, not calibrated risk.** "
                "The model was trained on the ConfTest synthetic-mutation corpus. On this repo the "
                "`repo` category is unseen (all-zero one-hot), there is no `mutant_operator`, and the "
                "8 historical features are median-imputed — so the absolute `P(fail)` values and the "
                "PAC coverage guarantee below **do not transfer** to this repository. The ranking of "
                "which tests relate to the change is still informative; this job skips no tests.",
                "",
            ]
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
