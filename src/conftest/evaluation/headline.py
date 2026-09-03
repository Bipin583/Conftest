"""
The numbers a dashboard puts on its front page, read from report artifacts.

Both dashboards used to hand-type these. The Streamlit portal claimed 100.0%
failure recall with "0 Escaped Bugs" and 68.6% test reduction; the FastAPI stub
claimed 45.2% time reduction, 99.4% recall, ECE 0.028 and $4,880/month. None of
those came from a measurement, and the project's own
reports/baseline_comparison.csv says the ConfTest selector recalled 40.0% of
failures and let 3 commits escape. A front page that contradicts the repository's
measured table is worse than one that shows nothing.

So every headline here is either read out of an artifact or reported as absent,
and each one carries the artifact it came from. `Headline.measured` is False when
the artifact does not exist yet; `source` then names the script that would
produce it, rather than a plausible-looking default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json

# Repo root, four parents up from src/conftest/evaluation/headline.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_CSV = Path("reports/baseline_comparison.csv")
CALIBRATION_JSON = Path("reports/calibration_report.json")
UNCERTAINTY_JSON = Path("reports/uncertainty_analysis.json")
POLICY_CONFIG = Path("models/policy_config.json")
SPLIT_METADATA = Path("data/splits/split_metadata.json")

# The sprint's definition of done, per MASTER_PLAN.md section 1: until this file
# exists no label in any report was produced by executing a test suite.
REAL_DATASET = Path("data/processed/real_features.csv")

# The strategy whose row the headline numbers are read from.
CONFTEST_MARKER = "ConfTest"

# Raw column headings in reports/baseline_comparison.csv, which the benchmark
# writes with the unit in the name and a '%' in every cell.
STRATEGY_COL = "Strategy / Baseline"
BASELINE_COLS = {
    "trr": "Test Reduction (TRR %)",
    "etr": "Time Reduction (ETR %)",
    "fr": "Failure Recall (FR %)",
    "mfr": "Missed-Failure (MFR %)",
    "ar": "Abstention Rate (AR %)",
    "escaped": "Escaped Commits",
}


class MissingArtifact(FileNotFoundError):
    """
    A report the caller asked for has not been produced yet.

    Raised instead of returning a stand-in, because a stand-in that carries the
    same shape as a measurement is indistinguishable from one at the call site.
    """

    def __init__(self, path: Path, produced_by: str):
        self.path = path
        self.produced_by = produced_by
        super().__init__(
            f"{path.as_posix()} does not exist. Produce it with: {produced_by}"
        )


@dataclass(frozen=True)
class Headline:
    """One front-page number, or the statement that it has not been measured."""

    label: str
    value: Optional[str]
    note: str
    source: str

    @property
    def measured(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class Provenance:
    """Whether the labels behind every reported number were measured or fabricated."""

    real_labels: bool
    detail: str


def _resolve(path: Path, root: Optional[Path] = None) -> Path:
    return path if path.is_absolute() else (root or PROJECT_ROOT) / path


def read_json(path: Path, produced_by: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Load a report, raising MissingArtifact rather than substituting defaults."""
    resolved = _resolve(path, root)
    if not resolved.exists():
        raise MissingArtifact(path, produced_by)
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_baseline_rows(root: Optional[Path] = None) -> List[Dict[str, str]]:
    """The benchmark comparison table, exactly as the runner wrote it."""
    resolved = _resolve(BASELINE_CSV, root)
    if not resolved.exists():
        raise MissingArtifact(BASELINE_CSV, "python scripts/train_baseline.py")
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_number(cell: Any) -> Optional[float]:
    """A percentage cell as a float. 'n/a' means undefined, and stays undefined."""
    if cell is None:
        return None
    text = str(cell).strip().rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def conftest_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """The ConfTest selector's row, which the headline claims are about."""
    for row in rows:
        if CONFTEST_MARKER in str(row.get(STRATEGY_COL, "")):
            return row
    raise ValueError(
        f"no strategy in {BASELINE_CSV.as_posix()} contains {CONFTEST_MARKER!r}; "
        f"found {[r.get(STRATEGY_COL) for r in rows]}"
    )


def calibration_block(report: Dict[str, Any], method: str) -> Dict[str, Any]:
    """
    One method's metrics out of a calibration report.

    The report keys some methods with a '_calibration' suffix and some without,
    so both spellings are tried rather than assuming either.
    """
    metrics = report.get("test_metrics", {})
    return metrics.get(f"{method}_calibration") or metrics.get(method) or {}


def describe_difference(diff: Optional[Dict[str, Any]], metric: str = "ECE") -> str:
    """
    A paired difference as a sentence, ending in what it decides.

    An interval that spans zero is the whole reason this project does not report
    its 25.47% ECE reduction as a gain: the paired difference behind that number
    is -0.00657 [-0.01119, +0.01102].
    """
    if not diff:
        return f"no paired interval on the {metric} difference"
    point, low, high = diff.get("point"), diff.get("ci_lower"), diff.get("ci_upper")
    if point is None or low is None or high is None:
        return f"no paired interval on the {metric} difference"
    verdict = "excludes zero" if diff.get("excludes_zero") else "spans zero, so not a gain"
    return f"{metric} difference {point:+.5f} [{low:+.5f}, {high:+.5f}] {verdict}"


def failure_recall_headline(root: Optional[Path] = None) -> Headline:
    """Recall of the failures that were available, and what escaped."""
    try:
        row = conftest_row(read_baseline_rows(root))
    except MissingArtifact as exc:
        return Headline("Failure Recall", None, "not measured", exc.produced_by)

    recall = as_number(row.get(BASELINE_COLS["fr"]))
    escaped = as_number(row.get(BASELINE_COLS["escaped"])) or 0.0
    missed = as_number(row.get(BASELINE_COLS["mfr"]))
    if recall is None:
        return Headline(
            "Failure Recall",
            None,
            "undefined: no commit in the split had a failing test to recall",
            BASELINE_CSV.as_posix(),
        )
    missed_text = "" if missed is None else f", missed-failure rate {missed:.1f}%"
    return Headline(
        "Failure Recall",
        f"{recall:.1f}%",
        f"{int(escaped)} commit(s) let a failure escape{missed_text}",
        BASELINE_CSV.as_posix(),
    )


def time_reduction_headline(root: Optional[Path] = None) -> Headline:
    """
    Measured wall-clock reduction, not the test-count ratio.

    These differ whenever durations are skewed, which they always are; the
    benchmark reports ETR from per-test durations for exactly that reason.
    """
    try:
        row = conftest_row(read_baseline_rows(root))
    except MissingArtifact as exc:
        return Headline("Test Time Reduction", None, "not measured", exc.produced_by)

    etr = as_number(row.get(BASELINE_COLS["etr"]))
    trr = as_number(row.get(BASELINE_COLS["trr"]))
    abstention = as_number(row.get(BASELINE_COLS["ar"]))
    parts = []
    if trr is not None:
        parts.append(f"test-count reduction {trr:.1f}%")
    if abstention is not None:
        parts.append(f"abstained on {abstention:.1f}% of commits")
    if etr is None:
        return Headline(
            "Test Time Reduction",
            None,
            "undefined: the dataset carries no per-test durations",
            BASELINE_CSV.as_posix(),
        )
    return Headline(
        "Test Time Reduction", f"{etr:.1f}%", "; ".join(parts), BASELINE_CSV.as_posix()
    )


def calibration_headline(root: Optional[Path] = None) -> Headline:
    """
    The ECE of the model actually being served, and why that model was chosen.

    When the selection declines to calibrate, the served model is the
    uncalibrated one and its ECE is the honest headline. Showing a rejected
    candidate's lower ECE instead would advertise a model that was not chosen.
    """
    try:
        report = read_json(CALIBRATION_JSON, "python scripts/calibrate_model.py", root)
    except MissingArtifact as exc:
        return Headline(
            "Expected Calibration Error", None, "not measured", exc.produced_by
        )

    best = str(report.get("best_method", "uncalibrated"))
    served = calibration_block(report, best)
    ece = served.get("ece")
    if ece is None:
        return Headline(
            "Expected Calibration Error",
            None,
            f"report names {best!r} but carries no ECE for it",
            CALIBRATION_JSON.as_posix(),
        )

    if best == "uncalibrated":
        reason = str(report.get("selection", {}).get("reason", "")).strip()
        note = f"served uncalibrated: {reason}" if reason else "served uncalibrated"
    else:
        note = f"served {best}: {describe_difference(served.get('ece_vs_uncalibrated'))}"
    return Headline(
        "Expected Calibration Error", f"{float(ece):.4f}", note, CALIBRATION_JSON.as_posix()
    )


def uncertainty_headline(root: Optional[Path] = None) -> Headline:
    """Ensemble disagreement, with the tail the abstention threshold sits in."""
    try:
        report = read_json(UNCERTAINTY_JSON, "python scripts/analyze_uncertainty.py", root)
    except MissingArtifact as exc:
        return Headline("Epistemic Disagreement", None, "not measured", exc.produced_by)

    mean = report.get("mean_epistemic_uncertainty")
    if mean is None:
        return Headline(
            "Epistemic Disagreement",
            None,
            "report carries no mean_epistemic_uncertainty",
            UNCERTAINTY_JSON.as_posix(),
        )
    p95 = report.get("p95_epistemic_uncertainty")
    samples = report.get("num_samples")
    parts = []
    if p95 is not None:
        parts.append(f"p95 {float(p95):.4f}")
    if samples is not None:
        parts.append(f"over {int(samples)} samples")
    return Headline(
        "Epistemic Disagreement",
        f"{float(mean):.4f}",
        " ".join(parts) or "mean ensemble standard deviation",
        UNCERTAINTY_JSON.as_posix(),
    )


def headline_metrics(root: Optional[Path] = None) -> List[Headline]:
    """The front-page row, in display order."""
    return [
        failure_recall_headline(root),
        time_reduction_headline(root),
        calibration_headline(root),
        uncertainty_headline(root),
    ]


def provenance(root: Optional[Path] = None) -> Provenance:
    """
    Whether the labels behind every number above were measured or fabricated.

    A dashboard that renders real artifacts is still misleading while those
    artifacts were themselves computed on sampled labels, so the answer travels
    with them.
    """
    if _resolve(REAL_DATASET, root).exists():
        return Provenance(
            True,
            f"labels measured by executing test suites; source {REAL_DATASET.as_posix()}",
        )
    detail = (
        "no test suite was executed to produce these labels: "
        f"{REAL_DATASET.as_posix()} does not exist yet, so every number above "
        "derives from the sampled-label splits in data/splits/. Build the real "
        "dataset with: python scripts/build_real_dataset.py --all"
    )
    meta_path = _resolve(SPLIT_METADATA, root)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            test = meta.get("test", {})
            detail += (
                f" (current test split: {test.get('num_commits')} commits, "
                f"{test.get('positive_failure_samples')} failing samples)"
            )
        except (ValueError, OSError):
            pass
    return Provenance(False, detail)
