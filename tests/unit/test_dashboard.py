import sys
from pathlib import Path

# Two packages in this repo are named `dashboard`: the live Streamlit app at
# ./dashboard (app.py, pages/, utils.py) and a legacy FastAPI stub at
# ./src/dashboard that has no utils module. pyproject sets pythonpath = ["src"],
# so `src` sits ahead of the repo root for the whole session and `import
# dashboard` resolves to the stub -- ModuleNotFoundError on dashboard.utils.
# A `not in sys.path` guard is not enough: the root IS present, just too late.
# Force it in front, so the resolution does not depend on which test ran first.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import types

import pandas as pd
import pytest

import dashboard.utils as utils
from dashboard.utils import (
    MissingArtifact,
    get_cached_engine,
    load_baseline_data,
    load_calibration_data,
    load_ensemble_metadata,
    load_policy_config,
    load_shap_report,
    load_uncertainty_analysis,
    reliability_bins,
    stop_on_missing_artifact,
)


def test_load_baseline_data_schema():
    """Verify load_baseline_data returns valid benchmark DataFrame."""
    df = load_baseline_data()
    assert not df.empty
    assert "strategy" in df.columns
    assert "failure_recall_pct" in df.columns
    assert "time_reduction_pct" in df.columns
    assert any("ConfTest" in s for s in df["strategy"])


def test_load_calibration_data_structure():
    """Verify load_calibration_data returns valid metrics dictionary."""
    data = load_calibration_data()
    assert "best_method" in data
    assert "test_metrics" in data
    assert "uncalibrated" in data["test_metrics"]
    assert "ece" in data["test_metrics"]["uncalibrated"]


def test_load_shap_report_structure():
    """Verify load_shap_report returns global SHAP feature list."""
    data = load_shap_report()
    assert "global_shap_importance" in data
    assert len(data["global_shap_importance"]) >= 5
    assert "feature" in data["global_shap_importance"][0]
    assert "mean_abs_shap" in data["global_shap_importance"][0]


def test_get_cached_engine_initialization():
    """
    Verify get_cached_engine returns functional ConfTestEngine.

    The engine must come up whether or not a calibrator was fitted. This
    previously asserted `engine.calibrator is not None`, which pinned the presence
    of `models/calibrator.joblib` rather than anything about the engine -- and the
    calibration selection now declines to fit one whenever no method's ECE gain
    clears the noise in the measurement, which is a legitimate outcome the engine
    handles by falling back to identity calibration.
    """
    engine = get_cached_engine()
    assert engine is not None
    assert engine.ensemble is not None, "the ensemble is what the engine cannot do without"
    assert engine.policy is not None, "and the abstention policy, which reads confidence"


# --------------------------------------------------------------------------
# Loaders against a controlled root
#
# The tests above read the repository's own artifacts, which is what makes them
# useful and also what makes them unable to test absence or a malformed cell.
# These point PROJECT_ROOT at a temporary tree instead.
# --------------------------------------------------------------------------

BASELINE_HEADERS = [
    "Strategy / Baseline",
    "Test Reduction (TRR %)",
    "Time Reduction (ETR %)",
    "Failure Recall (FR %)",
    "Missed-Failure (MFR %)",
    "Abstention Rate (AR %)",
    "Escaped Commits",
]


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_baseline_csv(root, rows):
    path = root / "reports" / "baseline_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BASELINE_HEADERS)
        writer.writerows(rows)
    return path


def test_percent_cells_are_parsed_to_numbers(fake_root):
    # Regression, pandas 3. The coercion used to be guarded on
    # `df[col].dtype == object`, which pandas 3 never satisfies for these
    # columns -- it infers StringDtype -- so the '%' was never stripped. Pages
    # then plotted strings on numeric axes, and Styler.highlight_max compared
    # them lexicographically: '20.0%' outranked '100.0%'.
    _write_baseline_csv(fake_root, [
        ["1. Retest-All", "0.0%", "0.0%", "100.0%", "0.0%", "0.0%", "0"],
        ["4. ConfTest (Calibrated + Selective)", "68.6%", "45.2%", "20.0%", "80.0%", "12.5%", "3"],
    ])

    df = load_baseline_data()

    assert pd.api.types.is_numeric_dtype(df["failure_recall_pct"])
    assert df["failure_recall_pct"].tolist() == [100.0, 20.0]
    # The lexicographic ranking bug, stated as the property it violated.
    assert df["failure_recall_pct"].idxmax() == 0
    assert df.loc[df["failure_recall_pct"].idxmax(), "strategy"] == "1. Retest-All"


def test_baseline_columns_are_renamed_for_the_pages(fake_root):
    _write_baseline_csv(fake_root, [
        ["4. ConfTest (Calibrated + Selective)", "68.6%", "45.2%", "40.0%", "60.0%", "12.5%", "3"],
    ])

    df = load_baseline_data()

    assert list(df.columns) == [
        "strategy",
        "test_reduction_pct",
        "time_reduction_pct",
        "failure_recall_pct",
        "missed_failure_pct",
        "abstention_rate_pct",
        "escaped_commits",
    ]
    # Never invented here: the old fallback carried a `safety_score` column the
    # real benchmark has never written.
    assert "safety_score" not in df.columns


def test_undefined_percent_cells_stay_undefined(fake_root):
    # errors="coerce", not fillna(0): a recall of 'n/a' means no commit had a
    # failure to recall, and 0.0 would render that as a total miss.
    _write_baseline_csv(fake_root, [
        ["4. ConfTest (Calibrated + Selective)", "68.6%", "n/a", "n/a", "n/a", "0.0%", "0"],
    ])

    df = load_baseline_data()

    assert pd.isna(df.loc[0, "failure_recall_pct"])
    assert pd.isna(df.loc[0, "time_reduction_pct"])
    assert df.loc[0, "test_reduction_pct"] == 68.6


def test_load_baseline_data_raises_instead_of_inventing_a_table(fake_root):
    with pytest.raises(MissingArtifact) as excinfo:
        load_baseline_data()

    assert excinfo.value.path.as_posix() == "reports/baseline_comparison.csv"
    assert excinfo.value.produced_by == "python scripts/train_baseline.py"


@pytest.mark.parametrize("loader, artifact, produced_by", [
    (load_calibration_data, "reports/calibration_report.json", "python scripts/calibrate_model.py"),
    (load_shap_report, "reports/explanations.json", "python scripts/generate_explanations.py"),
    (load_uncertainty_analysis, "reports/uncertainty_analysis.json",
     "python scripts/uncertainty_eval.py"),
    (load_ensemble_metadata, "models/ensembles/5_seed_lgbm/ensemble_metadata.json",
     "python scripts/train_ensemble.py"),
    (load_policy_config, "models/policy_config.json", "python scripts/tune_policy.py"),
])
def test_every_json_loader_names_the_script_behind_it(fake_root, loader, artifact, produced_by):
    with pytest.raises(MissingArtifact) as excinfo:
        loader()

    assert excinfo.value.path.as_posix() == artifact
    assert excinfo.value.produced_by == produced_by


# --------------------------------------------------------------------------
# Reliability curves and the missing-artifact page state
# --------------------------------------------------------------------------

def test_reliability_bins_drop_the_empty_ones():
    # An empty bin has no empirical failure rate. Plotting its zero draws a
    # measurement where none exists -- and on the current split only two of ten
    # bins are occupied, so this is most of the curve.
    report = {"reliability_diagram_bins": {"uncalibrated": [
        {"bin_lower": 0.0, "bin_upper": 0.1, "sample_count": 94, "empirical_rate": 0.021},
        {"bin_lower": 0.1, "bin_upper": 0.2, "sample_count": 0, "empirical_rate": 0.0},
        {"bin_lower": 0.9, "bin_upper": 1.0, "sample_count": 6, "empirical_rate": 0.833},
    ]}}

    df = reliability_bins(report, "uncalibrated")

    assert df["sample_count"].tolist() == [94, 6]
    # Reindexed, so a page that positions by row number does not skip a slot.
    assert df.index.tolist() == [0, 1]


def test_reliability_bins_are_empty_rather_than_absent():
    # Pages iterate the result; None would make every one of them crash on a
    # method the report happens not to carry.
    assert reliability_bins({}, "isotonic").empty
    assert reliability_bins({"reliability_diagram_bins": None}, "isotonic").empty
    assert reliability_bins({"reliability_diagram_bins": {"isotonic": []}}, "isotonic").empty


def test_reliability_bins_keep_a_curve_without_counts():
    # No sample_count column means nothing can be judged empty, so nothing is
    # dropped: silently returning an empty curve would look like a measurement
    # of zero occupied bins.
    report = {"reliability_diagram_bins": {"isotonic": [
        {"bin_lower": 0.0, "bin_upper": 0.5, "empirical_rate": 0.02},
    ]}}

    df = reliability_bins(report, "isotonic")

    assert len(df) == 1


def test_stop_on_missing_artifact_shows_the_command_and_stops(monkeypatch):
    calls = {"error": [], "stopped": 0}
    fake_streamlit = types.SimpleNamespace(
        error=lambda message: calls["error"].append(message),
        stop=lambda: calls.__setitem__("stopped", calls["stopped"] + 1),
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    stop_on_missing_artifact(
        MissingArtifact(
            Path("reports/calibration_report.json"), "python scripts/calibrate_model.py"
        )
    )

    assert calls["stopped"] == 1, "the page rendered on past the blocker"
    message = calls["error"][0]
    assert "reports/calibration_report.json" in message
    assert "python scripts/calibrate_model.py" in message
    assert "has none to show" in message
