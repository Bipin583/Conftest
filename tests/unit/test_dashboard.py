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

from dashboard.utils import load_baseline_data, load_calibration_data, load_shap_report, get_cached_engine


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
