"""
ConfTest Streamlit Dashboard Utility Functions & Data Loaders.

Every loader here reads an artifact or raises. There used to be a fallback
behind each one -- a hand-typed DataFrame of eight strategies topped by
"ConfTest Selective (Ours)" at 100.0% recall, a calibration dict asserting
best_method "temperature_scaling" with ECE 0.0192 and a 25.47% reduction, and a
list of five SHAP importances. They returned the same shape as the real thing,
so no page could tell which it had been handed, and the calibration fallback
asserted the opposite of what the current report measures (best_method is
"uncalibrated": no candidate's ECE gain cleared the noise). Missing evidence is
now an error the page has to render, not a number it can print.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from conftest.engine.selector_engine import ConfTestEngine
from conftest.evaluation.headline import (
    Headline,
    MissingArtifact,
    Provenance,
    headline_metrics,
    provenance,
    read_json,
)
from conftest.db.session import SessionLocal
from conftest.db import crud

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw benchmark headings -> the snake_case names the pages plot.
BASELINE_RENAME = {
    "Strategy / Baseline": "strategy",
    "Test Reduction (TRR %)": "test_reduction_pct",
    "Time Reduction (ETR %)": "time_reduction_pct",
    "Failure Recall (FR %)": "failure_recall_pct",
    "Missed-Failure (MFR %)": "missed_failure_pct",
    "Abstention Rate (AR %)": "abstention_rate_pct",
    "Escaped Commits": "escaped_commits",
}
PERCENT_COLUMNS = (
    "test_reduction_pct",
    "time_reduction_pct",
    "failure_recall_pct",
    "missed_failure_pct",
    "abstention_rate_pct",
)

__all__ = [
    "Headline",
    "MissingArtifact",
    "Provenance",
    "get_cached_engine",
    "headline_metrics",
    "load_baseline_data",
    "load_calibration_data",
    "load_ensemble_metadata",
    "load_policy_config",
    "load_shap_report",
    "load_uncertainty_analysis",
    "provenance",
    "reliability_bins",
    "stop_on_missing_artifact",
]


def get_cached_engine() -> ConfTestEngine:
    """Instantiate ConfTestEngine pointed to sample suite."""
    return ConfTestEngine(
        repo_root=str(PROJECT_ROOT / "tests" / "sample_suite"),
        ensemble_path=str(PROJECT_ROOT / "models" / "ensembles" / "5_seed_lgbm"),
        calibrator_path=str(PROJECT_ROOT / "models" / "calibrator.joblib"),
        policy_config_path=str(PROJECT_ROOT / "models" / "policy_config.json"),
    )


def load_baseline_data() -> pd.DataFrame:
    """
    The RTS baseline comparison table, with percentage cells parsed to floats.

    Raises MissingArtifact when the benchmark has not been run. The old fallback
    invented a `safety_score` column that the real CSV has never carried, so a
    page written against the fallback silently lost a column on real data.
    """
    csv_path = PROJECT_ROOT / "reports" / "baseline_comparison.csv"
    if not csv_path.exists():
        raise MissingArtifact(
            Path("reports/baseline_comparison.csv"), "python scripts/train_baseline.py"
        )
    df = pd.read_csv(csv_path).rename(columns=BASELINE_RENAME)
    for col in PERCENT_COLUMNS:
        if col in df.columns:
            # Coerced unconditionally. This was guarded on `dtype == object`,
            # which pandas 3 never satisfies -- it infers StringDtype for these
            # columns -- so the '%' was never stripped and every page plotted
            # strings on a numeric axis. highlight_max then compared them
            # lexicographically and ranked '20.0%' above '100.0%'.
            df[col] = pd.to_numeric(
                df[col].astype(str).str.rstrip("%"), errors="coerce"
            )
    return df


def load_calibration_data() -> Dict[str, Any]:
    """The calibration report, including the selection block that chose the model."""
    return read_json(
        Path("reports/calibration_report.json"),
        "python scripts/calibrate_model.py",
        PROJECT_ROOT,
    )


def load_shap_report() -> Dict[str, Any]:
    """Global and per-test SHAP attributions."""
    return read_json(
        Path("reports/explanations.json"),
        "python scripts/explain_predictions.py",
        PROJECT_ROOT,
    )


def load_uncertainty_analysis() -> Dict[str, Any]:
    """Ensemble disagreement statistics and the measured risk-coverage curve."""
    return read_json(
        Path("reports/uncertainty_analysis.json"),
        "python scripts/analyze_uncertainty.py",
        PROJECT_ROOT,
    )


def load_ensemble_metadata() -> Dict[str, Any]:
    """The ensemble that produced the disagreement figures: seeds and member count."""
    return read_json(
        Path("models/ensembles/5_seed_lgbm/ensemble_metadata.json"),
        "python scripts/train_ensemble.py",
        PROJECT_ROOT,
    )


def load_policy_config() -> Dict[str, Any]:
    """The thresholds the abstention policy actually runs with."""
    return read_json(
        Path("models/policy_config.json"), "python scripts/tune_policy.py", PROJECT_ROOT
    )


def reliability_bins(report: Dict[str, Any], method: str) -> pd.DataFrame:
    """
    One method's measured reliability curve, with empty bins dropped.

    A bin holding no samples has no empirical failure rate; plotting its zero as
    a point on the curve draws a measurement where there is none. On the current
    split only two of ten bins are occupied, so this is most of the curve.
    """
    bins = (report.get("reliability_diagram_bins") or {}).get(method) or []
    df = pd.DataFrame(bins)
    if df.empty:
        return df
    if "sample_count" in df.columns:
        df = df[df["sample_count"] > 0]
    return df.reset_index(drop=True)


def stop_on_missing_artifact(exc: MissingArtifact) -> None:
    """Render a missing report as the blocker it is, and stop the page."""
    import streamlit as st

    st.error(
        f"**{exc.path.as_posix()} has not been produced yet.**\n\n"
        f"This page reports measurements and has none to show. Produce them with:\n\n"
        f"```\n{exc.produced_by}\n```"
    )
    st.stop()
