"""
ConfTest Streamlit Analytics Portal - Main Application Entrypoint.

The KPI row is read from report artifacts by conftest.evaluation.headline. It
used to be four literals -- 100.0% recall with "0 Escaped Bugs", 68.6% test
reduction, ECE 0.0192 at "-25.47% Error (Calibrated)", disagreement 0.0193 --
none of which came from a measurement, and the first two of which the project's
own reports/baseline_comparison.csv contradicts: the ConfTest selector recalls
40.0% of failures and lets 3 commits escape.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    MissingArtifact,
    headline_metrics,
    load_baseline_data,
    load_calibration_data,
    provenance,
    stop_on_missing_artifact,
)

st.set_page_config(
    page_title="ConfTest | Intelligent Regression Test Selection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ ConfTest: Confidence-Calibrated RTS Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Selective Prediction & Uncertainty-Aware Regression Test Selection for CI/CD</div>', unsafe_allow_html=True)

# Provenance first: a real artifact computed on sampled labels is still not
# evidence about the technique, and the reader has to be told which they have.
prov = provenance(PROJECT_ROOT)
if prov.real_labels:
    st.success(f"✅ {prov.detail}")
else:
    st.error(f"⚠️ **Labels not measured.** {prov.detail}")

# Top KPI Row, every cell read from the artifact named beneath it.
for column, headline in zip(st.columns(4), headline_metrics(PROJECT_ROOT)):
    with column:
        st.metric(label=headline.label, value=headline.value or "not measured")
        st.caption(headline.note)
        st.caption(f"source: `{headline.source}`")

st.divider()

# Main Overview Grid
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 RTS Baseline Comparison (Recall vs. Time Saved)")
    try:
        df_baselines = load_baseline_data()
    except MissingArtifact as exc:
        stop_on_missing_artifact(exc)
    fig = px.scatter(
        df_baselines,
        x="time_reduction_pct",
        y="failure_recall_pct",
        color="strategy",
        size=[24 if "ConfTest" in s else 14 for s in df_baselines["strategy"]],
        text="strategy",
        title="Strategy Trade-Off: Regression Safety vs Compute Efficiency",
        labels={"time_reduction_pct": "Test Execution Reduction (%)", "failure_recall_pct": "Bug Detection Recall (%)"},
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(
        y=100.0,
        line_dash="dash",
        line_color="green",
        annotation_text="100% recall (no escaped failure)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🧠 System Architecture & Pillars")
    try:
        calibration = load_calibration_data()
    except MissingArtifact as exc:
        stop_on_missing_artifact(exc)
    chosen = str(calibration.get("best_method", "uncalibrated"))
    selection = calibration.get("selection", {})
    if chosen == "uncalibrated":
        calibration_line = (
            "**Declined on the current split.** "
            f"{selection.get('reason', 'no candidate cleared the noise in the measurement')} "
            f"Decided on {selection.get('basis', 'unknown')} evidence, "
            f"resampling by {selection.get('resampling_unit', 'unknown')}."
        )
    else:
        temperature = calibration.get("fitted_temperature")
        fitted = f" Fitted $T = {temperature}$." if temperature is not None else ""
        calibration_line = f"**{chosen}**, chosen over the uncalibrated model.{fitted}"

    st.markdown(f"""
    **ConfTest** prevents silent CI regression escapes using a four-stage pipeline:

    1. **32-Feature Extraction Pipeline**:
       - 12 Churn & Diff Metrics
       - 6 AST Semantic & Cyclomatic Metrics
       - 6 Static Dependency-Graph Reachability Hops
       - 8 Historical Failure Telemetry Metrics (Strict Anti-Leakage)

    2. **5-Seed Deep Ensemble**:
       - Quantifies epistemic model uncertainty $\sigma(c, t) = \text{{Std}}(\{{p_m\}})$.

    3. **Post-Hoc Calibration**:
       - {calibration_line}

    4. **Dual-Mode Selective Policy**:
       - `FAST_SELECTED`: Confident test ranking.
       - `SAFE_FULL_SUITE`: 100% full fallback on high uncertainty or OOD diffs.
    """)

st.info("💡 **Navigate to the sidebar pages** to run live PR evaluations, inspect calibration curves, drill into epistemic uncertainty, or explore SHAP feature explanations.")
