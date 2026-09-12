"""
ConfTest Streamlit Page 2: Confidence Calibration & Reliability Diagrams.

The reliability diagram is the measured one from reports/calibration_report.json.
It used to be drawn from two straight lines -- `uncal_acc = bins * 0.75 + 0.02`
and `cal_acc = bins * 0.96 + 0.01`, commented "Overconfident" and
"Well-calibrated" -- labelled with the real ECE values in the legend, on the page
whose entire subject is whether calibration worked. The report has carried the
measured per-bin confidence and accuracy all along.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from conftest.evaluation.headline import describe_difference
from dashboard.utils import (
    MissingArtifact,
    load_calibration_data,
    reliability_bins,
    stop_on_missing_artifact,
)

st.set_page_config(page_title="Confidence Calibration | ConfTest", page_icon="📉", layout="wide")

st.title("📉 Post-Hoc Confidence Calibration")
st.markdown(
    "Whether aligning raw ensemble scores with empirical failure rates measurably "
    "helps, judged on ECE, worst-case calibration (MCE) and the Brier score "
    "jointly -- each as a paired bootstrap difference against the uncalibrated model."
)

try:
    cal_data = load_calibration_data()
except MissingArtifact as exc:
    stop_on_missing_artifact(exc)

best_method = str(cal_data.get("best_method", "uncalibrated"))
selection = cal_data.get("selection", {})
metrics = {
    name: block
    for name, block in (cal_data.get("test_metrics") or {}).items()
    if isinstance(block, dict) and "ece" in block
}

# The decision, stated before any number, because the numbers alone read the
# other way round: a candidate here has the lowest ECE on the test split and was
# still rejected, since its paired interval spans zero.
if best_method == "uncalibrated":
    st.warning(
        f"🛡️ **No calibrator was fitted.** {selection.get('reason', '')}\n\n"
        f"Decided on **{selection.get('basis', 'unknown')}** evidence, resampling by "
        f"**{selection.get('resampling_unit', 'unknown')}** "
        f"({selection.get('num_bootstraps', 0)} resamples). The engine serves raw "
        "ensemble probabilities."
    )
else:
    temperature = cal_data.get("fitted_temperature")
    st.success(
        f"✅ **Serving {best_method}.**"
        + (f" Fitted temperature T = {temperature}." if temperature is not None else "")
    )

disqualified = selection.get("disqualified") or {}
if disqualified:
    st.subheader("🚫 Candidates rejected, and on what evidence")
    for method, reason in disqualified.items():
        st.markdown(f"- **{method}** — {reason}")

st.divider()

st.subheader("📐 Test-split calibration metrics, with paired intervals")
rows = []
for name, block in metrics.items():
    rows.append({
        "method": name,
        "ECE": block.get("ece"),
        "MCE": block.get("mce"),
        "Brier": block.get("brier_score"),
        "ECE vs uncalibrated": describe_difference(block.get("ece_vs_uncalibrated"), "ECE"),
        "MCE vs uncalibrated": describe_difference(block.get("mce_vs_uncalibrated"), "MCE"),
        "served": "yes" if name.startswith(best_method) else "",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "A lower ECE is not a gain until its paired difference excludes zero. On this "
    "split the lowest ECE belongs to a method whose difference does not."
)

st.divider()

st.subheader("📈 Measured reliability diagram")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration (y=x)",
    line=dict(dash="dash", color="#94a3b8"),
))

palette = {"uncalibrated": "#ef4444", "isotonic": "#f59e0b", "temperature_scaling": "#22c55e"}
plotted = 0
for name in (cal_data.get("reliability_diagram_bins") or {}):
    curve = reliability_bins(cal_data, name)
    if curve.empty:
        continue
    plotted += 1
    ece = (metrics.get(name) or metrics.get(f"{name}_calibration") or {}).get("ece")
    label = f"{name}" + (f" (ECE {ece:.4f})" if isinstance(ece, (int, float)) else "")
    fig.add_trace(go.Scatter(
        x=curve["confidence"],
        y=curve["accuracy"],
        mode="lines+markers",
        name=label,
        line=dict(color=palette.get(name), width=3 if name.startswith(best_method) else 2),
        marker=dict(size=(8 + 22 * curve["sample_count"] / curve["sample_count"].max())),
        customdata=curve[["sample_count", "bin_idx"]],
        hovertemplate=(
            "bin %{customdata[1]}<br>confidence %{x:.4f}"
            "<br>empirical failure rate %{y:.4f}<br>%{customdata[0]} samples<extra></extra>"
        ),
    ))

fig.update_layout(
    title="Empirical failure rate vs model confidence (marker area = samples in bin)",
    xaxis_title="Mean predicted confidence in bin",
    yaxis_title="Empirical failure rate in bin",
    height=480,
)
if plotted:
    st.plotly_chart(fig, use_container_width=True)
    occupancy = {
        name: int(reliability_bins(cal_data, name)["sample_count"].sum())
        for name in (cal_data.get("reliability_diagram_bins") or {})
        if not reliability_bins(cal_data, name).empty
    }
    st.caption(
        "Empty bins are dropped: a bin with no samples has no empirical failure "
        "rate, and plotting its zero would draw a measurement where there is none. "
        f"Occupied-bin sample totals: {occupancy}."
    )
else:
    st.info(
        "The report carries no reliability bins with samples in them. "
        "Re-run `python scripts/calibrate_model.py`."
    )

st.subheader("💡 Why Calibration Matters in CI Regression Selection")
st.markdown(r"""
- **Uncalibrated models** output scores that cannot be read as probabilities. A score of `0.80` might only fail 50% of the time, causing premature test omission.
- **Calibrated probabilities** let a risk budget be set directly: if the model says $\hat{p} = 0.10$, about $10\%$ of such tests should actually fail.
- **A calibration step is not free.** It is fitted on held-out data and can make worst-case miscalibration worse while improving the average, which is why MCE is judged alongside ECE and why a method is only adopted when its paired interval excludes zero.
""")
