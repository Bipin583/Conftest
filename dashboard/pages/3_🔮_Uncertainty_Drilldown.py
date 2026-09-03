"""
ConfTest Streamlit Page 3: Ensemble Uncertainty Drilldown.

Both figures on this page are read from reports/uncertainty_analysis.json. The
risk-coverage curve used to be `error_rate = 0.045 * (coverage ** 2.2)` -- a
closed-form curve standing in for the central claim of selective prediction,
that error falls on the subset the model is confident about. The scatter beneath
it drew 150 points from `np.random.beta(0.5, 5.0)` and `np.random.exponential`
and coloured them by the abstention threshold, which demonstrates only that the
threshold can be compared to a number. The report carries the measured curve.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    MissingArtifact,
    load_ensemble_metadata,
    load_policy_config,
    load_uncertainty_analysis,
    stop_on_missing_artifact,
)

st.set_page_config(page_title="Uncertainty Drilldown | ConfTest", page_icon="🔮", layout="wide")

st.title("🔮 Epistemic Uncertainty & Ensemble Disagreement")

try:
    analysis = load_uncertainty_analysis()
    policy = load_policy_config()
    ensemble = load_ensemble_metadata()
except MissingArtifact as exc:
    stop_on_missing_artifact(exc)

seeds = ensemble.get("seeds") or []
num_members = ensemble.get("num_members", len(seeds))
tau_abstain = policy.get("tau_abstain")
num_samples = analysis.get("num_samples")

st.markdown(
    f"Model doubt across **{num_members} random seeds** `{seeds}`, measured over "
    f"{num_samples} held-out samples."
)

curve = pd.DataFrame(analysis.get("risk_coverage_curve") or [])
if not curve.empty:
    curve["coverage"] = pd.to_numeric(
        curve["coverage_pct"].astype(str).str.rstrip("%"), errors="coerce"
    )
    curve = curve.sort_values("coverage").reset_index(drop=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ensemble Members", f"{num_members} models")
    st.caption(f"seeds {seeds} · source `models/ensembles/5_seed_lgbm/ensemble_metadata.json`")
with col2:
    st.metric(
        "Abstention Threshold (tau)",
        "not set" if tau_abstain is None else f"{float(tau_abstain):.4f}",
    )
    st.caption("source `models/policy_config.json`")
with col3:
    # Read off the measured curve rather than restated: this was "-34.1%", and
    # the artifact says -33.9%.
    full = curve[curve["coverage"] == 100.0]["mean_prediction_error"]
    at90 = curve[curve["coverage"] == 90.0]["mean_prediction_error"]
    if len(full) and len(at90) and float(full.iloc[0]):
        change = (float(at90.iloc[0]) - float(full.iloc[0])) / float(full.iloc[0]) * 100.0
        st.metric("Error change @ 90% coverage", f"{change:+.1f}%")
        st.caption(
            f"mean prediction error {float(full.iloc[0]):.4f} at full coverage → "
            f"{float(at90.iloc[0]):.4f} at 90% · source `reports/uncertainty_analysis.json`"
        )
    else:
        st.metric("Error change @ 90% coverage", "not measured")
        st.caption("the report carries no 100% and 90% coverage pair")

st.divider()

st.subheader("📉 Risk-Coverage Curve (measured)")
if curve.empty:
    st.info(
        "The report carries no risk-coverage curve. Re-run "
        "`python scripts/uncertainty_eval.py`."
    )
else:
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=curve["coverage"],
        y=curve["mean_prediction_error"],
        mode="lines+markers",
        name="Mean prediction error",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=9),
        customdata=curve[["retained_samples", "max_retained_uncertainty"]],
        hovertemplate=(
            "coverage %{x:.0f}%<br>mean error %{y:.4f}"
            "<br>%{customdata[0]} samples retained"
            "<br>max retained sigma %{customdata[1]:.4f}<extra></extra>"
        ),
    ))
    fig_rc.update_layout(
        title="Abstaining on the least certain samples, and what the error does",
        xaxis_title="Coverage / retention ratio (%)",
        yaxis_title="Mean prediction error",
        height=420,
    )
    st.plotly_chart(fig_rc, use_container_width=True)
    st.caption(
        f"{len(curve)} coverage levels over {num_samples} samples. A downward slope "
        "left-to-right is the property selective prediction needs: the samples "
        "dropped first are the ones the ensemble disagrees most about."
    )

    st.subheader("🎚️ Where the abstention threshold cuts")
    fig_tau = go.Figure()
    fig_tau.add_trace(go.Scatter(
        x=curve["coverage"],
        y=curve["max_retained_uncertainty"],
        mode="lines+markers",
        name="Largest sigma still retained",
        line=dict(color="#a855f7", width=3),
        marker=dict(size=9),
    ))
    if tau_abstain is not None:
        fig_tau.add_hline(
            y=float(tau_abstain),
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"tau_abstain = {float(tau_abstain):.4f}",
        )
    fig_tau.update_layout(
        title="Epistemic disagreement of the most uncertain retained sample, by coverage",
        xaxis_title="Coverage / retention ratio (%)",
        yaxis_title="Max retained epistemic uncertainty (sigma)",
        height=420,
    )
    st.plotly_chart(fig_tau, use_container_width=True)
    st.caption(
        "Read the threshold line against the curve: the coverage at which the curve "
        "crosses tau is the fraction of samples the policy would keep in FAST_SELECTED mode."
    )

st.divider()

st.subheader("📊 Measured disagreement statistics")
stats = {
    "mean epistemic uncertainty": analysis.get("mean_epistemic_uncertainty"),
    "p95 epistemic uncertainty": analysis.get("p95_epistemic_uncertainty"),
    "mean predictive entropy": analysis.get("mean_predictive_entropy"),
    "uncertainty / error correlation": analysis.get("uncertainty_error_correlation"),
    "samples": analysis.get("num_samples"),
}
st.dataframe(
    pd.DataFrame([{"statistic": k, "value": v} for k, v in stats.items()]),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "The correlation is the one number that says whether the uncertainty estimate "
    "is worth abstaining on: near zero, abstention drops samples at random."
)
