"""
ConfTest Streamlit Page 4: RTS Baseline Benchmark Comparisons.

The closing finding is derived from the table above it. It used to be a fixed
sentence -- "ConfTest is the only selective RTS system achieving 100.0% Failure
Recall while delivering 68.6% test execution reduction, eliminating regression
escapes" -- printed regardless of what the CSV said. The CSV says 40.0% recall
and 3 escaped commits.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    MissingArtifact,
    load_baseline_data,
    provenance,
    stop_on_missing_artifact,
)

st.set_page_config(page_title="RTS Baseline Comparison | ConfTest", page_icon="📊", layout="wide")

st.title("📊 RTS Baseline Comparison Benchmark")
st.markdown("Comparative evaluation across **8 Regression Test Selection strategies** under budget-matched testing constraints.")

prov = provenance(PROJECT_ROOT)
if not prov.real_labels:
    st.error(f"⚠️ **Labels not measured.** {prov.detail}")

try:
    df = load_baseline_data()
except MissingArtifact as exc:
    stop_on_missing_artifact(exc)

highlight_cols = [
    c for c in ("failure_recall_pct", "time_reduction_pct") if c in df.columns
]
st.dataframe(
    df.style.highlight_max(subset=highlight_cols, color="#1e3a8a") if highlight_cols else df,
    use_container_width=True,
)
st.caption(
    "Each column is highlighted independently, so the highlights do not identify a "
    "best strategy: the largest time reduction here belongs to a strategy that "
    "recalled no failures at all. The two columns have to be read together."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Failure Detection Recall (%)")
    fig_rec = px.bar(
        df,
        x="failure_recall_pct",
        y="strategy",
        orientation="h",
        color="strategy",
        title="Failure recall across strategies",
        labels={"failure_recall_pct": "Recall (%)", "strategy": "RTS Strategy"},
    )
    fig_rec.add_vline(x=100.0, line_dash="dash", line_color="green")
    fig_rec.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_rec, use_container_width=True)

with col2:
    st.subheader("⚡ Compute Time Reduction (%)")
    fig_time = px.bar(
        df,
        x="time_reduction_pct",
        y="strategy",
        orientation="h",
        color="strategy",
        title="Test execution time reduction across strategies",
        labels={"time_reduction_pct": "Time Saved (%)", "strategy": "RTS Strategy"},
    )
    fig_time.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_time, use_container_width=True)

# The finding, read off the table. Anything stated here has to survive a rerun
# on different data, so it is phrased from the numbers rather than about them.
selective = df[~df["strategy"].str.contains("Full", case=False, na=False)]
ours = df[df["strategy"].str.contains("ConfTest", na=False)]

if ours.empty or selective.empty:
    st.info("No ConfTest row in the benchmark table, so there is no comparison to state.")
else:
    row = ours.iloc[0]
    recall = row.get("failure_recall_pct")
    etr = row.get("time_reduction_pct")
    escaped = row.get("escaped_commits")
    rivals = selective[~selective["strategy"].str.contains("ConfTest", na=False)]
    best_rival = (
        rivals.loc[rivals["failure_recall_pct"].idxmax()]
        if not rivals.empty and rivals["failure_recall_pct"].notna().any()
        else None
    )

    lines = [
        f"Among the selective strategies, **{row['strategy']}** recalled "
        f"**{recall:.1f}%** of available failures at **{etr:.1f}%** measured time reduction."
    ]
    if best_rival is not None:
        verb = "ahead of" if recall > best_rival["failure_recall_pct"] else "behind"
        lines.append(
            f"That is {verb} the next-best selective baseline, "
            f"*{best_rival['strategy']}* at {best_rival['failure_recall_pct']:.1f}%."
        )
    if pd.notna(escaped) and float(escaped) > 0:
        lines.append(
            f"It still let **{int(escaped)} commit(s)** ship with an undetected failure, "
            "so the abstention policy is not yet buying a zero-escape guarantee on this split."
        )
    elif pd.notna(escaped):
        lines.append("No commit in this split shipped with an undetected failure.")

    st.info("🏁 **Finding (read from the table above):** " + " ".join(lines))
