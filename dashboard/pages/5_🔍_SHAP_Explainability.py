"""
ConfTest Streamlit Page 5: SHAP & Model Explainability.

The category pie was drawn from a hand-typed dict whose labels claimed
12/6/6/8 features but whose lists held 5/4/4/5 entries -- and several of those
entries (ast_func_count, ast_cyclomatic_delta, ast_num_asserts,
dep_coupling_coefficient, hist_flakiness_score) are not feature names the
pipeline has ever produced. The pie plotted len() of those lists, so the slices
disagreed with their own labels. Both now come from the canonical
FEATURE_NAMES, which is the list the model is actually trained on.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from conftest.features.pipeline import FEATURE_NAMES
from dashboard.utils import MissingArtifact, load_shap_report, provenance, stop_on_missing_artifact

st.set_page_config(page_title="SHAP Explainability | ConfTest", page_icon="🔍", layout="wide")

# Prefix -> human label. The counts are never written down; they are len() of
# what the pipeline exports, so a feature added to FEATURE_NAMES moves the pie.
FEATURE_GROUPS = {
    "diff_": "Code Churn & Diff",
    "ast_": "AST Syntactic & Complexity",
    "dep_": "Dependency Graph",
    "hist_": "Historical Telemetry",
}

st.title("🔍 SHAP & Model Explainability")
st.markdown(
    r"Game-theoretic feature attributions ($\phi_i$) explaining why individual "
    "regression tests are prioritized or omitted."
)

prov = provenance(PROJECT_ROOT)
if not prov.real_labels:
    st.error(f"⚠️ **Labels not measured.** {prov.detail}")

try:
    shap_data = load_shap_report()
except MissingArtifact as exc:
    stop_on_missing_artifact(exc)

st.caption(
    f"Attributions computed on `{shap_data.get('dataset_file', 'unknown dataset')}` "
    f"using `{shap_data.get('model_file', 'unknown model')}`."
)

global_imp = shap_data.get("global_shap_importance") or []

if not global_imp:
    st.warning(
        "The explanation report carries no global importances. Re-run "
        "`python scripts/generate_explanations.py`."
    )
else:
    df_shap = pd.DataFrame(global_imp)

    st.subheader("🌐 Global SHAP Feature Importance Rankings")
    fig_shap = px.bar(
        df_shap,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        title=f"{len(df_shap)} ranked features (mean |SHAP value|)",
        labels={"mean_abs_shap": "Mean |SHAP| Attribution", "feature": "Feature"},
    )
    fig_shap.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption(
        f"The report records {len(df_shap)} of the {len(FEATURE_NAMES)} features in the "
        "schema, so this is the ranking it saved and not a ranking of the whole schema. "
        "Features it omits are not thereby uninfluential -- they are unreported."
    )

st.divider()

st.subheader(f"🧩 {len(FEATURE_NAMES)}-Feature Category Distribution")

ranked = {row["feature"] for row in global_imp if isinstance(row, dict)}
rows = []
for prefix, label in FEATURE_GROUPS.items():
    members = [f for f in FEATURE_NAMES if f.startswith(prefix)]
    rows.append({
        "Category": f"{label} ({len(members)})",
        "Feature Count": len(members),
        "Ranked in report": sum(1 for f in members if f in ranked),
    })

ungrouped = [f for f in FEATURE_NAMES if not f.startswith(tuple(FEATURE_GROUPS))]
if ungrouped:
    rows.append({
        "Category": f"Other ({len(ungrouped)})",
        "Feature Count": len(ungrouped),
        "Ranked in report": sum(1 for f in ungrouped if f in ranked),
    })

cat_counts = pd.DataFrame(rows)
assert int(cat_counts["Feature Count"].sum()) == len(FEATURE_NAMES)

col_pie, col_tbl = st.columns([2, 1])
with col_pie:
    fig_pie = px.pie(
        cat_counts,
        names="Category",
        values="Feature Count",
        title="Feature schema composition (from conftest.features.pipeline.FEATURE_NAMES)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)
with col_tbl:
    st.dataframe(cat_counts.set_index("Category"), use_container_width=True)

unranked = [f for f in FEATURE_NAMES if f not in ranked]
if unranked:
    with st.expander(f"{len(unranked)} feature(s) carry no recorded attribution"):
        st.write(", ".join(f"`{f}`" for f in unranked))

st.info(
    "💡 **TreeExplainer guarantee:** Shapley attributions satisfy the additivity axiom, "
    rf"$f(\mathbf{{x}}) = \phi_0 + \sum_{{i=1}}^{{{len(FEATURE_NAMES)}}} \phi_i$, so the "
    "attributions above sum exactly to each prediction's departure from the base rate. "
    "That is a property of the estimator, not a measurement of this model's accuracy."
)
