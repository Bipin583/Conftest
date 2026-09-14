# ============================================================================
# APP.PY: Interactive Streamlit Dashboard
# Run: streamlit run app.py
#
# Same two inference paths as predict.py: the hybrid (CodeBERT + XGBoost) when
# models/codebert_flakiness/ exists and torch/transformers import, otherwise
# the tabular-only XGBoost -- the light path that CI and CPU deployments use.
# ============================================================================

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import inspect
from pathlib import Path

st.set_page_config(page_title="Flaky Test Detector", page_icon="🧪", layout="wide")

st.title("🧪 Hybrid Flaky Test Detection System")
st.markdown("**Multi-Modal Deep Learning & Gradient Boosting (CodeBERT + XGBoost)**")

# Anchored to this script's directory so the dashboard runs from anywhere.
TRAIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = TRAIN_DIR / "models"


def _codebert_available() -> bool:
    """The hybrid path needs the fine-tuned checkpoint AND the DL stack."""
    if not (MODELS_DIR / "codebert_flakiness").is_dir():
        return False
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
        return True
    except ImportError:
        return False


USE_HYBRID = _codebert_available()


# LOAD MODELS
@st.cache_resource
def load_models():
    if USE_HYBRID:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "codebert_flakiness")
        model_cb = AutoModelForSequenceClassification.from_pretrained(
            MODELS_DIR / "codebert_flakiness", use_safetensors=True
        )
        model_cb.eval()
        model_cb.to(device)
        model_xgb = joblib.load(MODELS_DIR / "xgb_hybrid.pkl")
        calibrator = joblib.load(MODELS_DIR / "xgb_calibrator_hybrid.pkl")
        features = joblib.load(MODELS_DIR / "xgb_features_hybrid.pkl")
        with open(MODELS_DIR / "metadata.json") as f:
            metadata = json.load(f)
        threshold = metadata['best_threshold']
        return tokenizer, model_cb, model_xgb, calibrator, features, threshold, device
    model_xgb = joblib.load(MODELS_DIR / "xgb_model_final.pkl")
    calibrator = joblib.load(MODELS_DIR / "xgb_calibrator_final.pkl")
    features = joblib.load(MODELS_DIR / "xgb_features_final.pkl")
    with open(MODELS_DIR / "xgb_metrics_final.json") as f:
        metrics = json.load(f)
    threshold = metrics['threshold']
    return None, None, model_xgb, calibrator, features, threshold, None


tokenizer, model_cb, model_xgb, calibrator, features, threshold, device = load_models()

# SIDEBAR
st.sidebar.header("⚙️ Test Configuration")
commit_message = st.sidebar.text_area("Commit Message", "fix: resolve deadlock in socket worker pool", height=100)
commit_body = st.sidebar.text_area("Commit Body (optional)", "")
failure_rate = st.sidebar.slider("Historical Failure Rate", 0.0, 1.0, 0.35, 0.05)
test_complexity = st.sidebar.slider("Test Complexity", 0, 100, 22)
lines_added = st.sidebar.slider("Lines Added", 0, 1000, 50)
lines_deleted = st.sidebar.slider("Lines Deleted", 0, 1000, 20)
files_changed = st.sidebar.slider("Files Changed", 0, 50, 5)

if not USE_HYBRID:
    st.sidebar.info(
        "CodeBERT checkpoint not found (or torch/transformers not installed): "
        "running the tabular XGBoost path. Run `python phase3_codebert.py` to "
        "fine-tune the text model for the hybrid path."
    )

# PREDICT BUTTON
if st.sidebar.button("🔮 Predict Flakiness", type="primary"):
    codebert_score = None
    codebert_logit = None

    # CODEBERT (hybrid path only)
    if USE_HYBRID:
        import torch

        text = commit_message + " " + commit_body
        enc = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_cb(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            codebert_score = float(probs[1])
            codebert_logit = float(np.log(codebert_score / (1 - codebert_score + 1e-10)))

    # FEATURES
    msg = commit_message
    feat_dict = {
        'lines_added': lines_added, 'lines_deleted': lines_deleted, 'files_changed': files_changed,
        'prior_failures': 0, 'failure_rate': failure_rate, 'avg_duration': 0,
        'test_complexity': test_complexity, 'code_coverage': 0, 'lines_of_code': 0,
        'cyclomatic_complexity': 0, 'repo_id': 0, 'language_id': 0,
        'msg_length': len(msg), 'msg_words': len(msg.split()),
        'has_bug_keyword': 1 if any(kw in msg.lower() for kw in ['bug', 'fix', 'error', 'fail', 'crash', 'issue', 'problem', 'broken']) else 0,
        'has_test_keyword': 1 if any(kw in msg.lower() for kw in ['test', 'spec', 'fixture', 'mock', 'assert']) else 0,
        'has_ci_keyword': 1 if any(kw in msg.lower() for kw in ['ci', 'cd', 'pipeline', 'build', 'deploy', 'github', 'action']) else 0,
        'uppercase_ratio': sum(1 for c in msg if c.isupper()) / max(len(msg), 1),
        'exclamation_count': msg.count('!'), 'question_count': msg.count('?'),
        'body_file_count': commit_body.count('.py') + commit_body.count('.java'),
    }
    if USE_HYBRID:
        feat_dict['codebert_score'] = codebert_score
        feat_dict['codebert_logit'] = codebert_logit
    X = pd.DataFrame([feat_dict])[features]

    # PREDICT
    score = model_xgb.predict_proba(X)[0, 1]
    cal_score = calibrator.predict([score])[0]
    pred = 1 if cal_score >= threshold else 0
    risk = "HIGH" if cal_score >= 0.7 else "MEDIUM" if cal_score >= 0.4 else "LOW"

    # DISPLAY
    st.header("📊 Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Flakiness Probability", f"{cal_score:.2%}")
    col2.metric("Decision Threshold", f"{threshold:.2f}")
    col3.metric("Risk Tier", risk)

    if pred == 1:
        st.error("🚨 **FLAKY TEST DETECTED**")
        st.info("**Recommendation:** Quarantine test and enable auto-retry")
    else:
        st.success("✅ **STABLE TEST**")
        st.info("**Recommendation:** Normal CI execution")

    # CODEBERT SCORE (hybrid path only)
    if USE_HYBRID:
        st.subheader("🤖 CodeBERT Analysis")
        progress_val = float(np.clip(codebert_score, 0.0, 1.0))
        st.progress(progress_val)
        st.write(f"Semantic flakiness score: {codebert_score:.2%}")

# SHOW METRICS
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance")
st.sidebar.write(f"**Model path:** {'Hybrid (CodeBERT + XGBoost)' if USE_HYBRID else 'XGBoost (tabular)'}")
st.sidebar.write("**F1 Score:** 55.50%")
st.sidebar.write("**AUC-ROC:** 60.49%")
st.sidebar.write("**Recall:** 87.74%")

# VISUALIZATIONS
st.markdown("---")
st.header("📊 Model Visualizations")

img_kwargs = {"use_container_width": True} if "use_container_width" in inspect.signature(st.image).parameters else {"use_column_width": True}

col1, col2 = st.columns(2)
with col1:
    st.image(str(TRAIN_DIR / "figures" / "metrics_comparison.png"), caption="Metrics Comparison", **img_kwargs)
with col2:
    st.image(str(TRAIN_DIR / "figures" / "roc_curves.png"), caption="ROC Curves", **img_kwargs)
