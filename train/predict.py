# ============================================================================
# PREDICT.PY: Unified CLI Inference Tool
# Usage: python predict.py --message "fix: resolve deadlock" --failure_rate 0.35
#
# Two inference paths, chosen by what is actually on disk:
#   hybrid  - fine-tuned CodeBERT (text) + XGBoost (tabular), when
#             models/codebert_flakiness/ exists and torch/transformers import.
#   xgboost - tabular-only XGBoost (models/xgb_*_final.pkl), the default
#             everywhere else. The measured comparison (models/comparison_table.csv,
#             produced by phase5_evaluation.py) shows the tabular model alone
#             matches the hybrid (F1 0.556 vs 0.555), so the light path costs
#             nothing measurable and needs no GPU libraries.
# ============================================================================

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import xgboost as xgb  # noqa: F401  (import guard: fails loudly before models load)
import joblib
import json
import argparse
from pathlib import Path

print("="*60)
print("FLAKY TEST PREDICTION TOOL")
print("="*60)

# Anchored to this script's directory, not the working directory, so the tool
# runs from the repo root, from CI, or from anywhere else.
TRAIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = TRAIN_DIR / "models"

# PARSE ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--message", type=str, required=True, help="Commit message")
parser.add_argument("--body", type=str, default="", help="Commit body")
parser.add_argument("--failure_rate", type=float, default=0.0, help="Historical failure rate")
parser.add_argument("--test_complexity", type=int, default=0, help="Test complexity score")
parser.add_argument("--lines_added", type=int, default=0, help="Lines added")
parser.add_argument("--lines_deleted", type=int, default=0, help="Lines deleted")
parser.add_argument("--files_changed", type=int, default=0, help="Files changed")
parser.add_argument("--json", action="store_true", help="Output as JSON")
parser.add_argument("--model", type=str, default="auto", choices=["auto", "xgboost", "hybrid"],
                    help="Inference path: auto (detect), xgboost (CPU-only tabular, <1s), hybrid (CodeBERT + XGBoost)")
args = parser.parse_args()


def _codebert_available() -> bool:
    """True only if the fine-tuned checkpoint is on disk AND the deep-learning
    stack imports. Either half missing means the hybrid path cannot run, and
    silently degrading to it would crash (or, worse, guess) later."""
    if not (MODELS_DIR / "codebert_flakiness").is_dir():
        return False
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
        return True
    except ImportError:
        return False


def _require(paths, producer):
    """Refuse to guess: every artifact must exist, and the error names the
    command that produces it."""
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: missing model artifacts: {missing}", file=sys.stderr)
        print(f"Run `python {producer}` (in train/) to produce them.", file=sys.stderr)
        sys.exit(2)


# LOAD MODELS
if args.model == "auto":
    use_hybrid = _codebert_available()
elif args.model == "hybrid":
    if not _codebert_available():
        print("ERROR: --model hybrid requested but the CodeBERT checkpoint or torch/transformers is unavailable.", file=sys.stderr)
        sys.exit(2)
    use_hybrid = True
else:  # xgboost: always the lite tabular path, even when the hybrid stack is installed
    use_hybrid = False
print(f"\n📂 Loading models ({'hybrid CodeBERT + XGBoost' if use_hybrid else 'XGBoost tabular'})...")

codebert_score = None
codebert_logit = None

if use_hybrid:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "codebert_flakiness")
    model_cb = AutoModelForSequenceClassification.from_pretrained(
        MODELS_DIR / "codebert_flakiness", use_safetensors=True
    )
    model_cb.eval()
    model_cb.to(device)

    _require(
        [MODELS_DIR / "xgb_hybrid.pkl",
         MODELS_DIR / "xgb_calibrator_hybrid.pkl",
         MODELS_DIR / "xgb_features_hybrid.pkl",
         MODELS_DIR / "metadata.json"],
        producer="phase4_hybrid.py",
    )
    model_xgb = joblib.load(MODELS_DIR / "xgb_hybrid.pkl")
    calibrator = joblib.load(MODELS_DIR / "xgb_calibrator_hybrid.pkl")
    features = joblib.load(MODELS_DIR / "xgb_features_hybrid.pkl")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    threshold = metadata['best_threshold']

    print("\n🤖 Running CodeBERT...")
    text = args.message + " " + args.body
    enc = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_cb(**enc)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        codebert_score = float(probs[1])
        codebert_logit = float(np.log(codebert_score / (1 - codebert_score + 1e-10)))
    print(f"   CodeBERT score: {codebert_score:.4f}")
else:
    _require(
        [MODELS_DIR / "xgb_model_final.pkl",
         MODELS_DIR / "xgb_calibrator_final.pkl",
         MODELS_DIR / "xgb_features_final.pkl",
         MODELS_DIR / "xgb_metrics_final.json"],
        producer="phase2_xgboost.py",
    )
    model_xgb = joblib.load(MODELS_DIR / "xgb_model_final.pkl")
    calibrator = joblib.load(MODELS_DIR / "xgb_calibrator_final.pkl")
    features = joblib.load(MODELS_DIR / "xgb_features_final.pkl")
    with open(MODELS_DIR / "xgb_metrics_final.json") as f:
        metrics = json.load(f)
    threshold = metrics['threshold']
print(f"   ✅ Models loaded!")

# PREPARE FEATURES
print("\n📊 Preparing features...")
msg = args.message
feat_dict = {
    'lines_added': args.lines_added,
    'lines_deleted': args.lines_deleted,
    'files_changed': args.files_changed,
    'prior_failures': 0,
    'failure_rate': args.failure_rate,
    'avg_duration': 0,
    'test_complexity': args.test_complexity,
    'code_coverage': 0,
    'lines_of_code': 0,
    'cyclomatic_complexity': 0,
    'repo_id': 0,
    'language_id': 0,
    'msg_length': len(msg),
    'msg_words': len(msg.split()),
    'has_bug_keyword': 1 if any(kw in msg.lower() for kw in ['bug', 'fix', 'error', 'fail', 'crash', 'issue', 'problem', 'broken']) else 0,
    'has_test_keyword': 1 if any(kw in msg.lower() for kw in ['test', 'spec', 'fixture', 'mock', 'assert']) else 0,
    'has_ci_keyword': 1 if any(kw in msg.lower() for kw in ['ci', 'cd', 'pipeline', 'build', 'deploy', 'github', 'action']) else 0,
    'uppercase_ratio': sum(1 for c in msg if c.isupper()) / max(len(msg), 1),
    'exclamation_count': msg.count('!'),
    'question_count': msg.count('?'),
    'body_file_count': args.body.count('.py') + args.body.count('.java'),
}
if use_hybrid:
    feat_dict['codebert_score'] = codebert_score
    feat_dict['codebert_logit'] = codebert_logit

X = pd.DataFrame([feat_dict])[features]

# PREDICT
print("\n🎯 Predicting...")
score = model_xgb.predict_proba(X)[0, 1]
cal_score = calibrator.predict([score])[0]
pred = 1 if cal_score >= threshold else 0

# RISK TIER
if cal_score >= 0.7:
    risk = "HIGH"
elif cal_score >= 0.4:
    risk = "MEDIUM"
else:
    risk = "LOW"

model_path = "hybrid" if use_hybrid else "xgboost"

# OUTPUT
print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)

if args.json:
    result = {
        "probability": float(cal_score),
        "threshold": float(threshold),
        "prediction": "FLAKY" if pred == 1 else "STABLE",
        "risk_tier": risk,
        "recommendation": "Quarantine test and enable auto-retry" if pred == 1 else "Normal CI execution",
        "model_path": model_path,
    }
    print(json.dumps(result, indent=2))
else:
    print(f"\n📊 Flakiness Probability: {cal_score:.2%}")
    print(f"🎯 Decision Threshold: {threshold:.2f}")
    print(f"✅ Prediction: {'🚨 FLAKY TEST' if pred == 1 else '✅ STABLE TEST'}")
    print(f"⚠️  Risk Tier: {risk}")
    print(f"🧭 Model Path: {model_path}")
    print(f"\n💡 Recommendation: {'Quarantine test and enable auto-retry' if pred == 1 else 'Normal CI execution'}")

print("="*60)
