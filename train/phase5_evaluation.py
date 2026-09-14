# ============================================================================
# PHASE 5: FULL COMPARATIVE EVALUATION & VISUALIZATION
# Compares: XGBoost (Phase 2) vs CodeBERT (Phase 3) vs Hybrid (Phase 4)
# ============================================================================

import sys
# Configure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("="*60)
print("PHASE 5: COMPARATIVE EVALUATION")
print("="*60)

# LOAD DATASET
print("\n📂 Loading dataset...")
df = pd.read_csv("dataset_full_realistic.csv")
print(f"✅ Loaded {len(df):,} samples")

# SPLIT (same as training)
df = df.sort_values('commit_id').reset_index(drop=True)
n = len(df)
test_df = df.iloc[int(n*0.85):].copy().reset_index(drop=True)
y_test = test_df['label'].values
print(f"   Test samples: {len(test_df):,} | Positive labels: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# ============================================================================
# MODEL 1: PHASE 2 - XGBOOST (TABULAR ONLY)
# ============================================================================
print("\n" + "="*60)
print("MODEL 1: XGBOOST (TABULAR ONLY)")
print("="*60)

features_tabular = [
    'lines_added', 'lines_deleted', 'files_changed', 'prior_failures',
    'failure_rate', 'avg_duration', 'test_complexity', 'code_coverage',
    'lines_of_code', 'cyclomatic_complexity', 'repo_id', 'language_id'
]

X_test_tab = test_df[features_tabular]

# Load or recreate XGBoost for tabular
print("\n🤖 Loading/training XGBoost (tabular)...")
model_xgb_tab = None
for path in ["models/xgb_tabular_only.pkl", "models/xgb_model_final.pkl", "models/xgb_model_realistic.pkl"]:
    if Path(path).exists():
        model_xgb_tab = joblib.load(path)
        print(f"   ✅ Loaded existing model from {path}")
        break

if model_xgb_tab is None:
    print("   ⚠️  Training new model...")
    model_xgb_tab = xgb.XGBClassifier(
        n_estimators=600, max_depth=12, learning_rate=0.02, random_state=42,
        tree_method="hist", n_jobs=-1, scale_pos_weight=(df['label']==0).sum() / max((df['label']==1).sum(), 1)
    )
    train_df = df.iloc[:int(n*0.7)]
    X_train_tab = train_df[features_tabular]
    y_train = train_df['label']
    model_xgb_tab.fit(X_train_tab, y_train, verbose=False)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model_xgb_tab, "models/xgb_tabular_only.pkl")
    print("   ✅ Saved to models/xgb_tabular_only.pkl!")

# Predict
scores_xgb_tab = model_xgb_tab.predict_proba(X_test_tab)[:, 1]

# Calibrate
cal_xgb_tab = None
for path in ["models/calibrator_tabular.pkl", "models/xgb_calibrator_final.pkl", "models/xgb_calibrator_realistic.pkl"]:
    if Path(path).exists():
        cal_xgb_tab = joblib.load(path)
        print(f"   ✅ Loaded calibrator from {path}")
        break

if cal_xgb_tab is None:
    val_df = df.iloc[int(n*0.7):int(n*0.85)]
    X_val_tab = val_df[features_tabular]
    y_val = val_df['label']
    val_scores = model_xgb_tab.predict_proba(X_val_tab)[:, 1]
    cal_xgb_tab = IsotonicRegression(out_of_bounds="clip")
    cal_xgb_tab.fit(val_scores, y_val)
    joblib.dump(cal_xgb_tab, "models/calibrator_tabular.pkl")

scores_xgb_tab_cal = cal_xgb_tab.predict(scores_xgb_tab)

# Metrics (optimal threshold or 0.30 from Phase 2)
thresh_xgb = 0.30
pred_xgb_tab = (scores_xgb_tab_cal >= thresh_xgb).astype(int)
prec_xgb = precision_score(y_test, pred_xgb_tab, zero_division=0)
rec_xgb = recall_score(y_test, pred_xgb_tab, zero_division=0)
f1_xgb = f1_score(y_test, pred_xgb_tab, zero_division=0)
auc_xgb = roc_auc_score(y_test, scores_xgb_tab_cal)
brier_xgb = brier_score_loss(y_test, scores_xgb_tab_cal)

print(f"\n📊 XGBoost (Tabular - Threshold={thresh_xgb:.2f}):")
print(f"   Precision: {prec_xgb:.4f} ({prec_xgb*100:.2f}%)")
print(f"   Recall:    {rec_xgb:.4f} ({rec_xgb*100:.2f}%)")
print(f"   F1:        {f1_xgb:.4f} ({f1_xgb*100:.2f}%)")
print(f"   AUC-ROC:   {auc_xgb:.4f} ({auc_xgb*100:.2f}%)")
print(f"   Brier:     {brier_xgb:.4f}")

# ============================================================================
# MODEL 2: PHASE 3 - CODEBERT (TEXT ONLY)
# ============================================================================
print("\n" + "="*60)
print("MODEL 2: CODEBERT (TEXT ONLY)")
print("="*60)

print("\n🤖 Loading CodeBERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Inference device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

tokenizer = AutoTokenizer.from_pretrained("models/codebert_flakiness")
model_cb = AutoModelForSequenceClassification.from_pretrained("models/codebert_flakiness", use_safetensors=True)
model_cb.eval()
model_cb.to(device)
if torch.cuda.is_available():
    model_cb = model_cb.half()

print("\n📝 Extracting predictions on test set...")
start_cb = time.time()
# High-speed unique commit extraction on GPU
unique_test_commits = test_df[['commit_id', 'commit_message', 'commit_body']].drop_duplicates(subset=['commit_id']).copy()
unique_test_commits['text'] = (unique_test_commits['commit_message'].fillna('') + ' ' + unique_test_commits['commit_body'].fillna('')).astype(str)
test_commit_texts = unique_test_commits['text'].tolist()
test_commit_ids = unique_test_commits['commit_id'].tolist()
print(f"   Inference on {len(test_commit_texts):,} unique test commits across {len(test_df):,} test rows...")

all_scores_cb_unique = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(test_commit_texts), batch_size):
        batch = test_commit_texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        out = model_cb(**enc)
        probs = torch.softmax(out.logits, dim=1).float().cpu().numpy()
        all_scores_cb_unique.append(probs[:, 1])

scores_cb_unique = np.concatenate(all_scores_cb_unique)
cb_score_map = pd.DataFrame({'commit_id': test_commit_ids, 'codebert_score': scores_cb_unique})

# Map back to full test set
test_df = test_df.merge(cb_score_map, on='commit_id', how='left')
scores_cb = test_df['codebert_score'].values
print(f"   ✅ CodeBERT predictions generated in {time.time()-start_cb:.1f} seconds!")

# Metrics
thresh_cb = 0.50
pred_cb = (scores_cb >= thresh_cb).astype(int)
prec_cb = precision_score(y_test, pred_cb, zero_division=0)
rec_cb = recall_score(y_test, pred_cb, zero_division=0)
f1_cb = f1_score(y_test, pred_cb, zero_division=0)
auc_cb = roc_auc_score(y_test, scores_cb)
brier_cb = brier_score_loss(y_test, scores_cb)

print(f"\n📊 CodeBERT (Text - Threshold={thresh_cb:.2f}):")
print(f"   Precision: {prec_cb:.4f} ({prec_cb*100:.2f}%)")
print(f"   Recall:    {rec_cb:.4f} ({rec_cb*100:.2f}%)")
print(f"   F1:        {f1_cb:.4f} ({f1_cb*100:.2f}%)")
print(f"   AUC-ROC:   {auc_cb:.4f} ({auc_cb*100:.2f}%)")
print(f"   Brier:     {brier_cb:.4f}")

# ============================================================================
# MODEL 3: PHASE 4 - HYBRID (CODEBERT + XGBOOST)
# ============================================================================
print("\n" + "="*60)
print("MODEL 3: HYBRID (CODEBERT + XGBOOST)")
print("="*60)

print("\n🤖 Loading Hybrid model...")
model_hybrid = joblib.load("models/xgb_hybrid.pkl")
cal_hybrid = joblib.load("models/xgb_calibrator_hybrid.pkl")
features_hybrid = joblib.load("models/xgb_features_hybrid.pkl")

# Prepare features
print("⚙️  Preparing hybrid features for test set...")
test_df['codebert_logit'] = np.log(scores_cb / (1 - scores_cb + 1e-10))
test_df['msg_length'] = test_df['commit_message'].fillna('').str.len()
test_df['msg_words'] = test_df['commit_message'].fillna('').str.split().str.len()
bug_keywords = ['bug', 'fix', 'error', 'fail', 'crash', 'issue', 'problem', 'broken', 'defect']
test_df['has_bug_keyword'] = test_df['commit_message'].fillna('').str.lower().apply(lambda x: 1 if any(kw in x for kw in bug_keywords) else 0)
test_keywords = ['test', 'spec', 'fixture', 'mock', 'assert', 'flaky', 'check']
test_df['has_test_keyword'] = test_df['commit_message'].fillna('').str.lower().apply(lambda x: 1 if any(kw in x for kw in test_keywords) else 0)
ci_keywords = ['ci', 'cd', 'pipeline', 'build', 'deploy', 'github', 'action', 'travis', 'jenkins']
test_df['has_ci_keyword'] = test_df['commit_message'].fillna('').str.lower().apply(lambda x: 1 if any(kw in x for kw in ci_keywords) else 0)
test_df['uppercase_ratio'] = test_df['commit_message'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
test_df['exclamation_count'] = test_df['commit_message'].fillna('').str.count('!')
test_df['question_count'] = test_df['commit_message'].fillna('').str.count(r'\?')
test_df['body_file_count'] = test_df['commit_body'].fillna('').str.count(r'\.[a-zA-Z0-9]+').fillna(0)

X_test_hyb = test_df[features_hybrid]
scores_hyb = model_hybrid.predict_proba(X_test_hyb)[:, 1]
scores_hyb_cal = cal_hybrid.predict(scores_hyb)

# Load best threshold
with open("models/metadata.json") as f:
    metadata = json.load(f)
best_thresh = metadata.get('best_threshold', 0.30)

pred_hyb = (scores_hyb_cal >= best_thresh).astype(int)
prec_hyb = precision_score(y_test, pred_hyb, zero_division=0)
rec_hyb = recall_score(y_test, pred_hyb, zero_division=0)
f1_hyb = f1_score(y_test, pred_hyb, zero_division=0)
auc_hyb = roc_auc_score(y_test, scores_hyb_cal)
brier_hyb = brier_score_loss(y_test, scores_hyb_cal)

print(f"\n📊 Hybrid (CodeBERT + XGBoost - Threshold={best_thresh:.2f}):")
print(f"   Precision: {prec_hyb:.4f} ({prec_hyb*100:.2f}%)")
print(f"   Recall:    {rec_hyb:.4f} ({rec_hyb*100:.2f}%)")
print(f"   F1:        {f1_hyb:.4f} ({f1_hyb*100:.2f}%)")
print(f"   AUC-ROC:   {auc_hyb:.4f} ({auc_hyb*100:.2f}%)")
print(f"   Brier:     {brier_hyb:.4f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*60)
print("COMPARISON TABLE")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['XGBoost (Tabular)', 'CodeBERT (Text)', 'Hybrid (Multi-modal)'],
    'Precision': [f"{prec_xgb:.4f}", f"{prec_cb:.4f}", f"{prec_hyb:.4f}"],
    'Recall': [f"{rec_xgb:.4f}", f"{rec_cb:.4f}", f"{rec_hyb:.4f}"],
    'F1-Score': [f"{f1_xgb:.4f}", f"{f1_cb:.4f}", f"{f1_hyb:.4f}"],
    'AUC-ROC': [f"{auc_xgb:.4f}", f"{auc_cb:.4f}", f"{auc_hyb:.4f}"],
    'Brier Score': [f"{brier_xgb:.4f}", f"{brier_cb:.4f}", f"{brier_hyb:.4f}"]
})

print("\n" + comparison.to_string(index=False))

# Save to CSV
comparison.to_csv("models/comparison_table.csv", index=False)
print("\n   ✅ Saved to models/comparison_table.csv")

# ============================================================================
# VISUALIZATION 1: METRICS COMPARISON
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

Path("figures").mkdir(exist_ok=True)

# Figure 1: Metrics Comparison
print("\n📊 Plot 1: Metrics Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['F1-Score', 'AUC-ROC', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.25

values_xgb = [f1_xgb, auc_xgb, prec_xgb, rec_xgb]
values_cb = [f1_cb, auc_cb, prec_cb, rec_cb]
values_hyb = [f1_hyb, auc_hyb, prec_hyb, rec_hyb]

ax.bar(x - width, values_xgb, width, label='XGBoost (Tabular)', alpha=0.8)
ax.bar(x, values_cb, width, label='CodeBERT (Text)', alpha=0.8)
ax.bar(x + width, values_hyb, width, label='Hybrid', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved to figures/metrics_comparison.png")

# ============================================================================
# VISUALIZATION 2: ROC CURVES
# ============================================================================
print("\n📊 Plot 2: ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 8))

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, scores_xgb_tab_cal)
fpr_cb, tpr_cb, _ = roc_curve(y_test, scores_cb)
fpr_hyb, tpr_hyb, _ = roc_curve(y_test, scores_hyb_cal)

ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})', linewidth=2)
ax.plot(fpr_cb, tpr_cb, label=f'CodeBERT (AUC={auc_cb:.3f})', linewidth=2)
ax.plot(fpr_hyb, tpr_hyb, label=f'Hybrid (AUC={auc_hyb:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("figures/roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved to figures/roc_curves.png")

# ============================================================================
# VISUALIZATION 3: PRECISION-RECALL CURVES
# ============================================================================
print("\n📊 Plot 3: Precision-Recall Curves...")
fig, ax = plt.subplots(figsize=(8, 8))

prec_xgb_pr, rec_xgb_pr, _ = precision_recall_curve(y_test, scores_xgb_tab_cal)
prec_cb_pr, rec_cb_pr, _ = precision_recall_curve(y_test, scores_cb)
prec_hyb_pr, rec_hyb_pr, _ = precision_recall_curve(y_test, scores_hyb_cal)

ax.plot(rec_xgb_pr, prec_xgb_pr, label=f'XGBoost', linewidth=2)
ax.plot(rec_cb_pr, prec_cb_pr, label=f'CodeBERT', linewidth=2)
ax.plot(rec_hyb_pr, prec_hyb_pr, label=f'Hybrid', linewidth=2)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(loc='lower left')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("figures/pr_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved to figures/pr_curves.png")

# ============================================================================
# VISUALIZATION 4: FEATURE IMPORTANCE (HYBRID)
# ============================================================================
print("\n📊 Plot 4: Feature Importance...")
fig, ax = plt.subplots(figsize=(12, 10))

importance = model_hybrid.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': features_hybrid,
    'Importance': importance
}).sort_values('Importance', ascending=True)

# Top 15 features
top_15 = feat_imp.tail(15)

ax.barh(top_15['Feature'], top_15['Importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Top 15 Features - Hybrid Model')
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved to figures/feature_importance.png")

# ============================================================================
# VISUALIZATION 5: CALIBRATION CURVE
# ============================================================================
print("\n📊 Plot 5: Calibration Curve...")
fig, ax = plt.subplots(figsize=(8, 8))

prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, scores_xgb_tab_cal, n_bins=10)
prob_true_cb, prob_pred_cb = calibration_curve(y_test, scores_cb, n_bins=10)
prob_true_hyb, prob_pred_hyb = calibration_curve(y_test, scores_hyb_cal, n_bins=10)

ax.plot(prob_pred_xgb, prob_true_xgb, 'o-', label='XGBoost', linewidth=2)
ax.plot(prob_pred_cb, prob_true_cb, 's-', label='CodeBERT', linewidth=2)
ax.plot(prob_pred_hyb, prob_true_hyb, '^-', label='Hybrid', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves')
ax.legend(loc='upper left')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("figures/calibration_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved to figures/calibration_curves.png")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*60)
print("PHASE 5 COMPLETE!")
print("="*60)
print("\n📊 RESULTS:")
print(f"   XGBoost (Tabular):  F1={f1_xgb*100:.2f}%, AUC={auc_xgb*100:.2f}%")
print(f"   CodeBERT (Text):    F1={f1_cb*100:.2f}%, AUC={auc_cb*100:.2f}%")
print(f"   Hybrid:             F1={f1_hyb*100:.2f}%, AUC={auc_hyb*100:.2f}%")
print(f"\n📈 IMPROVEMENT:")
print(f"   Hybrid vs XGBoost:  {'+' if f1_hyb>=f1_xgb else ''}{(f1_hyb-f1_xgb)*100:.2f}% F1")
print(f"   Hybrid vs CodeBERT: {'+' if f1_hyb>=f1_cb else ''}{(f1_hyb-f1_cb)*100:.2f}% F1")
print("\n💾 SAVED:")
print("   - models/comparison_table.csv")
print("   - figures/metrics_comparison.png")
print("   - figures/roc_curves.png")
print("   - figures/pr_curves.png")
print("   - figures/feature_importance.png")
print("   - figures/calibration_curves.png")
print("\n🎉 PHASE 5 DONE!")
print("="*60)
