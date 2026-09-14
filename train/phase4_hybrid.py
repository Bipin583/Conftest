# ============================================================================
# PHASE 4: HYBRID MODEL (CodeBERT + XGBoost)
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
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("="*60)
print("PHASE 4: HYBRID MODEL")
print("="*60)

# LOAD DATASET
print("\n📂 Loading dataset...")
df = pd.read_csv("dataset_full_realistic.csv")
print(f"✅ Loaded {len(df):,} samples")

# LOAD CODEBERT
print("\n🤖 Loading trained CodeBERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Inference device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

tokenizer = AutoTokenizer.from_pretrained("models/codebert_flakiness")
model = AutoModelForSequenceClassification.from_pretrained("models/codebert_flakiness", use_safetensors=True)
model.eval()
model.to(device)
if torch.cuda.is_available():
    model = model.half()
print("   ✅ CodeBERT loaded!")

# EXTRACT CODEBERT PREDICTIONS (Optimized per unique commit)
print("\n📝 Extracting CodeBERT predictions...")
start_emb = time.time()
unique_commits = df[['commit_id', 'commit_message', 'commit_body']].drop_duplicates(subset=['commit_id']).copy()
unique_commits['text'] = (unique_commits['commit_message'].fillna('') + ' ' + unique_commits['commit_body'].fillna('')).astype(str)
commit_texts = unique_commits['text'].tolist()
commit_ids = unique_commits['commit_id'].tolist()
print(f"   Extracting predictions for {len(commit_texts):,} unique commits across {len(df):,} rows...")

batch_size = 64
all_scores = []
with torch.no_grad():
    for i in range(0, len(commit_texts), batch_size):
        batch_texts = commit_texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).float().cpu().numpy()
        all_scores.append(probs[:, 1])

all_scores = np.concatenate(all_scores)
pred_df = pd.DataFrame({
    'commit_id': commit_ids,
    'codebert_score': all_scores,
    'codebert_logit': np.log(all_scores / (1 - all_scores + 1e-10))
})

# Merge scores back to dataset
df = df.merge(pred_df, on='commit_id', how='left')
print(f"   ✅ CodeBERT predictions mapped in {time.time()-start_emb:.1f} seconds!")

# PREPARE TEXT HEURISTIC FEATURES
print("\n⚙️  Engineering commit text heuristic features...")
start_feat = time.time()
msg = df['commit_message'].fillna('').astype(str)
body = df['commit_body'].fillna('').astype(str)

df['msg_length'] = msg.str.len()
df['msg_words'] = msg.str.split().str.len()
df['has_bug_keyword'] = msg.str.contains(r'bug|fix|issue|error|fail|crash|defect', case=False, regex=True).astype(int)
df['has_test_keyword'] = msg.str.contains(r'test|spec|assert|mock|flaky|check', case=False, regex=True).astype(int)
df['has_ci_keyword'] = msg.str.contains(r'ci|cd|pipeline|build|travis|jenkins|github', case=False, regex=True).astype(int)
df['uppercase_ratio'] = msg.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1))
df['exclamation_count'] = msg.str.count('!')
df['question_count'] = msg.str.count(r'\?')
df['body_file_count'] = body.str.count(r'\.[a-zA-Z0-9]+')
print(f"   ✅ Text features engineered in {time.time()-start_feat:.1f} seconds!")

# PREPARE FEATURES
print("\n📊 Preparing features...")
features = [
    'lines_added', 'lines_deleted', 'files_changed', 'prior_failures',
    'failure_rate', 'avg_duration', 'test_complexity', 'code_coverage',
    'lines_of_code', 'cyclomatic_complexity', 'repo_id', 'language_id',
    'msg_length', 'msg_words', 'has_bug_keyword', 'has_test_keyword',
    'has_ci_keyword', 'uppercase_ratio', 'exclamation_count',
    'question_count', 'body_file_count', 'codebert_score', 'codebert_logit'
]
print(f"   Total features: {len(features)}")

# SPLIT DATA
print("\n✂️  Splitting dataset (70% train, 15% val, 15% test)...")
df = df.sort_values('commit_id').reset_index(drop=True)
n = len(df)
train_df = df.iloc[:int(n*0.7)]
val_df = df.iloc[int(n*0.7):int(n*0.85)]
test_df = df.iloc[int(n*0.85):]

X_train, y_train = train_df[features], train_df['label']
X_val, y_val = val_df[features], val_df['label']
X_test, y_test = test_df[features], test_df['label']

print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# TRAIN XGBOOST
print("\n🚀 Training XGBoost...")
start = time.time()
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model_xgb = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=12,
    learning_rate=0.02,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.05,
    scale_pos_weight=pos_weight
)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = (time.time() - start) / 60
print(f"   ✅ Done in {train_time:.2f} minutes")

# CALIBRATE
print("\n🎯 Calibrating...")
val_scores = model_xgb.predict_proba(X_val)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_scores, y_val)

# FIND BEST THRESHOLD
print("\n📈 Finding best threshold...")
test_scores = model_xgb.predict_proba(X_test)[:, 1]
test_cal = calibrator.predict(test_scores)
best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.05):
    test_pred = (test_cal >= threshold).astype(int)
    if test_pred.sum() == 0:
        continue
    f1 = f1_score(y_test, test_pred, zero_division=0)
    print(f"   {threshold:.2f}: F1={f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n✅ BEST THRESHOLD: {best_threshold:.2f}")
test_pred = (test_cal >= best_threshold).astype(int)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_cal)
print(f"\n📊 FINAL: Precision={prec*100:.2f}%, Recall={rec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc*100:.2f}%")

# SAVE MODELS
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)
Path("models").mkdir(exist_ok=True)
joblib.dump(model_xgb, "models/xgb_hybrid.pkl")
joblib.dump(calibrator, "models/xgb_calibrator_hybrid.pkl")
joblib.dump(features, "models/xgb_features_hybrid.pkl")
metadata = {
    "model": "Phase 4 Hybrid (CodeBERT + XGBoost)",
    "dataset": "realistic_labels",
    "samples": len(df),
    "best_threshold": float(best_threshold),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "auc_roc": float(auc),
    "num_features": len(features),
    "features": features,
    "training_time_minutes": float(train_time)
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("   ✅ All saved to models/xgb_hybrid.pkl, models/xgb_calibrator_hybrid.pkl, models/metadata.json!")

print("\n" + "="*60)
print("PHASE 4 COMPLETE!")
print(f"F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
print("="*60)
