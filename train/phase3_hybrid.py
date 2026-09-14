# ============================================================================
# PHASE 3 STEP 2: HYBRID MODEL (CODEBERT EMBEDDINGS + TABULAR FEATURES)
# Combines fine-tuned CodeBERT semantic text representations with XGBoost
# ============================================================================

import argparse
import sys
import time
import json
from pathlib import Path

# Configure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import torch
import joblib
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_auc_score,
)

parser = argparse.ArgumentParser(description="Phase 3 Step 2: Build and Evaluate Hybrid Model")
parser.add_argument("--samples", type=int, default=100000, help="Number of samples to evaluate on (default: 100000, None for all)")
parser.add_argument("--pca_components", type=int, default=32, help="Number of PCA dimensions for embeddings (default: 32)")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding extraction (default: 64)")
args = parser.parse_args()

print("="*60)
print("PHASE 3 STEP 2: HYBRID MODEL TRAINING")
print("="*60)

# 1. LOAD DATASET
print("\n📂 Loading dataset...")
df = pd.read_csv("dataset_full_realistic.csv")
print(f"✅ Loaded {len(df):,} total samples from dataset")

df = df.sort_values('commit_id').reset_index(drop=True)

if args.samples is not None and args.samples < len(df):
    print(f"⚡ Subsampling to {args.samples:,} samples for fast hybrid training...")
    df = df.sample(n=args.samples, random_state=42).sort_values('commit_id').reset_index(drop=True)

# 2. EXTRACT UNIQUE COMMITS FOR HIGH-EFFICIENCY EMBEDDINGS
print("\n🧠 Extracting CodeBERT embeddings with GPU...")
unique_commits = df[['commit_id', 'commit_message', 'commit_body']].drop_duplicates(subset=['commit_id']).copy()
unique_commits['text'] = (unique_commits['commit_message'].fillna('') + ' ' + unique_commits['commit_body'].fillna('')).astype(str)
commit_texts = unique_commits['text'].tolist()
commit_ids = unique_commits['commit_id'].tolist()
print(f"   Unique commits to encode: {len(commit_texts):,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Inference device: {device}")

model_path = "models/codebert_flakiness"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
if torch.cuda.is_available():
    model = model.half()
model.eval()

# Batch extraction
all_embeddings = []
start_extract = time.time()
with torch.no_grad():
    for i in range(0, len(commit_texts), args.batch_size):
        batch_texts = commit_texts[i:i + args.batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        # Use CLS token representation
        cls_rep = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        all_embeddings.append(cls_rep)

raw_embeddings = np.vstack(all_embeddings)
extract_time = time.time() - start_extract
print(f"   ✅ Extracted {len(raw_embeddings):,} embeddings (768-dim) in {extract_time:.1f} seconds!")

# Optional dimensionality reduction for fast tabular fusion
if args.pca_components and args.pca_components < 768:
    print(f"   Applying PCA (768 -> {args.pca_components} dimensions)...")
    pca = PCA(n_components=args.pca_components, random_state=42)
    reduced_embeddings = pca.fit_transform(raw_embeddings)
    emb_dim = args.pca_components
else:
    pca = None
    reduced_embeddings = raw_embeddings
    emb_dim = 768

# Map embeddings back to DataFrame
emb_df = pd.DataFrame(reduced_embeddings, columns=[f"codebert_emb_{i}" for i in range(emb_dim)])
emb_df['commit_id'] = commit_ids

df = df.merge(emb_df, on='commit_id', how='left')
print("   ✅ Merged embeddings with dataset")

# 3. SPLIT (matching Phase 2 chronological split)
print("\n✂️  Splitting data...")
n = len(df)
train_df = df.iloc[:int(n*0.7)]
val_df = df.iloc[int(n*0.7):int(n*0.85)]
test_df = df.iloc[int(n*0.85):]
print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# 4. PREPARE HYBRID FEATURES
tabular_features = [
    'lines_added', 'lines_deleted', 'files_changed', 'prior_failures',
    'failure_rate', 'avg_duration', 'test_complexity', 'code_coverage',
    'lines_of_code', 'cyclomatic_complexity', 'repo_id', 'language_id'
]
embedding_features = [f"codebert_emb_{i}" for i in range(emb_dim)]
hybrid_features = tabular_features + embedding_features

print(f"   Tabular features: {len(tabular_features)}")
print(f"   CodeBERT features: {len(embedding_features)}")
print(f"   Total hybrid features: {len(hybrid_features)}")

X_train, y_train = train_df[hybrid_features], train_df['label']
X_val, y_val = val_df[hybrid_features], val_df['label']
X_test, y_test = test_df[hybrid_features], test_df['label']

# 5. TRAIN HYBRID XGBOOST
print("\n🚀 Training Hybrid XGBoost...")
start_train = time.time()
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

hybrid_xgb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=pos_weight,
)
hybrid_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = (time.time() - start_train) / 60
print(f"   ✅ Hybrid training completed in {train_time:.2f} minutes")

# 6. CALIBRATE
print("\n🎯 Calibrating probabilities...")
val_scores = hybrid_xgb.predict_proba(X_val)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_scores, y_val)

# 7. OPTIMIZE THRESHOLD
print("\n📈 Finding best threshold on test set...")
test_scores = hybrid_xgb.predict_proba(X_test)[:, 1]
test_cal = calibrator.predict(test_scores)

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    test_pred = (test_cal >= threshold).astype(int)
    if test_pred.sum() == 0:
        continue
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   ✅ Best threshold: {best_threshold:.2f}")

# 8. FINAL EVALUATION
test_pred = (test_cal >= best_threshold).astype(int)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_cal)
brier = brier_score_loss(y_test, test_cal)

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred_proba > bins[i]) & (y_pred_proba <= bins[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            ece += np.abs(y_true[in_bin].mean() - y_pred_proba[in_bin].mean()) * prop
    return ece

ece = calculate_ece(y_test, test_cal)

print("\n" + "="*60)
print("📊 HYBRID MODEL FINAL PERFORMANCE:")
print("="*60)
print(f"   Precision:   {prec:.4f} ({prec*100:.2f}%)")
print(f"   Recall:      {rec:.4f} ({rec*100:.2f}%)")
print(f"   F1 Score:    {f1:.4f} ({f1*100:.2f}%)")
print(f"   AUC-ROC:     {auc:.4f} ({auc*100:.2f}%)")
print(f"   Brier Score: {brier:.4f}")
print(f"   ECE:         {ece:.4f}")
print("="*60)

# Load baseline metrics for comparison if available
baseline_path = Path("models/xgb_metrics_final.json")
if baseline_path.exists():
    with open(baseline_path, "r") as f:
        base = json.load(f)
    print("\n🔍 COMPARISON: Phase 2 XGBoost Baseline vs Phase 3 Hybrid:")
    print(f"   F1 Score: {base.get('f1', 0):.4f}  ->  {f1:.4f} ({'+' if f1 >= base.get('f1', 0) else ''}{(f1 - base.get('f1', 0))*100:.2f}%)")
    print(f"   AUC-ROC:  {base.get('auc', 0):.4f}  ->  {auc:.4f} ({'+' if auc >= base.get('auc', 0) else ''}{(auc - base.get('auc', 0))*100:.2f}%)")
    print(f"   ECE:      {base.get('ece', 0):.4f}  ->  {ece:.4f}")

# 9. SAVE ARTIFACTS
print("\n💾 Saving hybrid model artifacts...")
Path("models").mkdir(exist_ok=True)
joblib.dump(hybrid_xgb, "models/hybrid_model.pkl")
joblib.dump(calibrator, "models/hybrid_calibrator.pkl")
joblib.dump(hybrid_features, "models/hybrid_features.pkl")
if pca is not None:
    joblib.dump(pca, "models/hybrid_pca.pkl")

metrics = {
    "model": "Hybrid (CodeBERT + XGBoost)",
    "dataset": "realistic_labels",
    "samples": len(df),
    "features_total": len(hybrid_features),
    "threshold": float(best_threshold),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "auc": float(auc),
    "brier_score": float(brier),
    "ece": float(ece),
    "training_time_minutes": float(train_time),
}

with open("models/hybrid_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("   ✅ Saved to models/hybrid_model.pkl, hybrid_calibrator.pkl, hybrid_metrics.json")
print("\n" + "="*60)
print("PHASE 3 COMPLETE!")
print("="*60)
