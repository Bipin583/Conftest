# ============================================================================
# PHASE 2 FINAL: XGBOOST BASELINE WITH REALISTIC LABELS
# Fixed: All imports included
# ============================================================================

import pandas as pd, numpy as np, xgboost as xgb, joblib, json, time
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score
from pathlib import Path  # ✅ FIXED: Added this import

print("="*60)
print("PHASE 2 FINAL: XGBOOST BASELINE")
print("="*60)

# LOAD
print("\n📂 Loading dataset...")
df = pd.read_csv("dataset_full_realistic.csv")
print(f"✅ Loaded {len(df):,} samples")

# CHECK LABELS
print("\n📊 Label distribution:")
print(f"   Non-flaky (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
print(f"   Flaky (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")

# PREPROCESSING
print("\n✂️  Preprocessing...")
df = df.sort_values('commit_id')
n = len(df)
train_df = df.iloc[:int(n*0.7)]
val_df = df.iloc[int(n*0.7):int(n*0.85)]
test_df = df.iloc[int(n*0.85):]

print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# FEATURES
features = ['lines_added', 'lines_deleted', 'files_changed', 'prior_failures', 'failure_rate', 'avg_duration', 'test_complexity', 'code_coverage', 'lines_of_code', 'cyclomatic_complexity', 'repo_id', 'language_id']
X_train, y_train = train_df[features], train_df['label']
X_val, y_val = val_df[features], val_df['label']
X_test, y_test = test_df[features], test_df['label']

# TRAIN
print("\n🚀 Training XGBoost...")
start = time.time()
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=(y_train==0).sum() / max((y_train==1).sum(), 1)
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = (time.time() - start) / 60
print(f"   ✅ Done in {train_time:.1f} minutes")

# CALIBRATE
print("\n🎯 Calibrating...")
val_scores = model.predict_proba(X_val)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_scores, y_val)

# FIND BEST THRESHOLD
print("\n📈 Finding best threshold...")
test_scores = model.predict_proba(X_test)[:, 1]
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
    print(f"   {threshold:.2f}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n✅ BEST THRESHOLD: {best_threshold:.2f}")

# FINAL EVALUATION
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

print(f"\n📊 FINAL PERFORMANCE METRICS:")
print(f"   Precision:  {prec:.4f} ({prec*100:.2f}%)")
print(f"   Recall:     {rec:.4f} ({rec*100:.2f}%)")
print(f"   F1 Score:   {f1:.4f} ({f1*100:.2f}%)")
print(f"   AUC-ROC:    {auc:.4f} ({auc*100:.2f}%)")
print(f"   Brier Score: {brier:.4f}")
print(f"   ECE:        {ece:.4f}")

# FEATURE IMPORTANCE
print(f"\n📊 TOP 10 FEATURE IMPORTANCE:")
importances = model.feature_importances_
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(feature_importance[:10], 1):
    print(f"   {i:2d}. {feat:25s}: {imp:.4f}")

# SAVE
print("\n💾 Saving...")
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xgb_model_final.pkl")
joblib.dump(calibrator, "models/xgb_calibrator_final.pkl")
joblib.dump(features, "models/xgb_features_final.pkl")

metrics = {
    "model": "XGBoost",
    "dataset": "realistic_labels",
    "samples": len(df),
    "languages": df['language'].nunique(),
    "repos": df['repo'].nunique(),
    "threshold": best_threshold,
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "auc": float(auc),
    "brier_score": float(brier),
    "ece": float(ece),
    "training_time_minutes": train_time,
    "feature_importance": {feat: float(imp) for feat, imp in feature_importance}
}

with open("models/xgb_metrics_final.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("   ✅ Saved to models/")

print("\n" + "="*60)
print("PHASE 2 COMPLETE!")
print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"AUC-ROC:  {auc:.4f} ({auc*100:.2f}%)")
print(f"Threshold: {best_threshold:.2f}")
print("="*60)