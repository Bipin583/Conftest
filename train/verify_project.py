import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import json
import subprocess

print("="*60)
print("FINAL PROJECT VERIFICATION")
print("="*60)
print("")

# CHECK FILES EXIST
print("📁 Checking files...")
files = ["predict.py", "app.py", "dataset_full_realistic.csv", "models", "figures"]
for f in files:
    p = Path(f)
    if p.exists():
        print(f"   ✅ {f} exists")
    else:
        print(f"   ❌ {f} MISSING")
print("")

# TEST PREDICT.PY
print("🔮 Testing predict.py...")
cmd = [
    sys.executable, "predict.py",
    "--message", "fix: resolve deadlock in socket worker pool",
    "--failure_rate", "0.35",
    "--test_complexity", "22",
    "--lines_added", "50",
    "--files_changed", "5"
]
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
if res.stdout:
    print(res.stdout)
if res.stderr:
    # Print only non-progress stderr if any
    filtered = "\n".join([line for line in res.stderr.splitlines() if "Loading weights" not in line])
    if filtered.strip():
        print(filtered)
print("")

# CHECK FIGURES
print("📊 Checking figures...")
figs = [
    "metrics_comparison.png",
    "roc_curves.png",
    "pr_curves.png",
    "feature_importance.png",
    "calibration_curves.png"
]
for fig in figs:
    p = Path("figures") / fig
    if p.exists():
        print(f"   ✅ {fig}")
    else:
        print(f"   ❌ {fig} MISSING")
print("")

# SHOW MODEL INFO
print("📊 Model Performance:")
meta_p = Path("models/metadata.json")
if meta_p.exists():
    with open(meta_p) as f:
        m = json.load(f)
    print(f"   F1 Score: {m.get('f1_score', 0)*100:.2f}%")
    print(f"   AUC-ROC: {m.get('auc_roc', 0)*100:.2f}%")
    print(f"   Threshold: {m.get('best_threshold', 0):.2f}")
else:
    print("   ❌ metadata.json not found")
print("")

print("="*60)
print("VERIFICATION COMPLETE!")
print("="*60)
print("")
print("🚀 NEXT STEPS:")
print("   1. If all ✅: Project is ready!")
print("   2. Launch dashboard: streamlit run app.py")
print("   3. Prepare your demo presentation")
print("")
