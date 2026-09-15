# ============================================================================
# PREDICT_LITE.PY: CPU-only launcher for the XGBoost tabular path.
#
# Forces predict.py's --model xgboost path even when the CodeBERT checkpoint
# and GPU stack are installed, giving the <1s "XGBoost Lite" deployment mode
# (F1 0.556 vs 0.555 hybrid — no measurable accuracy cost, per
# models/comparison_table.csv). All arguments are forwarded to predict.py.
#
# Usage: python predict_lite.py --message "fix: resolve deadlock" --failure_rate 0.35 --json
# ============================================================================

import runpy
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.argv = [sys.argv[0], "--model", "xgboost"] + sys.argv[1:]
runpy.run_path(str(Path(__file__).resolve().parent / "predict.py"), run_name="__main__")
