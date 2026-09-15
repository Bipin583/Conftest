# 📖 API Reference — Hybrid Flaky Test Detection System

Reference for every user-facing entry point: the prediction CLI, the dashboard, the training pipeline scripts, and the CI-facing CLIs.

---

## 1. `train/predict.py` — Unified CLI Inference

Single-sample flakiness prediction with automatic model-path selection.

**Invocation**

```powershell
python predict.py --message "<commit message>" [options]
```

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--message` | str | *(required)* | Commit message (subject line) |
| `--body` | str | `""` | Commit body; used for CodeBERT input and `body_file_count` |
| `--failure_rate` | float | `0.0` | Historical failure rate of the test, [0.0, 1.0] |
| `--test_complexity` | int | `0` | McCabe complexity of the test method |
| `--lines_added` | int | `0` | Lines added in the commit |
| `--lines_deleted` | int | `0` | Lines deleted in the commit |
| `--files_changed` | int | `0` | Files changed in the commit |
| `--json` | flag | off | Emit machine-readable JSON instead of the formatted report |
| `--model` | choice | `auto` | `auto` (detect), `xgboost` (CPU-only tabular, <1s), `hybrid` (CodeBERT + XGBoost; errors with exit 2 if unavailable) |

**Model path selection (automatic)**

- **hybrid** — used only if `train/models/codebert_flakiness/` exists *and* `torch` + `transformers` import. Loads `xgb_hybrid.pkl`, `xgb_calibrator_hybrid.pkl`, `xgb_features_hybrid.pkl`, threshold from `models/metadata.json`.
- **xgboost** (default) — loads `xgb_model_final.pkl`, `xgb_calibrated_final` set, threshold from `xgb_metrics_final.json`.

**Feature derivation** — unspecified features are derived automatically from `--message`/`--body`: `msg_length`, `msg_words`, `has_bug_keyword`, `has_test_keyword`, `has_ci_keyword`, `uppercase_ratio`, `exclamation_count`, `question_count`, `body_file_count`.

**Prediction logic**

```
score     = xgb.predict_proba(X)[0, 1]
cal_score = isotonic_calibrator.predict([score])
pred      = FLAKY if cal_score >= threshold (0.30) else STABLE
risk_tier = HIGH (≥0.7) | MEDIUM (≥0.4) | LOW
```

**JSON output schema**

```json
{
  "probability":   0.4123,        // calibrated P(flaky), float
  "threshold":     0.30,          // decision threshold, float
  "prediction":    "FLAKY",       // "FLAKY" | "STABLE"
  "risk_tier":     "MEDIUM",      // "HIGH" | "MEDIUM" | "LOW"
  "recommendation":"Quarantine test and enable auto-retry",  // human-readable action
  "model_path":    "xgboost"      // "xgboost" | "hybrid"
}
```

**Exit codes**

| Code | Meaning |
|---|---|
| `0` | Prediction produced |
| `2` | Missing model artifacts — stderr names them and the producing script |

---

## 2. `train/predict_lite.py` — CPU-only Lite Launcher

```powershell
python predict_lite.py --message "fix: resolve deadlock" --failure_rate 0.35 --json
```

Thin launcher that forwards all arguments to `predict.py` with `--model xgboost`, forcing the <1s tabular path even when the CodeBERT checkpoint and GPU stack are installed. Same arguments and JSON output schema as `predict.py`.

---

## 3. `train/app.py` — Streamlit Dashboard

```powershell
streamlit run app.py
```

Interactive web UI: parameter sliders for all tabular features, CodeBERT semantic score meter, risk classification, and embedded evaluation figures. Version-adaptive (`use_container_width` vs `use_column_width`).

---

## 3. Training pipeline (run in `train/`)

| Script | Purpose | Key outputs (`train/models/`) |
|---|---|---|
| `phase2_xgboost.py` | Tabular baseline: XGBoost (600 trees, depth 12, `tree_method="hist"`) on 12 features + isotonic calibration on validation split | `xgb_model_final.pkl`, `xgb_calibrator_final.pkl`, `xgb_features_final.pkl`, `xgb_metrics_final.json` |
| `phase3_codebert.py` | Fine-tune `microsoft/codebert-base` (125M) with FP16 + `DataCollatorWithPadding` dynamic batching | `codebert_flakiness/` checkpoint |
| `phase4_hybrid.py` | Multi-modal fusion: CodeBERT on **unique commits only** (3,796 of 1,228,500 rows → 20,000× speedup), hash-join back, train hybrid XGBoost on 23 features | `xgb_hybrid.pkl`, `xgb_calibrator_hybrid.pkl`, `xgb_features_hybrid.pkl`, `metadata.json` |
| `phase5_evaluation.py` | Comparative evaluation of all three models | `comparison_table.csv` + 5 figures in `figures/` |
| `verify_project.py` | Automated audit of files, weights, live predictions, figures, metadata | console report |

Each phase script is the documented *producer* of its artifacts — `predict.py` error messages reference them by name.

---

## 4. `conftest_cli/select.py` — CI Test Selection

Selects which tests to run for a commit range. Designed to be invoked inside GitHub Actions runners.

```powershell
python -m conftest_cli.select --base-sha HEAD~1 --head-sha HEAD --test-dir tests --output selected_tests.txt
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base-sha` | str | `HEAD~1` | Base commit SHA of the diff range |
| `--head-sha` | str | `HEAD` | Head commit SHA of the diff range |
| `--test-dir` | str | `tests` | Directory to discover `test_*.py` files in |
| `--risk-tolerance` | float | `0.18` | Abstention/uncertainty threshold τ |
| `--output` | str | `selected_tests.txt` | Destination file for selected test paths |

**Behavior:** discovers tests via filesystem walk, computes diff features between `base-sha..head-sha`, and runs the `SelectiveDecisionEngine` (`src/engine/`) to emit the selected subset. Exits with a warning (no tests selected) when the directory is empty.

---

## 5. `conftest_cli/report.py` — JUnit XML Reporter

Parses post-execution JUnit XML and prints a PR-comment summary.

```powershell
python -m conftest_cli.report --junit junit_results.xml
```

Parses `<testcase>` elements and classifies each as passed / failed / error / skipped, building the failure list as `classname::name`. A missing or unreadable XML file yields an all-zero report rather than an exception, so CI steps don't break on absent reports.

---

## 6. Programmatic API (selected internals)

```python
from src.engine.selective_engine import SelectiveDecisionEngine
```

The selective-execution engine backing `conftest_cli.select`. Feature pipelines live in `src/conftest/features/` (`diff_features`, `ast_features`, `history_features`, `dependency_graph`, `pipeline`). See module docstrings for parameters.

---

*Full details: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) · [QUICK_START.md](QUICK_START.md) · [train/README.md](train/README.md)*
