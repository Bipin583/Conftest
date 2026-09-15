# ⚡ Quick Start — Hybrid Flaky Test Detection System

Get from clone to a working flakiness prediction in under 10 minutes.

## 1. Prerequisites

- Python **3.11**
- (Optional) NVIDIA GPU with CUDA 12.1 for the hybrid CodeBERT path — everything works CPU-only on the default XGBoost path

## 2. Install

```powershell
cd train
pip install -r requirements.txt

# GPU variant (CUDA 12.1 wheels):
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

## 3. Verify the installation

```powershell
python verify_project.py
```

This audits files, model weights, live predictions, figures, and metadata. If anything is missing, it names the phase script that produces it (e.g. `python phase2_xgboost.py`).

## 4. Make a prediction

```powershell
# Human-readable output
python predict.py --message "fix: resolve deadlock in socket worker pool" --failure_rate 0.35

# JSON output (for CI / scripting)
python predict.py --message "refactor: clean up comments" --failure_rate 0.05 --json
```

Example JSON result:

```json
{
  "probability": 0.4123,
  "threshold": 0.30,
  "prediction": "FLAKY",
  "risk_tier": "MEDIUM",
  "recommendation": "Quarantine test and enable auto-retry",
  "model_path": "xgboost"
}
```

`predict.py` automatically uses the hybrid CodeBERT+XGBoost path if the fine-tuned checkpoint and GPU libraries are available, and the XGBoost tabular path otherwise (equal measured F1).

## 5. Launch the dashboard

```powershell
streamlit run app.py
```

Parameter sliders, CodeBERT semantic score meter, risk classification, and embedded evaluation figures.

## 6. Re-run evaluation & regenerate figures

```powershell
python phase5_evaluation.py
```

Produces `models/comparison_table.csv` and 5 publication-quality 300-DPI figures in `figures/`.

## 7. Use in CI (test selection)

From the repo root:

```powershell
python -m conftest_cli.select --base-sha HEAD~1 --head-sha HEAD --test-dir tests --output selected_tests.txt
python -m conftest_cli.report --junit junit_results.xml
```

`select.py` picks the tests worth running for a commit range; `report.py` turns the JUnit XML into a PR comment summary.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `UnicodeEncodeError: charmap` on Windows | Tools already reconfigure stdout to UTF-8; if it persists, run `chcp 65001` first |
| `ERROR: missing model artifacts` | Run the producer script named in the error (e.g. `python phase2_xgboost.py`) |
| Slow / no GPU | Expected on the XGBoost path — it's CPU-first by design; hybrid only engages when torch + checkpoint are present |

---

*Full details: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) · [API_REFERENCE.md](API_REFERENCE.md)*
