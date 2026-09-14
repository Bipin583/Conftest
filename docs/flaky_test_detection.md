# Flaky test detection (hybrid subproject)

ConfTest ships a second, standalone ML subsystem in [`train/`](../train/README.md): a **hybrid flaky test detection** system that predicts whether the test touched by a commit is flaky, using a fine-tuned CodeBERT for the commit text and XGBoost for tabular commit/test features. It is kept as a self-contained subproject — its own README, requirements and phase scripts — and is exposed to ConfTest consumers through one API endpoint, `POST /api/v1/flakiness/predict` (see [api.md](api.md)).

## Why standalone

The main ConfTest engine predicts *which regression tests to run for a commit*; this subsystem predicts *whether a given test is flaky*. Different task, different feature space, different model stack (PyTorch + transformers, which the main API deliberately does not depend on). The process boundary is deliberate: `train/predict.py` is invoked as a subprocess, so the main API needs only the light tabular path's dependencies.

## Inference paths

`train/predict.py` picks its path from what is actually on disk:

| Path | Model | Requires | When |
|---|---|---|---|
| `hybrid` | CodeBERT + XGBoost fusion | `train/models/codebert_flakiness/` (499 MB, not tracked) and torch/transformers | Full-accuracy local/GPU use |
| `xgboost` | Tabular XGBoost (12 features) + isotonic calibrator | `train/models/xgb_*_final.pkl` (tracked, ~16 MB) | Fresh clones, CI, CPU deployment — the default |

The fallback costs nothing measurable: on the held-out test split the tabular model **matches or slightly beats** the hybrid (see the table below). A fresh clone of this repository can run the `xgboost` path immediately.

```bash
python train/predict.py --message "fix: resolve deadlock" --failure_rate 0.35 --test_complexity 22 --json
```

## Measured performance

Every number below is read from `train/models/comparison_table.csv`, produced by `train/phase5_evaluation.py` on the held-out test split of the `realistic_labels` dataset (1,228,500 samples; per-model details in `train/models/metadata.json` and `train/models/xgb_metrics_final.json`):

| Model | Precision | Recall | F1 | AUC-ROC | Brier |
|---|---:|---:|---:|---:|---:|
| XGBoost (tabular) | 0.4073 | 0.8753 | 0.5559 | 0.6066 | 0.2272 |
| CodeBERT (text only) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2358 |
| Hybrid (CodeBERT + XGBoost) | 0.4058 | 0.8774 | 0.5549 | 0.6049 | 0.2275 |

Two honest caveats, visible in the table itself:

- **CodeBERT alone scored AUC 0.5000 — random guessing.** Commit text carries intent, but cannot identify *which* test in a suite flakes; flakiness lives in execution history and churn, which is what the tabular features encode.
- **The hybrid did not beat the tabular model.** F1 0.5549 vs 0.5559. The text model adds no measurable signal on this dataset; the fusion is retained because the comparison is the finding, not an inconvenience.

## Training pipeline

The phase scripts in `train/` regenerate everything (run inside `train/`):

1. `phase2_xgboost.py` — tabular baseline; produces the tracked `models/xgb_*_final.pkl` artifacts.
2. `phase3_codebert.py` — fine-tunes CodeBERT on commit text (GPU; produces the untracked checkpoint).
3. `phase4_hybrid.py` — trains the fusion model (`models/xgb_hybrid.pkl` and friends).
4. `phase5_evaluation.py` — comparative evaluation; writes `models/comparison_table.csv` and the figures in `train/figures/`.

The ~300 MB training datasets (`train/dataset_full.csv`, `train/dataset_full_realistic.csv`) and the CodeBERT checkpoints are deliberately **not tracked** — see `.gitignore`. They stay on the machine that trained them; the tracked tabular artifacts keep inference reproducible from a fresh clone.

## Relationship to ConfTest's own flakiness work

This subproject is complementary to the measurement-based flakiness analysis in [flakiness_robustness.md](flakiness_robustness.md): that page measures how flakiness *degrades ConfTest's selection* (noise injection, stress reruns); this subproject *predicts* per-test flakiness from commit and test features. They share no code.
