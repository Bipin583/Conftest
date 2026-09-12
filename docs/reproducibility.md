# ConfTest Reproducibility Guide

This guide maps data collection, temporal splitting, training, calibration, policy tuning, evaluation, and committed reports. It separates inspection from recomputation and protects the held-out test split from threshold selection.

## Reproducibility principles

1. Record the repository revision, Python environment, input files, random seeds, commands, and output paths.
2. Obtain labels from executed tests or an explicitly identified external ground-truth adapter; never fabricate missing outcomes.
3. Split chronologically so later commits do not inform earlier training rows.
4. Fit models on training data, select calibration/policy behavior on validation data, and report final behavior once on held-out test data.
5. Resample at the commit or other declared independent unit, not correlated commit-test rows, when estimating uncertainty for commit-level metrics.
6. Preserve artifact provenance and refuse missing metrics instead of substituting examples.

## Inspect without recomputing

```bash
python scripts/evaluate.py
python scripts/check_no_fabricated_labels.py
```

The first command prints each evaluation stage, producer, and expected artifact status. It does not run every experiment. The second checks label provenance. Start here when reviewing published claims.

## Data collection

Mine a real repository with `scripts/collect_repository_data.py`; inspect its current interface before use:

```bash
python scripts/collect_repository_data.py --help
python scripts/collect_repository_data.py \
  --repo-path /path/to/repository \
  --name owner/repository \
  --max-commits 500 \
  --output data/raw
```

The script also has an explicit `--synthetic` mode for acknowledged synthetic experiments. Synthetic rows are not evidence of real-repository effectiveness and must remain labeled as synthetic. `--persist-db` stores collected records when database persistence is desired.

Collection should retain repository identity, commit ordering, changed files, candidate tests, execution outcomes, and any grouping identifier used by later resampling. Record failed or skipped executions rather than silently converting them to passes.

## Build temporal splits

```bash
python scripts/build_splits.py --help
python scripts/build_splits.py \
  --input data/processed/features.csv \
  --output-dir data/splits \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

Verify timestamps are ordered and that a commit or mutation group does not cross split boundaries. Training data fits model parameters; validation data chooses calibration and policy; test data measures the frozen pipeline.

## Train the ensemble

```bash
python scripts/train_ensemble.py --help
python scripts/train_ensemble.py \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv \
  --output-dir models/ensembles/5_seed_lgbm
```

The canonical input order is `src/conftest/features/pipeline.py::FEATURE_NAMES`. Check for constant columns and preserve the resulting diagnostics. The committed experiment has 13 constant training features, including all 12 diff/churn columns; that limitation must not be erased during comparison.

## Fit calibration

```bash
python scripts/calibrate_model.py --help
python scripts/calibrate_model.py \
  --val data/splits/val.csv \
  --test data/splits/test.csv \
  --ensemble models/ensembles/5_seed_lgbm \
  --bootstraps 2000 \
  --output-calibrator models/calibrator.joblib \
  --output-report reports/calibration_report.json
```

The script evaluates supported methods; there is no `--method` flag. Report which split supplies selection evidence and which supplies final metrics. Keep paired intervals and resampling units with point estimates.

## Tune the policy

```bash
python scripts/tune_policy.py --help
python scripts/tune_policy.py \
  --val data/splits/val.csv \
  --test data/splits/test.csv \
  --ensemble models/ensembles/5_seed_lgbm \
  --calibrator models/calibrator.joblib \
  --output-config models/policy_config.json \
  --budget 0.25 \
  --objective zero_escape \
  --output-report reports/policy_tuning_report.json
```

Supported objectives are `zero_escape` and `recall_floor`; use `--recall-floor` only with the latter. Grid options are `--tau-abstain-grid` and `--tau-conf-grid`. There is no `--target-recall` flag.

Choose the operating point from validation results. A sweep over held-out test thresholds may diagnose generalization but must never be used to relabel the best test threshold as the selected policy.

## Run evaluation stages

List stages with `python scripts/evaluate.py`, then run one stage at a time:

```bash
python scripts/evaluate.py --run calibration
python scripts/evaluate.py --run policy
python scripts/evaluate.py --run baselines -- --bootstraps 2000
python scripts/evaluate.py --run latency
python scripts/evaluate.py --run economics
```

The dispatcher includes calibration, policy, baselines, recall-floor, uncertainty, ablation, flakiness, latency, economics, cross-repository, and continuous-learning stages. Arguments after `--` are forwarded to the producer. Use `--all` only after reviewing inputs and cost; cross-repository and continuous-learning stages retrain repeatedly.

For flakiness experiments, the supported option is `--noise-levels`, not `--noise`:

```bash
python scripts/run_flakiness_test.py --noise-levels 0.0,0.1,0.2
```

## Evidence map

| Claim | Committed artifact | Producer path |
|---|---|---|
| Baseline recall, escapes, reduction, abstention | `reports/baseline_comparison.csv` | stage `baselines` |
| Calibration metrics and intervals | `reports/calibration_report.json` | stage `calibration` |
| Recall-floor generalization | `reports/g5_recall_floor.json` | stage `recall-floor` |
| Feature-group ablation | `reports/ablation_study.json` | stage `ablation` |
| Scoring latency | `reports/latency_benchmark.json` | stage `latency` |
| Economic projection | `reports/economic_analysis.json` | stage `economics` |

Use `scripts/evaluate.py` as the live authority for exact producer commands and complete artifact names. If the map and a report disagree, investigate rather than merging values from different runs.

## Interpret the shipped result

On 183 held-out commits, the shipped policy observed 100.00% failure recall and zero escaped commits, 3.11% test-execution reduction, 97.81% abstention, and 0.0% measured wall-clock reduction. A validation-selected higher-reduction point reached 87.69% held-out recall and failed the 95% recall gate. These are empirical observations, not guarantees.

Published reports predate the row-bagging trainer fix. Retraining creates a new experimental result; it does not reproduce the old numbers exactly and requires regeneration of calibration, policy, and all dependent reports.

## Reproduction checklist

- Record input checksums and repository revision.
- Run the provenance guard.
- Confirm canonical feature order and constant-column diagnostics.
- Verify temporal and group separation.
- Freeze model before calibration, and calibration before policy tuning.
- Select thresholds only on validation.
- Keep test-set diagnostics labeled post hoc.
- Name report producer, split, sample count, unit, and interval method.
- Distinguish test-count reduction from duration reduction.
- Preserve failed commands and missing artifacts in the final account.
