# Experiment Protocols and Artifact Map

ConfTest has multiple evaluation stages. `scripts/evaluate.py` is the authoritative dispatcher: without arguments it lists each report and its producer; `--run` invokes one producer; `--all` runs every stage in dependency order and is intentionally expensive.

## Split discipline

The committed mutation corpus is split by whole mutant observations using its harvest-order surrogate: 849 train, 181 validation, and 183 held-out test observations. Test rows from one mutant stay together. Model fitting uses training data; calibration and policy choices use validation data; frozen artifacts are measured on the test split. See [Dataset](dataset.md) for why the timestamps are not natural commit dates.

## Baseline comparison

`scripts/train_baseline.py` evaluates eight strategies on the held-out split:

1. full test suite;
2. random selection at the common budget;
3. changed-file selection;
4. static AST call-graph selection;
5. historical failure-frequency ranking;
6. uncalibrated LightGBM;
7. calibrated LightGBM without abstention; and
8. calibrated selective ConfTest.

The full suite is a complete-execution reference, not an “oracle safety” guarantee. Budgeted strategies use a 25% test budget where applicable. `reports/baseline_comparison.csv` carries headline point estimates; `baseline_comparison_intervals.json`, `baseline_per_commit.csv`, and `statistical_significance.json` carry resampling and paired-analysis detail. The shipped operating point is summarized in [Selective prediction](selective_prediction.md) rather than duplicated here.

## Metric definitions

For candidate set `T`, selected set `S`, failing set `F`, and measured test durations `d(t)`:

```text
TRR = 1 - |S| / |T|
ETR = 1 - sum(d(t), t in S) / sum(d(t), t in T)
FR  = |S intersect F| / |F|
MFR = 1 - FR
AR  = abstained observations / evaluated observations
```

TRR is count reduction; ETR requires measured durations. Failure recall is undefined for an individual observation with no failures, although aggregate producers may define a pooled rate over all failing rows. Report consumers must read each artifact's denominator metadata.

Calibration stages additionally report Brier score, ECE, MCE, and reliability bins. Ranking studies report PR-AUC, ROC-AUC, and recall at a per-observation budget. See [Statistical methodology](statistical_methodology.md).

## Evaluation stages

| Stage | Producer | Primary report | Scope |
|---|---|---|---|
| Calibration | `scripts/calibrate_model.py` | `calibration_report.json` | validation selection, held-out measurement |
| Policy | `scripts/tune_policy.py` | `policy_tuning_report.json` | validation threshold sweep |
| Baselines | `scripts/train_baseline.py` | `baseline_comparison.csv` | held-out strategy comparison |
| Recall floor | `scripts/check_g5_recall_floor.py` | `g5_recall_floor.json` | frozen validation-picked point on test |
| Uncertainty | `scripts/uncertainty_eval.py` | `uncertainty_analysis.json` | disagreement and risk-coverage diagnostics |
| Ablation | `scripts/run_ablation_study.py` | `ablation_study.json` | retrained feature-group variants |
| Flakiness | `scripts/run_flakiness_test.py` | `flakiness_robustness.json` | injected training-label noise |
| Latency | `scripts/run_latency_benchmark.py` | `latency_benchmark.json` | scoring path only |
| Economics | `scripts/run_economic_analysis.py` | `economic_analysis.json` | scenario projections |
| Cross-repo | `scripts/run_cross_repo_eval.py` | `cross_repo_generalization.json` | leave-one-project-out transfer |
| Continuous learning | `scripts/run_continuous_learning.py` | `continuous_learning.json` | detector/adaptation trade-offs |

## Reproduction

```bash
python scripts/evaluate.py
python scripts/evaluate.py --run baselines -- --bootstraps 2000
```

Regenerating the ensemble invalidates its calibrator and tuned policy. The committed reports also predate effective row bagging; they must not be attributed to the corrected trainer until every dependent artifact is rebuilt. See [Uncertainty estimation](uncertainty_estimation.md) and [Limitations](limitations.md).
