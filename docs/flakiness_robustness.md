# Flakiness Noise Stress Study

This experiment tests sensitivity to synthetic training-label corruption. It does not estimate production flaky-test prevalence or prove that ensemble disagreement detects real flakiness.

## Protocol

`scripts/run_flakiness_test.py` flips 0%, 5%, 10%, 20%, or 30% of training labels selected by a generated beta-distributed score. Validation and held-out labels remain clean. At each level it compares:

- a 30-tree LightGBM model trained on corrupted labels without per-row weights; and
- a 30-tree model using `clip(1 - 0.7 * score, 0.1, 1.0)`, followed by temperature calibration on clean validation data.

Recall uses a 25% budget within each held-out observation. This is a focused stress-study configuration, not the five-member selective serving pipeline.

## Committed outcomes

| Flips | Training positive rate | Standard PR-AUC | Weighted/calibrated PR-AUC | Standard recall | Weighted/calibrated recall |
|---:|---:|---:|---:|---:|---:|
| 0% | 5.02% | 0.1393 | 0.1393 | 0.4819 | 0.4819 |
| 5% | 9.52% | 0.1589 | 0.1685 | 0.5277 | 0.5371 |
| 10% | 14.03% | 0.1635 | 0.1638 | 0.5440 | 0.5365 |
| 20% | 23.01% | 0.1683 | 0.1779 | 0.5486 | 0.5405 |
| 30% | 32.00% | 0.1653 | 0.1786 | 0.5558 | 0.5627 |

The weighted/calibrated variant improves PR-AUC at most nonzero levels, but recall is lower at 10% and 20%. It is therefore inaccurate to claim uniform robustness benefits.

## Central confound

The clean training corpus is only about 5% positive. Symmetric label flips consequently convert many more passes to failures than failures to passes, raising positive prevalence to 32% at the largest setting. Increasing PR-AUC or recall as noise rises can reflect that changed task distribution rather than resilience to flakiness.

The synthetic score is generated per row and does not reproduce repeated intermittent behavior of a real test, environment-sensitive failures, ordering effects, or shared infrastructure faults. The committed corpus also has constant `hist_flaky_score` because its baseline universe was screened for stable tests.

## Scope of claims

The experiment supports comparisons only within its injected-noise protocol. It does not show that the primary trained model uses the same weights, that calibration always compensates for noise, or that selective abstention responds correctly to real flaky tests.

```bash
python scripts/evaluate.py --run flakiness
```

See `reports/flakiness_robustness.json`, [Task formulation](task_formulation.md), and [Limitations](limitations.md).
