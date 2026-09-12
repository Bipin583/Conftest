# Continuous Learning and Drift Evaluation

ConfTest implements Page-Hinkley detection and replay retraining as an experimental subsystem. It is not connected to the default selection-serving path as an autonomous production loop, and the committed evaluation does not show that adaptation improves performance.

## Implemented mechanism

`src/conftest/models/continuous_learning.py` provides:

- `PageHinkleyDriftDetector`, which accumulates increases in prediction error above a running minimum; and
- `OnlineContinualLearner`, which stores recent rows in a replay buffer and retrains a LightGBM predictor when the detector signals.

The detector is one-sided: shifts that lower error are invisible by construction. The learner requires observed labels, so selective execution alone cannot provide complete outcomes for omitted tests.

## Evaluation protocol

`scripts/run_continuous_learning.py` uses real harvested mutant rows. For each repository it initializes on 40 mutants and streams 120-mutant segments. Cross-repository shifts are created by arranging source and target segments adjacently; rows and labels are measured, while only their adjacency is constructed.

The report evaluates thresholds `0.5, 1, 2, 5, 10, 15`. It intentionally selects no threshold. Error is mean absolute probability error across all test rows for one mutant. The evaluation uses a 5,000-row replay buffer and 30-tree adaptation models.

## Detector trade-off

| Threshold | False alarms over 600 stationary mutants | Shifts detected | Median latency |
|---:|---:|---:|---:|
| 0.5 | 11 | 4/5 | 4.5 mutants |
| 1.0 | 6 | 3/5 | 18 mutants |
| 2.0 | 4 | 2/5 | 33.5 mutants |
| 5.0 | 0 | 2/5 | 44.5 mutants |
| 10.0 | 0 | 0/5 | undefined |
| 15.0 | 0 | 0/5 | undefined |

More sensitive thresholds detect more worsening shifts sooner but generate stationary false alarms. The `tabulate -> validators` shift lowers mean error and is not detected by the one-sided statistic.

## Adaptation outcome

In the three stationary repositories where no-adaptation and more-adaptation runs are comparable, more adaptation increased mean error:

- `pyjwt`: +0.0756;
- `sqlparse`: +0.1240;
- `tabulate`: +0.1778.

This does not establish that retraining is always harmful; it establishes that benefit was not demonstrated by this configuration. Detector activation and adaptation quality are separate questions.

## Operational requirements

A production loop would still need complete labels or audits, a threshold selected against explicit false-alarm cost, retraining approval, artifact versioning, rollback, and new calibration and policy tuning after every model replacement. Silent replacement would invalidate the existing operating-point evidence.

```bash
python scripts/evaluate.py --run continuous-learning
```

See `reports/continuous_learning.json`, [Architecture](architecture.md), and [Limitations](limitations.md).
