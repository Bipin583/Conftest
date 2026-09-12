# Feature Ablation Study

The ablation study retrains models to compare feature groups. It measures the behavior of these retrained variants on the committed held-out split; it does not assign a causal percentage of performance to any one feature family.

## Protocol

`scripts/run_ablation_study.py` evaluates nine variants over the canonical 32-column schema:

- all features;
- leave out diff/churn, AST complexity, dependency graph, or history telemetry; and
- use each of those four groups alone.

Models train on the training split, use validation data for temperature calibration, and are evaluated on 183 held-out observations (86,469 rows). Recall is measured when each observation runs the top 25% of its own test universe, pooled over the 135 observations containing at least one failure.

## Committed results

| Variant | PR-AUC | ROC-AUC | Calibrated ECE | Recall@25% |
|---|---:|---:|---:|---:|
| Full, 32 features | 0.1393 | 0.8326 | 0.0289 | 0.4819 |
| Without diff/churn | 0.1393 | 0.8326 | 0.0289 | 0.4819 |
| Without AST | 0.1515 | 0.8512 | 0.0283 | 0.5087 |
| Without dependency graph | 0.0674 | 0.7136 | 0.0222 | 0.3267 |
| Without history | 0.2214 | 0.8770 | 0.1587 | 0.5234 |
| Diff/churn only | 0.0371 | 0.5000 | 0.0178 | 0.2101 |
| AST only | 0.0911 | 0.7605 | 0.2100 | 0.4059 |
| Dependency only | 0.2005 | 0.8590 | 0.1625 | 0.4757 |
| History only | 0.0557 | 0.6493 | 0.0324 | 0.3753 |

## Interpretation

The committed training artifact has 13 constant columns: all 12 diff/churn features plus `hist_flaky_score`. Removing diff/churn therefore changes nothing, while the diff-only model has no informative input and ROC-AUC 0.5.

Dependency features carry substantial ranking signal in this corpus: removing them lowers PR-AUC by 0.0719 and Recall@25% by 0.1552. However, their single-group model has poor calibrated ECE (0.1625), so ranking strength does not imply calibrated probability quality.

Removing history improves PR-AUC and recall but worsens calibrated ECE from 0.0289 to 0.1587. This is a trade-off, not evidence that history is simply beneficial or harmful. Removing AST also slightly improves these point estimates. Correlation, model refitting, calibration, and constant columns prevent additive “percentage contribution” interpretations.

## Reproduction and limits

```bash
python scripts/evaluate.py --run ablation
```

The authoritative report is `reports/ablation_study.json`. It contains point estimates rather than uncertainty intervals for differences, and it inherits the mutation-corpus and pre-row-bagging limitations. See [Feature schema](feature_schema.md), [Dataset](dataset.md), and [Limitations](limitations.md).
