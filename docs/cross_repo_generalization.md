# Cross-Repository Generalization

ConfTest's leave-one-project-out experiment measures zero-shot transfer across the five mutation-harvest repositories. The result is mixed and does not show that cold start has been overcome.

## Protocol

For each target repository, `scripts/run_cross_repo_eval.py` trains on the other four repositories. Source observations are divided by commit identifier into model-fit and calibration subsets, with no observation present in both. The frozen model and calibrator are then evaluated on all rows from the unseen target repository without target fine-tuning.

Recall is measured with the top 25% of each target observation's own test universe. The report macro-averages repositories so each target counts once regardless of row count.

## Committed results

| Target | PR-AUC | ROC-AUC | Calibrated ECE | Recall@25% |
|---|---:|---:|---:|---:|
| `pathspec` | 0.1295 | 0.7074 | 0.2957 | 0.4668 |
| `pyjwt` | 0.1136 | 0.6841 | 0.2899 | 0.5546 |
| `sqlparse` | 0.1362 | 0.6999 | 0.1184 | 0.3728 |
| `tabulate` | 0.0868 | 0.4747 | 0.0826 | 0.3000 |
| `validators` | 0.1180 | 0.9359 | 0.0074 | 0.8842 |
| Macro mean | 0.1168 | 0.7004 | 0.1588 | 0.5157 |

Repository variation is substantial. `tabulate` is below random-order ROC-AUC at 0.4747, while `validators` has high ROC-AUC and recall but a much lower target failure rate (0.75%). Macro ECE of 0.1588 is weak relative to the within-corpus held-out calibration result.

## Interpretation

Shared feature names do not make distributions invariant. Repositories differ in suite size, positive prevalence, dependency topology, AST patterns, and available history. All 12 diff/churn columns are constant in this corpus, so this experiment cannot demonstrate transfer of change-level signals.

The study supports the statement that some ranking signal transfers in some source-target combinations. It does not justify immediate subset selection in a newly onboarded repository. A new project should start with full-suite or shadow-mode operation, gather local outcomes, recalibrate, and retune policy thresholds before using a fast operating point.

## Reproduction and limits

```bash
python scripts/evaluate.py --run cross-repo
```

This stage retrains per target and is expensive. Its labels remain mutation outcomes, not natural cross-company regressions. See `reports/cross_repo_generalization.json`, [Dataset](dataset.md), and [Limitations](limitations.md).
