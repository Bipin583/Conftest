# Selective Prediction Policy

ConfTest can decline to choose a subset. This converts model confidence and disagreement into one of two actions:

```text
FAST_SELECTED    rank tests and take a budget-limited subset
SAFE_FULL_SUITE  abstain from subsetting and retain every discovered test
```

`SAFE_FULL_SUITE` reduces the risk introduced by selection, but neither branch is a formal bug-free guarantee.

## Executable decision rule

`src/conftest/models/policy.py::SelectivePredictionPolicy` receives calibrated test scores, per-test ensemble standard deviations, diff size, and a budget. It computes maximum calibrated score and maximum uncertainty and abstains if any of these conditions holds:

- maximum uncertainty exceeds `tau_abstain`;
- maximum score is below `tau_conf`; or
- changed-file count or total churn exceeds the configured out-of-distribution limit.

On abstention, all candidate test IDs are returned. In fast mode, scores are sorted descending and the first `max(1, int(total_tests * budget_ratio))` IDs are selected. This is floor-based integer conversion with a minimum of one—not a ceiling rule.

Thresholds and diff limits for the shipped operating point are loaded from `models/policy_config.json`. The values in the Python constructor are explicitly untuned placeholders. When no tuned artifact is available, the maintained fallback is an always-abstain policy rather than arbitrary subset selection.

## Tuning protocol

`scripts/tune_policy.py` searches candidate policy settings on the validation split. Policy selection and model evaluation must remain separate:

1. fit the base ensemble on training data;
2. fit/select calibration on validation data;
3. tune the policy on validation commits against a named objective and constraints;
4. evaluate the frozen artifacts once on held-out test commits.

Choosing thresholds from held-out outcomes would leak evaluation information and overstate generalization. Use the exact artifact producer:

```bash
python scripts/tune_policy.py
```

## Committed held-out outcomes

At the shipped operating point, `reports/baseline_comparison.csv` records 183 held-out commits (86,469 commit-test rows):

| Metric | Observed result |
|---|---:|
| Failure recall | 100.00% |
| Escaped commits | 0 of 183 |
| Test-execution reduction | 3.11% |
| Abstention rate | 97.81% |
| Measured wall-clock reduction | 0.0% |

A validation-selected higher-reduction operating point reached 32.85% execution reduction but only 87.69% held-out failure recall and 15 escaped commits, failing the project's 95% recall gate. These comparisons demonstrate why reduction alone is not the deployment objective.

The zero observed escapes at the shipped point mean only that none occurred in this committed held-out evaluation. The result is coupled to very high abstention and cannot be generalized into “zero escapes guaranteed.” Reports also predate the effective row-bagging fix; see [Uncertainty estimation](uncertainty_estimation.md).

## Metric meanings

- **Failure recall**: detected failing tests divided by all failing tests in records with complete ground truth.
- **Escaped commit**: a commit for which at least one failing test was omitted.
- **Test-execution reduction**: reduction in number of executed tests; it assumes no duration weighting.
- **Wall-clock reduction**: reduction computed from measured durations where both selected and full-suite durations are available.
- **Abstention rate**: fraction of commits assigned `SAFE_FULL_SUITE`.

Code fields named `estimated_time_saved_pct` or `estimated_saving` currently carry a count reduction in the selection path. Interfaces and papers must not relabel that value as observed time saving.

## Operational interpretation

A deployment should use a policy only with the ensemble and calibrator for which it was tuned, retain mandatory tests outside the learned subset where required, and run periodic complete-suite audits to measure misses. Changing the dataset, feature schema, ensemble, calibrator, budget, or risk requirement invalidates the existing operating-point evidence and requires retuning and reevaluation.

See [Statistical methodology](statistical_methodology.md), [Economic analysis](economic_cost_benefit.md), and [Limitations](limitations.md).
