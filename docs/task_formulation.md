# Prediction Task Formulation

ConfTest ranks candidate tests for one change and may abstain from selecting a subset. The learned score is an empirical test-failure score, not a proof that an omitted test will pass.

## Unit of prediction

For change observation `c`, candidate test `t`, and canonical feature vector `x(c,t) ∈ R^32`, the model estimates:

```text
p_hat(c,t) = model score for y(c,t) = 1 given x(c,t)
```

The serving ensemble averages five member scores. A separately fitted calibrator maps that mean score, and the selective policy consumes calibrated scores plus ensemble disagreement. See [Feature schema](feature_schema.md), [Calibration](confidence_calibration.md), and [Selective prediction](selective_prediction.md).

## Committed label semantics

Although `src/conftest/models/task_definition.py::LabelDefinition` names direct-failure, descendant-failure, and replay-oracle formulations, the committed training and evaluation corpus uses measured mutation outcomes. One AST mutant is applied to a repository checkout and the screened pytest universe is executed:

```text
y(c,t) = 1  if test t kills mutant c
y(c,t) = 0  otherwise
```

This is replay-oracle-style evidence for injected faults. It must not be described as an observed production-regression label. Mutants with untrustworthy execution outcomes are refused by the dataset producer; undetected mutants retain all-zero rows. See [Dataset](dataset.md).

## Class imbalance

The training split is 5.017% positive. `LightGBMTestPredictor.train()` computes:

```text
scale_pos_weight = N_negative / N_positive
```

and passes it to LightGBM. The task-definition module can also construct per-row class/flakiness weights, but the maintained trainer does not apply those weights unless a caller explicitly supplies `sample_weight`. The committed primary result therefore must not be described as using a fixed flaky-test discount.

The flakiness stress experiment is a separate controlled caller. It injects training-label flips and uses `clip(1 - 0.7 * synthetic_flakiness_score, 0.1, 1.0)` for its robust variant. That experimental rule is not the general serving policy. See [Flakiness robustness](flakiness_robustness.md).

## Split and tuning discipline

The mutation sequence is ordered by a documented harvest-order surrogate, then partitioned by whole observations:

1. earliest 70%: base-model training;
2. next 15%: early stopping, calibration selection, and policy tuning as defined by each producer;
3. latest 15%: held-out evaluation.

Historical features are accumulated causally before the current mutant's labels are recorded. The timestamps are synthetic ordering keys, not dates of natural commits. Cross-repository evaluation uses a different leave-one-project-out protocol and states its own calibration split. See [Dataset](dataset.md) and [Cross-repository generalization](cross_repo_generalization.md).

## Decision objective

Ranking quality alone is insufficient. The deployment-facing operating point balances:

- failure recall on records with complete ground truth;
- test-count and measured-duration reduction;
- escaped commits;
- abstention rate; and
- calibration and uncertainty diagnostics.

Thresholds are selected on validation data under a named policy objective and then frozen for held-out measurement. The zero observed escapes at the shipped point are a held-out observation coupled to 97.81% abstention, not a formal safety guarantee.
