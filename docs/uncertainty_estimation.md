# Uncertainty Estimation

ConfTest uses disagreement among LightGBM ensemble members as an epistemic-uncertainty proxy. It does not directly estimate all sources of uncertainty, and a low disagreement value is not proof that a prediction is correct.

## Ensemble outputs

For member probabilities `p_1 ... p_M` on one `(commit, test)` feature vector, `src/conftest/models/ensemble.py` computes:

- mean score: `mean(p_m)`;
- epistemic standard deviation: `std(p_m)`;
- epistemic variance: `var(p_m)`; and
- binary entropy of the mean score.

The committed/default configuration uses five seeds (`42`, `101`, `2024`, `777`, `999`). At commit level, the implementation reports maximum, mean, and 95th-percentile test-level standard deviation plus maximum and mean failure score. The selective policy currently uses the maximum member standard deviation across candidate tests.

Predictive entropy and ensemble disagreement answer different questions. Entropy is high when the mean score is near 0.5 even if every member agrees; disagreement is high when members produce different scores. Neither measure isolates test flakiness or distribution shift by itself.

## How diversity is created

The current trainer configures different random seeds, feature subsampling (`colsample_bytree`), and row bagging (`subsample` with a positive `subsample_freq`). Effective row bagging requires both `subsample < 1` and `subsample_freq > 0`; setting only `subsample` does not enable LightGBM bagging.

### Committed-artifact caveat

The checked-in ensemble metadata predates the `subsample_freq` fix. Those members were trained on all rows even though a `subsample` ratio was recorded. Their remaining diversity comes from randomness that was actually enabled (including feature subsampling), and their disagreement may be narrower than that of a newly trained bagged ensemble. Current loading code detects legacy metadata and warns; published reports must not be described as measurements of the corrected trainer until the artifacts and evaluations are regenerated.

## Policy use

For candidate tests `T(c)`, the policy uses:

```text
U(c) = max(std(member predictions for test t)) for t in T(c)
```

It abstains to `SAFE_FULL_SUITE` if `U(c)` exceeds the tuned uncertainty threshold, if maximum calibrated confidence is below its threshold, or if the diff exceeds configured file/churn limits. Otherwise, it enters `FAST_SELECTED`. Exact shipped thresholds come from `models/policy_config.json`, not from constructor defaults or prose. See [Selective prediction](selective_prediction.md).

## What the signal does not establish

- Member agreement can be confidently wrong, especially when members share data, features, model family, and biases.
- A constant or information-poor feature space can suppress meaningful diversity.
- Static Python analysis misses dynamic imports, monkey patching, generated code, runtime dispatch, and environment behavior.
- New repositories can map to familiar-looking vectors while still being out of distribution.
- Aleatoric effects such as flaky tests are not removed merely by computing ensemble variance.

For those reasons, thresholds are tuned empirically on validation data and evaluated on held-out commits. Results are observations for those artifacts/splits, not a universal uncertainty guarantee. See [Limitations](limitations.md), [Cross-repository generalization](cross_repo_generalization.md), and [Flakiness robustness](flakiness_robustness.md).

## Reproduction

The ensemble is produced by:

```bash
python scripts/train_ensemble.py
```

Uncertainty diagnostics are produced through the `uncertainty` stage listed by:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --run uncertainty
```

Retraining changes the uncertainty distribution, so policy thresholds must be retuned and held-out reports regenerated after replacing the ensemble.
