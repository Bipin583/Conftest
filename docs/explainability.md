# Explainability

ConfTest exposes local tree-model attributions and rule-based reason cards. These outputs explain how a configured model produced a score; they do not prove that the score, dependency graph, or selected subset is correct.

## SHAP implementation

`src/conftest/explainability/shap_explainer.py::ShapExplainer` wraps SHAP `TreeExplainer` for one trained `LightGBMTestPredictor`. It uses the exact 32-name order from `src/conftest/features/pipeline.py::FEATURE_NAMES` and returns:

- the model's predicted probability;
- the explainer's base expected value;
- positive and negative feature contributions ranked by absolute magnitude; and
- all per-feature attributions for dataset/global aggregation.

A positive attribution increases the model output relative to its baseline; a negative attribution decreases it. Attribution sign is local to that model/input. It is not a causal statement about what would happen if source code changed.

The API endpoint `POST /api/v1/explain` resolves the first model file named in the serving ensemble metadata. Thus its SHAP values explain one ensemble member, not the ensemble mean or the post-calibration transformation. The endpoint response names its probability generically, so consumers must not describe it as a calibrated ensemble probability. See [API](api.md).

## Rule-based reason cards

`src/conftest/explainability/rules.py::RuleBasedExplainer` converts feature values and leading SHAP drivers into developer-facing reasons and a risk category. The same formatter builds commit Markdown summaries used by API and integration paths.

Reason cards are summaries of available signals. A statement such as “directly coupled” inherits the limits of the static dependency analysis; historical reasons inherit the completeness and anti-leakage properties of stored runs.

## Missing artifacts and integrity

Explainability is artifact-bound. If ensemble metadata is missing, unreadable, empty, or names a missing member, the API returns 503 and names `python scripts/train_ensemble.py` as the producer. It does not explain a different model as a fallback. This matters because plausible-looking attributions from the wrong artifact would be fabricated evidence.

Input feature dictionaries should follow the [canonical feature schema](feature_schema.md). The API zero-fills missing canonical keys, which makes the request executable but can produce an explanation of an unintended vector.

## Global importance

`explain_dataset()` computes mean absolute SHAP magnitude per feature over supplied rows. This ranks model reliance within that sample, not standalone predictive contribution. Correlated features can divide or transfer importance, and constant columns cannot express variation. Use retrained [Feature ablation](feature_ablation.md) for intervention-style comparison, while retaining its own confounders.

## Integration boundaries

The checked-in GitHub Actions workflow can post a selection summary generated from CLI JSON. Webhook mode can generate and post a reason summary when a GitHub token is available. These are separate flows; neither guarantees that a comment will be posted. The webhook reports `comment_posted` and abstains if it cannot retrieve a real PR diff.

## Practical reading guide

Use explanations to audit surprising rankings, inspect feature/data defects, and communicate why a model acted. Do not use a reason card as evidence that omitted tests are irrelevant, as a causal diagnosis of a regression, or as a substitute for complete-suite verification. See [Architecture](architecture.md) and [Limitations](limitations.md).
