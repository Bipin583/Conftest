# ConfTest Documentation Index

This index organizes the maintained documentation by audience and task. Begin with the shortest path that matches your goal; use the technical references when implementing or reviewing a specific subsystem.

## Learning paths

### Users

1. [User manual](user_manual.md): install ConfTest, verify artifacts, select tests, and interpret a decision.
2. [Troubleshooting](troubleshooting.md): diagnose setup, discovery, artifact, API, database, and integration failures.
3. [Dashboard guide](dashboard.md): understand the interactive pages and their report-backed metrics.
4. [Limitations](limitations.md): decide whether the evidence supports use on your repository.

### Developers

1. [Developer guide](development.md): set up the repository, understand maintained package boundaries, extend components, and validate changes.
2. [Architecture](architecture.md): trace a request through collection, features, prediction, policy, execution, and persistence.
3. [Feature schema](feature_schema.md): inspect the canonical ordered 32-feature contract.
4. [API reference](api.md) and [database reference](database.md): modify interfaces and persistence safely.

### Operators and integrators

1. [Operations guide](operations.md): configure, start, monitor, back up, and upgrade the services.
2. [GitHub integration](github_integration.md): choose between the checked-in Action and signed webhook mode.
3. [Docker deployment](deployment_docker.md): run the API and dashboard containers.
4. [Troubleshooting](troubleshooting.md): resolve common runtime and integration faults.

### Researchers and academic reviewers

1. [Reproducibility guide](reproducibility.md): follow the data-to-report evidence chain without leaking the held-out test split into tuning.
2. [Task formulation](task_formulation.md), [dataset](dataset.md), and [experiments](experiments.md): understand the prediction task, samples, splits, and comparisons.
3. [Uncertainty](uncertainty_estimation.md), [calibration](confidence_calibration.md), and [selective prediction](selective_prediction.md): review the model and decision methodology.
4. [Statistical methodology](statistical_methodology.md) and the focused evaluation guides below.
5. [Limitations](limitations.md): interpret results, confounders, and maturity.

## Task-oriented guides

| Task | Primary document | Related reference |
|---|---|---|
| Install and run locally | [User manual](user_manual.md) | [Troubleshooting](troubleshooting.md) |
| Configure and operate services | [Operations](operations.md) | [Docker deployment](deployment_docker.md) |
| Call the API | [API](api.md) | [Architecture](architecture.md) |
| Use the dashboard | [Dashboard](dashboard.md) | [User manual](user_manual.md) |
| Integrate GitHub | [GitHub integration](github_integration.md) | [Operations](operations.md) |
| Add or change a feature | [Developer guide](development.md) | [Feature schema](feature_schema.md) |
| Rebuild experimental reports | [Reproducibility](reproducibility.md) | [Experiments](experiments.md) |
| Quote a project result | [Limitations](limitations.md) | [Reproducibility](reproducibility.md) |
| Prepare for project review | [Viva guide](viva.md) | [Architecture](architecture.md) |

## Technical reference

- [Architecture](architecture.md): maintained execution paths and subsystem boundaries.
- [Feature schema](feature_schema.md): exact names, order, meanings, and missing-history behavior.
- [API](api.md): routes, request/response models, errors, and examples.
- [Database](database.md): SQLAlchemy entities, relations, and persistence semantics.
- [Dashboard](dashboard.md): page inputs, reports, and refusal behavior.
- [GitHub integration](github_integration.md): Action and webhook flows.
- [Explainability](explainability.md): Tree SHAP outputs and interpretation.

## Methodology and evaluation

- [Dataset](dataset.md)
- [Task formulation](task_formulation.md)
- [Uncertainty estimation](uncertainty_estimation.md)
- [Confidence calibration](confidence_calibration.md)
- [Selective prediction](selective_prediction.md)
- [Statistical methodology](statistical_methodology.md)
- [Experiments](experiments.md)
- [Feature ablation](feature_ablation.md)
- [Flakiness robustness](flakiness_robustness.md)
- [Flaky test detection](flaky_test_detection.md)
- [Latency benchmarks](latency_benchmarks.md)
- [Economic cost-benefit](economic_cost_benefit.md)
- [Cross-repository generalization](cross_repo_generalization.md)
- [Continuous learning](continuous_learning.md)
- [Limitations](limitations.md)

## Documentation sources of truth

| Subject | Authority |
|---|---|
| Runtime behavior | `src/conftest/` |
| Script commands and flags | Each script's `argparse` declarations |
| API routes and payloads | `src/conftest/api/main.py`, route decorators, `api/schemas.py` |
| Runtime settings | `src/conftest/config.py`, `.env.example` |
| Tuned decision thresholds | `models/policy_config.json` |
| Feature names and order | `src/conftest/features/pipeline.py::FEATURE_NAMES` |
| Quantitative claims | `reports/` plus the producer map in `scripts/evaluate.py` |
| Package dependencies | `pyproject.toml` |

Generated prose is subordinate to these sources. `configs/default.yaml` is not the runtime-settings authority, and compatibility modules outside the maintained `src/conftest/` path must not be presented as independent implementations.

## Evidence and wording rules

- Say **observed** when reporting failure recall or escaped commits; ConfTest provides no formal bug-free guarantee.
- Name the split, sample unit, artifact, and producer for quantitative claims.
- Do not choose thresholds on the held-out test split.
- Do not replace missing report values with examples or placeholders.
- Distinguish test-execution reduction from wall-clock reduction.
- State that current reports predate the trainer's row-bagging fix when discussing the shipped model.

## Compatibility entry points

The repository-level [documentation hub](../DOCUMENTATION.md) is the stable overview. The former monolithic `DOCUMENTATION_DETAILED.md` is retained only as a pointer to this index so existing links do not break.
