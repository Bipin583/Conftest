# ConfTest Documentation Hub

This page is the stable entry point for ConfTest documentation. Detailed material is split by task and audience so that commands, schemas, and measured results have one maintained home.

## Start here

| Goal | Document |
|---|---|
| Install ConfTest and make a selection | [User manual](docs/user_manual.md) |
| Understand the system | [Architecture](docs/architecture.md) |
| Integrate with the REST API | [API reference](docs/api.md) |
| Operate or deploy the services | [Operations guide](docs/operations.md) |
| Develop or extend ConfTest | [Developer guide](docs/development.md) |
| Reproduce the experiment | [Reproducibility guide](docs/reproducibility.md) |
| Diagnose a failure | [Troubleshooting](docs/troubleshooting.md) |
| Interpret claims safely | [Limitations](docs/limitations.md) |
| Browse every guide | [Documentation index](docs/README.md) |

## Documentation contract

ConfTest documentation follows this authority order:

1. **Maintained behavior:** `src/conftest/`, script argument parsers, API route decorators and Pydantic schemas.
2. **Runtime configuration:** `src/conftest/config.py` and `.env.example`.
3. **Decision policy:** `models/policy_config.json`, produced by `scripts/tune_policy.py`.
4. **Feature order:** `src/conftest/features/pipeline.py::FEATURE_NAMES`.
5. **Quantitative results:** committed artifacts under `reports/`, together with their producer commands in `scripts/evaluate.py`.
6. **Explanation:** the focused guides linked from this hub.

When prose conflicts with an authority above, the authority wins and the prose is a documentation defect. `configs/default.yaml` is not the runtime settings source. Top-level legacy modules outside `src/conftest/` are not the maintained production path unless a guide explicitly says otherwise.

## Evidence policy

Published numbers must identify a committed artifact and its producing script. The correct summary of the shipped operating point is:

> On 183 held-out commits, ConfTest observed 100.00% failure recall and zero escaped commits while reducing test executions by 3.11%. It abstained to the full suite on 97.81% of commits and produced 0.0% measured wall-clock reduction.

This is an empirical result, not a formal guarantee. The committed reports predate a row-bagging fix, so retraining may change ensemble disagreement and the operating frontier. See [limitations](docs/limitations.md).

Inspect the current report map without recomputing results:

```bash
python scripts/evaluate.py
```

## System map

- **Collection:** `conftest.repository` mines commits, diffs and test history.
- **Discovery/execution:** `conftest.tests` discovers pytest nodes and executes selected nodes in a subprocess.
- **Features:** `conftest.features.pipeline.FeatureExtractionPipeline` composes diff, AST, dependency and historical features into the canonical 32-column order.
- **Model:** `conftest.models.ensemble.EnsembleUncertaintyPredictor` supplies mean probabilities and epistemic disagreement; a serialized calibrator maps probabilities after training.
- **Policy:** `conftest.models.policy.SelectivePredictionPolicy` applies tuned uncertainty, confidence, budget and OOD gates.
- **Orchestration:** `conftest.engine.selector_engine.ConfTestEngine` combines discovery, features, prediction, policy, optional execution and persistence.
- **Interfaces:** FastAPI (`conftest.api.main`), Streamlit (`dashboard/app.py`), standalone CLI (`scripts/select_tests.py`), GitHub Action and signed webhook.

Full trace: [architecture](docs/architecture.md).

## Core workflows

### Local selection

```bash
python scripts/select_tests.py \
  --repo-path tests/sample_suite \
  --commit-sha HEAD \
  --budget 0.25 \
  --output-json reports/local_selection.json
```

Use `--execute` only when the target tests are safe to run locally. Use `--persist-db` to store decisions. Complete option table: [user manual](docs/user_manual.md#selection-cli-reference).

### API and dashboard

```bash
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run dashboard/app.py --server.port 8501
```

Use `/health` or `/api/v1/health` for probes, `/docs` for generated OpenAPI UI, and the dashboard at port 8501.

### Evaluation

```bash
python scripts/evaluate.py                  # show stage/artifact map
python scripts/evaluate.py --run calibration
python scripts/evaluate.py --run baselines -- --bootstraps 2000
```

The held-out test split is for final reporting, not threshold selection. Read [reproducibility](docs/reproducibility.md) before regenerating artifacts.

## Configuration summary

Copy `.env.example` to `.env` and adjust only documented `CONFTEST_*` values. Runtime settings include API host/port/workers, database URL, artifact paths, default repository root and budget, CORS origins, GitHub secret/token, and data/model directories. Tuned policy thresholds do **not** belong in `.env`; they live in `models/policy_config.json`.

Production startup intentionally fails with the public development webhook secret or wildcard credentialed CORS. See [operations](docs/operations.md#production-security).

## Verification

```bash
python scripts/check_no_fabricated_labels.py
python -m pytest tests/ -q
python -m ruff check src tests dashboard
```

The test suite includes guards for documented import paths, producer references, configuration drift, provenance refusals and legacy module status. Contributors should update those contracts when an authoritative interface changes.

## Complete topic index

The [docs index](docs/README.md) links installation, architecture, features, uncertainty, calibration, selective prediction, explainability, datasets, statistics, evaluations, economics, deployment, GitHub integration, dashboard, limitations, viva material and troubleshooting.
