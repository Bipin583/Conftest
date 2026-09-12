# ConfTest Developer Guide

This guide describes the maintained package, setup, extension points, and validation workflow. It does not redefine feature names, API schemas, settings, or measured results; links point to their authorities.

## Development setup

POSIX:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
```

Python and dependency constraints are defined in `pyproject.toml`. Run development commands from the repository root.

## Maintained layout

| Path | Responsibility |
|---|---|
| `src/conftest/repository/` | Git collection and repository metadata. |
| `src/conftest/tests/` | Pytest discovery and subprocess execution. |
| `src/conftest/features/` | Diff, AST, dependency, history, and canonical pipeline. |
| `src/conftest/models/` | LightGBM members, ensemble, calibration, and policy. |
| `src/conftest/engine/` | End-to-end selection orchestration. |
| `src/conftest/db/` | SQLAlchemy models, sessions, CRUD, and schema initialization. |
| `src/conftest/api/` | FastAPI application, routes, schemas, and dependencies. |
| `dashboard/` | Maintained Streamlit application. |
| `scripts/` | Training, evaluation, selection, and evidence producers. |
| `tests/` | Unit, integration, contract, and sample-suite tests. |

Top-level `src/features`, `src/models`, `src/engine`, and `src/dashboard` are legacy or compatibility surfaces, not parallel production implementations. `conftest.features.extractor` and `conftest.engine.abstention` intentionally re-export maintained objects for documented compatibility.

## Architectural contracts

- Feature order is exactly `conftest.features.pipeline.FEATURE_NAMES`; serialized models depend on it.
- Runtime settings belong in `conftest.config.Settings` and `.env.example`.
- Tuned thresholds belong only in `models/policy_config.json` and are produced by `scripts/tune_policy.py`.
- API behavior is defined by route decorators and Pydantic models under `conftest.api`.
- Quantitative claims require a report artifact and producer listed by `scripts/evaluate.py`.
- A missing metric or provenance field must remain unavailable rather than becoming a placeholder.

See [architecture](architecture.md), [feature schema](feature_schema.md), [API](api.md), and [database](database.md).

## Common extension points

### Add or change a feature

1. Implement extraction in the appropriate module under `src/conftest/features/`.
2. Update the pipeline and canonical `FEATURE_NAMES` ordering deliberately.
3. Update the readable [feature schema](feature_schema.md).
4. Retrain all artifacts whose input schema changed; old models are incompatible.
5. Add focused extraction, ordering, missing-context, and serialization tests.
6. Re-run calibration, policy tuning, and evaluation before publishing metrics.

Do not silently reorder columns or duplicate extraction logic in a compatibility module.

### Change model or calibration behavior

Keep training-only fitting separate from held-out evaluation. Preserve deterministic seeds and artifact metadata. If ensemble sampling changes, regenerate uncertainty, calibration, policy, and downstream reports together. The currently published reports predate the row-bagging fix and must not be described as measurements of a newly trained model.

### Change policy behavior

Modify the maintained policy implementation and its tests, then regenerate `models/policy_config.json` through `scripts/tune_policy.py`. Constructor defaults are not the shipped operating point. Test both fast selection and each full-suite fallback gate.

### Change the API

Update route code and `src/conftest/api/schemas.py`, then inspect `/openapi.json`. Update [api.md](api.md) and contract tests in the same change. Preserve explicit status codes and structured errors.

### Change persistence

Update SQLAlchemy models and CRUD together. Exercise a temporary database and consider migration/compatibility implications for existing `data/conftest.db` files. Never infer unobserved escapes from selective-only outcomes.

## Validation workflow

Run focused tests while editing, then the repository checks:

```bash
python scripts/check_no_fabricated_labels.py
python -m pytest tests/ -q
python -m ruff check src tests dashboard
```

Use `python scripts/evaluate.py` to inspect producer/artifact coverage without starting expensive stages. Use each script's `--help` before documenting a flag.

Useful focused checks include:

```bash
python -m pytest tests/unit/test_documented_import_paths.py -q
python -m pytest tests/unit/test_producer_references.py -q
python -m pytest tests/unit/test_config.py -q
python -m pytest tests/unit/test_provenance_refusals.py -q
```

The `Makefile` offers POSIX conveniences. Its cleanup commands use POSIX utilities and are not a portable Windows interface; prefer the Python commands above.

## Style and contribution discipline

Match the surrounding type hints, naming, exception handling, and comment density. Ruff and Black use the repository settings in `pyproject.toml`. Add tests for behavior changes and avoid static test-count claims in docs.

Before submitting a change:

1. Review the diff for generated files, credentials, datasets, and accidental artifact replacement.
2. Run the most focused tests plus full validation appropriate to the change.
3. Update only documentation backed by the new code or regenerated reports.
4. State failed or skipped validation explicitly.
5. Do not select or tune on the held-out test split.

## Experimental development

Read [reproducibility](reproducibility.md) before rebuilding data or models. Cross-repository and continuous-learning stages retrain repeatedly and are materially more expensive than inspecting existing artifacts.
