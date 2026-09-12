# System Architecture

ConfTest is a research-grade regression-test-selection system for Python repositories. The maintained implementation is under `src/conftest/`; top-level packages such as `src/features`, `src/models`, and `src/engine` are legacy compatibility surfaces and are not the authority for current behavior.

## End-to-end path

```text
commit, pull request, or API diff
  -> pytest test discovery
  -> diff + test-file AST + static dependency + prior-history features
  -> fixed 32-column feature vector
  -> five LightGBM member scores
  -> mean risk score + ensemble disagreement
  -> post-hoc calibrator
  -> tuned selective policy
       -> FAST_SELECTED: rank and take a budget-limited subset
       -> SAFE_FULL_SUITE: abstain and retain every discovered test
  -> optional pytest execution
  -> optional SQLAlchemy persistence
  -> CLI / FastAPI / Streamlit / GitHub reporting
```

The two policy outcomes describe actions, not guarantees. The committed held-out evaluation observed zero escaped commits at the shipped operating point, but that point abstained on 97.81% of commits and measured no wall-clock reduction. See [Selective prediction](selective_prediction.md) and [Limitations](limitations.md).

## Ingestion and test discovery

`ConfTestEngine` in `src/conftest/engine/selector_engine.py` coordinates a request. Inputs include a repository path, commit/ref, changed-file records, commit message, and budget. The CLI can mine a local Git diff; the selection API requires a non-empty `changed_files` list; the webhook obtains pull-request files through the GitHub API and recommends the full suite when it cannot retrieve them.

Tests are collected as pytest node IDs by the maintained discovery and runner services under `src/conftest/tests/`. Empty or failed discovery cannot produce a meaningful ranked subset and is handled as a failure/fallback condition rather than by inventing candidates.

## Feature extraction

`src/conftest/features/pipeline.py::FeatureExtractionPipeline` composes four groups:

1. commit diff and message metrics;
2. AST metrics for each candidate test file;
3. static dependency-graph relationships between the test and changed paths; and
4. execution history strictly preceding the target timestamp.

The exact names and order live in `FEATURE_NAMES`; [Canonical feature schema](feature_schema.md) is the descriptive reference. Cold-start history is explicitly filled, so absence of history is distinguishable in the implementation from a populated database even though most fallback values are zero.

## Prediction, calibration, and uncertainty

`src/conftest/models/ensemble.py` loads the configured ensemble directory and aggregates member predictions. Their mean supplies the raw risk ranking and their disagreement supplies an epistemic-uncertainty signal. Post-hoc calibration is loaded from `CONFTEST_CALIBRATOR_PATH`.

The committed artifacts are a five-member LightGBM ensemble with temperature scaling, but documentation must not hard-code fitted parameters as runtime defaults. Artifact metadata and `reports/calibration_report.json` describe a particular training run; deployment paths come from `src/conftest/config.py`. The committed reports also predate the trainer's effective row-bagging fix. See [Uncertainty estimation](uncertainty_estimation.md) and [Confidence calibration](confidence_calibration.md).

## Selective policy

`src/conftest/models/policy.py::SelectivePredictionPolicy` evaluates calibrated confidence, aggregate uncertainty, and related tuned conditions. `src/conftest/engine/selector_engine.py::ConfTestEngine` applies the result:

- `FAST_SELECTED` ranks tests and selects up to the requested budget.
- `SAFE_FULL_SUITE` represents abstention and selects/recommends all discovered tests.

The shipped thresholds are read from `models/policy_config.json`, produced by `scripts/tune_policy.py`. Constructor placeholders and stale YAML values are not the operating point.

## Execution and persistence

Execution is optional and uses the maintained pytest runner service under `src/conftest/tests/`. Selection without `execute` reports a decision but does not measure test outcomes or wall-clock saving.

SQLAlchemy models in `src/conftest/db/models.py` persist repositories, commits, diffs, test cases/runs, features, predictions, decisions, and outcomes. SQLite is the configured and tested default, with foreign keys and WAL enabled. Outcome fields requiring a full-suite comparison remain null unless `ground_truth_complete` is true; selective execution alone cannot establish missed-failure counts. See [Database](database.md).

## Interfaces

- `scripts/select_tests.py`: local selection, optional execution, JSON output, and persistence.
- `src/conftest/api/main.py`: FastAPI application; see [API](api.md).
- `dashboard/app.py` and `dashboard/pages/`: Streamlit views over committed reports and persisted data.
- `.github/workflows/conftest.yml`: checked-in GitHub Actions workflow using the CLI.
- `POST /api/v1/github/webhook`: signature-verified webhook mode with optional PR commenting.

GitHub Actions and webhook mode are separate integrations. Neither changes the policy's evidentiary limits.

## Configuration boundaries

Runtime settings come from `src/conftest/config.py` and `CONFTEST_*` environment variables. Model paths select immutable artifacts; policy thresholds come from the policy artifact. Evaluation producers under `scripts/` write reports but are not invoked automatically by the serving API. `scripts/evaluate.py` maps evaluation stages to their producers and outputs.
