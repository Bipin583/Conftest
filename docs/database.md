# Database Schema and Evidence Semantics

`src/conftest/db/models.py` is the schema authority. ConfTest uses SQLAlchemy 2.0 and defaults to SQLite. `src/conftest/db/session.py` enables SQLite foreign keys and WAL mode and creates one request-scoped session for FastAPI.

The configured `database_url` is passed to SQLAlchemy, but this repository's documented and exercised deployment path is SQLite. The use of portable-looking column types is not evidence of tested PostgreSQL compatibility.

## Entity relationships

```text
Repository 1---* Commit 1---* ChangedFile
     |             | 1---* TestRun *---1 TestCase *---1 Repository
     |             | 1---* FeatureRecord *---1 TestCase
     |             | 1---* Prediction *---1 TestCase
     |             | 1---0..1 SelectionDecision
     |             ` 1---0..1 Outcome
     `-------------* TestCase
```

Deleting a repository cascades through commits and test cases; deleting a commit cascades through its dependent records.

## Tables

### `repositories`

`id` integer primary key; unique indexed `full_name`; required `url`, `language` (default `python`), `default_branch` (default `main`), and `local_path`; `created_at` UTC application timestamp.

### `commits`

`id`; indexed `repository_id`; globally unique indexed `sha`; nullable `parent_sha`, `author_hash`, and `message`; required indexed `timestamp`; `ci_status` (default `pending`); `total_duration` (default `0.0`); `created_at`.

The global SHA uniqueness means the same Git object cannot currently be represented under two repository rows.

### `changed_files`

`id`; indexed `commit_id`; required `file_path`; `change_type` (default `MODIFIED`); `lines_added`, `lines_deleted`, and `cyclomatic_complexity` with zero defaults.

### `test_cases`

`id`; indexed `repository_id`; indexed `test_id`; required `test_path` and `test_function`; `framework` (default `pytest`); `average_duration` and `flaky_indicator`; `created_at`. The pair `(repository_id, test_id)` is unique.

### `test_runs`

`id`; indexed `commit_id` and `test_case_id`; required `status`; `duration`; `retry_count`; `source` (default `ci`); `executed_at`. Expected status vocabulary in current code is `PASSED`, `FAILED`, `SKIPPED`, or `ERROR`; source examples are `ci`, `local`, and `replay`.

### `feature_records`

`id`; indexed `commit_id` and `test_case_id`; required JSON `feature_vector`; `created_at`. `(commit_id, test_case_id)` is unique. The JSON object must follow the [canonical feature schema](feature_schema.md); the database does not enforce names or array order.

### `predictions`

`id`; indexed `commit_id` and `test_case_id`; required `raw_score`, `uncertainty`, `calibrated_confidence`, and `model_version`; `created_at`. `(commit_id, test_case_id)` is unique.

### `selection_decisions`

`id`; unique indexed `commit_id`; required `mode`, `abstained`, `uncertainty_score`, `threshold_used`, selected/total counts; `estimated_saving`; nullable JSON `reasons`; `created_at`.

`estimated_saving` is currently populated as test-count reduction (`1 - selected_count / total_count`). Despite the historical column name, it is not measured duration saving.

### `outcomes`

`id`; unique indexed `commit_id`; indexed `ground_truth_complete`; always-observable `detected_failures` and `selected_duration`; nullable `actual_failures`, `missed_failures`, `full_duration`, and `time_reduction_ratio`; `created_at`.

The nullable fields are measurements only when `ground_truth_complete` is true. A selective run observes executed tests but cannot determine whether an omitted test would have failed or how long the complete suite would have taken. Consumers must not coerce null missed failures or savings to zero.

## Integrity constraints

The schema enforces one feature record and prediction per `(commit, test)`, one decision and outcome per commit, and one test ID per repository. Foreign keys use `ON DELETE CASCADE`; SQLite enforcement depends on the connection hook in `session.py` and must remain enabled.

The ORM does not add database `CHECK` constraints for status vocabulary, probabilities, counts, or mode strings. Application schemas and services perform part of that validation, so direct database writers must preserve the same invariants.

## Initialization and configuration

Initialize the configured database with:

```bash
python -m conftest.db.init_db
```

The default URL and override key are defined by `src/conftest/config.py` and `.env.example`; use `CONFTEST_DATABASE_URL`. The API initializes tables during startup, but explicit initialization is useful for CLI-only workflows.

## Evidence-safe querying

Safety and efficiency aggregates require different evidence:

- Count reduction comes from `selection_decisions`.
- Observed selected duration comes from `outcomes.selected_duration`.
- Missed failures, full duration, and wall-clock reduction require `outcomes.ground_truth_complete = true`.
- An empty aggregate is absent evidence, not zero; the analytics API returns null for averages or missed failures it cannot establish.

Back up the SQLite database together with model/policy identifiers if decisions must remain auditable. A stored prediction without its corresponding artifact version and feature contract is not independently reproducible.
