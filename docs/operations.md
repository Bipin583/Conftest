# ConfTest Operations Guide

This guide covers runtime configuration, service startup, health checks, persistence, security, backups, and upgrades. ConfTest is a research prototype; validate it on the target repository and keep full-suite fallback enabled.

## Configuration authority

Copy `.env.example` to `.env` and edit only keys implemented by `src/conftest/config.py`. Every runtime variable uses the `CONFTEST_` prefix.

| Area | Settings |
|---|---|
| Environment | `CONFTEST_ENV`, `CONFTEST_DEBUG`, `CONFTEST_LOG_LEVEL` |
| API | `CONFTEST_API_HOST`, `CONFTEST_API_PORT`, `CONFTEST_API_WORKERS` |
| Database | `CONFTEST_DATABASE_URL`, `CONFTEST_DB_ECHO` |
| Artifacts | `CONFTEST_ENSEMBLE_PATH`, `CONFTEST_CALIBRATOR_PATH`, `CONFTEST_POLICY_CONFIG_PATH` |
| Defaults | `CONFTEST_DEFAULT_REPO_ROOT`, `CONFTEST_DEFAULT_BUDGET_RATIO` |
| Browser access | `CONFTEST_CORS_ALLOW_ORIGINS` |
| GitHub | `CONFTEST_GITHUB_WEBHOOK_SECRET`, `CONFTEST_GITHUB_TOKEN` |
| Storage roots | `CONFTEST_DATA_DIR`, `CONFTEST_MODELS_DIR` |

`CONFTEST_DEFAULT_BUDGET_RATIO` is a request fallback, not a safety threshold. Tuned `tau_abstain`, `tau_conf`, budget, and OOD limits are loaded from `models/policy_config.json`. Do not copy stale values from `configs/default.yaml` into runtime configuration.

## Local startup

Initialize the SQLite schema:

```bash
python -m conftest.db.init_db
```

Start the API:

```bash
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard separately:

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

The API also initializes the schema during startup. For multi-process deployments, verify SQLite locking behavior under the expected write load before raising `CONFTEST_API_WORKERS`.

## Health and readiness

Probe either health alias:

```text
GET /health
GET /api/v1/health
```

OpenAPI is available at `/openapi.json`; interactive UIs are `/docs` and `/redoc`. A healthy HTTP response does not prove that every model/report artifact exists or that target-repository test execution is safe. Perform a non-executing selection smoke test after deployment.

## Artifact deployment

Deploy the ensemble directory, calibrator file, and policy JSON as one validated set. The policy depends on the model's uncertainty distribution; mixing artifacts from different training runs invalidates that relationship.

Before switching versions:

1. Stage artifacts at new paths.
2. Confirm feature schema and model metadata compatibility.
3. Run a non-executing selection and relevant report checks.
4. Update all three artifact path settings together.
5. Restart workers and inspect health and startup logs.
6. Retain the prior artifact set for rollback.

Never replace a missing calibrator or policy with an arbitrary object or constructor default.

## Production security

Set `CONFTEST_ENV=production` only after replacing development defaults. Configuration validation intentionally refuses startup when:

- `CONFTEST_GITHUB_WEBHOOK_SECRET` is the public development placeholder.
- credentialed CORS uses wildcard origin `*`.

Use a high-entropy webhook secret supplied by the deployment secret store. Set `CONFTEST_CORS_ALLOW_ORIGINS` to an explicit JSON list such as `[
"https://conftest.example.org"]`. Give `CONFTEST_GITHUB_TOKEN` only the repository permissions required to fetch PR files or post the configured response; do not commit it to `.env`, Compose files, logs, or reports.

Webhook requests must carry GitHub's `X-Hub-Signature-256` HMAC SHA-256 signature. If PR changed files cannot be fetched, the maintained behavior is an HTTP 202 safe full-suite recommendation, not a fabricated diff.

Bind directly to `127.0.0.1` unless a reverse proxy or container network requires broader exposure. Terminate TLS and enforce authentication/authorization at the deployment boundary; the project should not be treated as a hardened public multi-tenant service.

## Database and backups

The default URL is `sqlite:///./data/conftest.db`, with WAL behavior configured by the database layer. Back up the database only after coordinating writers so the main database and WAL state are consistent.

A conservative file-level procedure is:

1. Stop API, dashboard actions that write, and selector jobs using `--persist-db`.
2. Copy the database and any existing `-wal`/`-shm` companions together, or use SQLite's online backup facility.
3. Record the application revision and artifact versions with the backup.
4. Restart services and verify both health aliases.

Test restoration to a separate path before relying on a backup. Reports, model artifacts, and dataset files are not contained in SQLite and require separate retention.

## Logs and observability

Set `CONFTEST_LOG_LEVEL` to an appropriate standard level. Monitor:

- startup/configuration validation failures;
- missing or incompatible artifacts;
- selection mode and fallback reasons;
- test discovery and subprocess failures;
- GitHub signature or changed-file retrieval failures;
- database lock/write errors;
- latency and resource use at the complete request level.

The committed latency report measures scoring, not Git mining, feature extraction, process startup, or target tests. Do not use its 3.345 ms mean as an end-to-end service SLO.

## Docker Compose

Use the repository's Compose definition only after providing production secrets and explicit CORS origins through the deployment environment. Validate resolved configuration before startup:

```bash
docker compose config
docker compose up --build
```

Confirm API port 8000, dashboard port 8501, artifact mounts, database persistence, and health checks. See [Docker deployment](deployment_docker.md). If Docker is unavailable, use the local Python commands and report that container validation was not performed.

## Upgrade and rollback

1. Back up the database and retain current artifacts.
2. Install the new revision in a fresh environment or rebuild images.
3. Review changes to `Settings`, database models, feature order, schemas, and policy format.
4. Run tests, provenance checks, schema initialization, non-executing selection, and health probes.
5. Deploy gradually; keep prior code and artifacts available.

Rollback code and its matching artifact set together. A database schema change may require a tested restoration or migration path.

## Operational limitations

The shipped operating point observed zero escapes by abstaining on 97.81% of held-out commits and measured no wall-clock reduction. Cross-repository transfer is weak, 13 training features are constant, reports predate row bagging, and Python static analysis misses dynamic behavior. Treat [limitations](limitations.md) as required operational reading.
