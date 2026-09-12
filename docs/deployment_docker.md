# Docker Deployment

This guide covers the checked-in two-service Compose stack: the FastAPI API and the Streamlit dashboard. Runtime settings are defined by [`src/conftest/config.py`](../src/conftest/config.py); [`.env.example`](../.env.example) is the environment template. Tuned selection thresholds are not environment settings: they live only in [`models/policy_config.json`](../models/policy_config.json).

## Prerequisites

- Docker Engine with the Compose v2 plugin (`docker compose`).
- The model, calibrator, and policy artifacts under `models/`.
- A strong, deployment-specific GitHub webhook secret.

Compose runs the API with `CONFTEST_ENV=production`. Production startup intentionally refuses the public development secret and wildcard CORS. Set the required secret before starting:

```bash
export CONFTEST_GITHUB_WEBHOOK_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export CONFTEST_CORS_ALLOW_ORIGINS='["https://dashboard.example.com"]'
docker compose up -d --build
```

In PowerShell:

```powershell
$env:CONFTEST_GITHUB_WEBHOOK_SECRET = python -c "import secrets; print(secrets.token_hex(32))"
$env:CONFTEST_CORS_ALLOW_ORIGINS = '["https://dashboard.example.com"]'
docker compose up -d --build
```

For a local-only deployment, the Compose default CORS list permits `http://localhost:8501` and `http://127.0.0.1:8501`; setting `CONFTEST_CORS_ALLOW_ORIGINS` is then optional. The webhook secret remains mandatory because the API still runs in production mode.

Do not commit the generated secret. Configure the same value in the GitHub webhook settings described in [GitHub integration](github_integration.md).

## Services and endpoints

| Service | Host endpoint | Container dependency |
|---|---|---|
| `conftest-api` | `http://localhost:8000` | SQLite volume, read-only `models/` and `reports/` |
| `conftest-dashboard` | `http://localhost:8501` | SQLite volume and `reports/` |

Useful API endpoints:

- Health: `http://localhost:8000/health`
- Versioned health: `http://localhost:8000/api/v1/health`
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI: `http://localhost:8000/openapi.json`

Check the stack:

```bash
docker compose ps
curl --fail http://localhost:8000/health
docker compose logs --tail=100 conftest-api
docker compose logs --tail=100 conftest-dashboard
```

## Configuration used by Compose

| Variable | Compose value or source | Purpose |
|---|---|---|
| `CONFTEST_ENV` | `production` | Enables production validation. |
| `CONFTEST_DATABASE_URL` | `sqlite:////app/data/conftest.db` | SQLite file in the shared volume. |
| `CONFTEST_API_HOST` | `0.0.0.0` | Makes the API reachable through the published port. |
| `CONFTEST_API_PORT` | `8000` | API listen port. |
| `CONFTEST_LOG_LEVEL` | `INFO` | Application log level. |
| `CONFTEST_CORS_ALLOW_ORIGINS` | host environment or local dashboard origins | JSON list of permitted browser origins. |
| `CONFTEST_GITHUB_WEBHOOK_SECRET` | required host environment variable | HMAC SHA-256 webhook secret. |

Use `CONFTEST_DATABASE_URL`, not the obsolete `CONFTEST_DB_URL`. See [`.env.example`](../.env.example) for all supported `CONFTEST_*` keys.

## Persistence and artifacts

The named volume `conftest-data` stores `/app/data`, including `conftest.db`. `docker compose down` retains it; `docker compose down -v` deletes it and therefore requires an intentional backup decision.

Back up the volume without modifying it:

```bash
docker run --rm -v conftest-shared-data:/source:ro -v "${PWD}:/backup" alpine \
  tar -czf /backup/conftest-data.tar.gz -C /source .
```

The API receives `./models` and `./reports` as read-only mounts. Replace model artifacts on the host and restart the API to load a new version:

```bash
docker compose restart conftest-api
```

The dashboard's reports mount is writable in the checked-in stack, but dashboard pages only read evidence artifacts. Missing artifacts are reported with the producer command; they are not replaced by synthetic values. See [Dashboard](dashboard.md).

## Lifecycle and troubleshooting

```bash
# Rebuild after source or dependency changes
docker compose up -d --build

# Stop while retaining data
docker compose down

# Follow service logs
docker compose logs -f conftest-api
```

Common startup failures:

- **Webhook-secret interpolation error:** set `CONFTEST_GITHUB_WEBHOOK_SECRET` in the shell launching Compose.
- **Production settings validation error:** replace wildcard CORS with an explicit JSON list and do not use the development secret.
- **Unhealthy API:** inspect API logs, then verify the model mounts and SQLite volume are accessible.
- **Dashboard has no measurements:** run the producer named in the page error; consult [`scripts/evaluate.py`](../scripts/evaluate.py) for the report-to-producer map.
- **Port already allocated:** change the host side of the relevant `ports` mapping while retaining container ports 8000 and 8501.

## Deployment scope

The checked-in stack is a reproducible project deployment, not a high-availability reference architecture. It uses one local SQLite database, no TLS termination, no external secret manager, and no horizontal API coordination. Put it behind a TLS reverse proxy and use platform secret storage for any exposed deployment. Review [Limitations](limitations.md) before making reliability or safety claims.
