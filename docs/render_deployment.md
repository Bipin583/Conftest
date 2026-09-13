# Deploying ConfTest to Render

ConfTest ships as a **single Docker service**: nginx on the platform port,
routing to the FastAPI API and Streamlit dashboard, both bound to loopback
inside the container.

```
                    ┌─────────────── one Render web service ───────────────┐
 browser / GitHub ──▶ nginx :10000 (PORT)                                 │
                    │   ├─ /health, /api/*, /docs, /redoc ──▶ uvicorn :8000 │
                    │   └─ everything else (dashboard UI) ──▶ streamlit :8501│
                    │   /app/data ◀── persistent disk (SQLite)              │
                    └───────────────────────────────────────────────────────┘
```

Relevant files:

| File | Role |
|---|---|
| `Dockerfile` | Multi-stage build; copies `models/`, `reports/`, `tests/sample_suite/` into the image |
| `deploy/entrypoint.sh` | Starts uvicorn + Streamlit + nginx; exits the container if any of them dies |
| `deploy/nginx.conf.template` | Port routing; `${PORT}` substituted at start |
| `render.yaml` | Render blueprint: service, health check, disk, env vars |

## Deploy steps

1. Push this branch and merge to `main` (Render builds the connected
   branch's Dockerfile).
2. In Render: **New → Blueprint**, point it at this repository. The
   `render.yaml` blueprint creates one web service named `conftest`.
   - The blueprint sets `plan: starter` because persistent disks require a
     paid plan. On the free plan, delete the `disk` block — the service runs,
     but the SQLite database is wiped on every deploy/restart.
3. Render generates `CONFTEST_GITHUB_WEBHOOK_SECRET` automatically
   (production refuses the public placeholder from `.env.example` — this is
   enforced at startup by `config.py`). Copy the generated value from
   **Environment** in the Render dashboard.
4. Wait for the build. First deploy takes longer (pip installs the ML stack);
   later builds reuse Render's layer cache.
5. Verify:
   - `https://<service>.onrender.com/` → Streamlit dashboard
   - `https://<service>.onrender.com/health` → `{"status": "ok", ...}`
   - `https://<service>.onrender.com/docs` → FastAPI Swagger UI
6. For the GitHub webhook integration, configure the PR webhook in your
   repository's settings to point at
   `https://<service>.onrender.com/api/v1/github/webhook` with the secret
   from step 3. Optionally set `CONFTEST_GITHUB_TOKEN` in the dashboard to
   lift the 60 req/hr unauthenticated rate limit.

## What lives where

- **SQLite database** — `/app/data/conftest.db` on the persistent disk.
  Survives deploys and restarts on a paid plan. Both the API and the
  dashboard read it in-process (the dashboard does not call the API over
  HTTP), which is why the stack is one service rather than two.
- **Model & report artifacts** — baked into the image (`models/`,
  `reports/`, `tests/sample_suite/`, ~1.6 MB total). A new model or report
  means a new deploy; that is deliberate: the served ensemble, calibrator
  and tuned policy change together, atomically, with the code that expects
  them.
- **Secrets** — set in the Render dashboard, never in the image.

## Local verification of the same image

```bash
docker build -t conftest-render .
docker run --rm -p 10000:10000 \
  -e CONFTEST_ENV=production \
  -e CONFTEST_GITHUB_WEBHOOK_SECRET=local-test-secret \
  conftest-render
# then open http://localhost:10000/ and http://localhost:10000/health
```

`docker-compose.yml` remains the two-service local development setup
(API and dashboard separately, with `models/` and `reports/` bind-mounted
from the host); it overrides the image's default command, so both setups
share the same Dockerfile.
