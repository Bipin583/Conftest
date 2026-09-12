# FastAPI Reference

The application is defined in `src/conftest/api/main.py`; route decorators and `src/conftest/api/schemas.py` are the executable contract. The default local base URL is `http://127.0.0.1:8000`. OpenAPI is available at `/openapi.json`, Swagger UI at `/docs`, and ReDoc at `/redoc`.

Start locally:

```bash
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000
```

## Endpoint table

| Method | Path | Success | Purpose |
|---|---|---:|---|
| GET | / | 200 | Service metadata and endpoint links. |
| `GET` | `/health` | 200 | Liveness and database probe. |
| `GET` | `/api/v1/health` | 200 | Alias of `/health`. |
| `POST` | `/api/v1/select` | 200 | Rank/select tests or abstain to the full suite. |
| `POST` | `/api/v1/explain` | 200 | Explain one feature vector with one serving-ensemble member. |
| `GET` | `/api/v1/calibration` | 200 | Serve committed calibration diagnostics. |
| `GET` | `/api/v1/repositories` | 200 | List repositories (`skip=0`, `limit=50`). |
| `POST` | `/api/v1/repositories` | 201 | Register a repository or return the existing record. |
| `GET` | `/api/v1/repositories/{repo_id}` | 200 | Repository metadata and bounded counts. |
| `GET` | `/api/v1/repositories/{repo_id}/commits` | 200 | Recent commits/decisions (`skip=0`, `limit=50`). |
| `GET` | `/api/v1/analytics` | 200 | Aggregate persisted telemetry. |
| `POST` | `/api/v1/github/webhook` | 200/202 | Process signed GitHub events or recommend full-suite fallback. |

Pydantic returns 422 for structurally invalid request/query data.

## Selection

`POST /api/v1/select` accepts:

```json
{
  "repository_name": "owner/repo",
  "commit_sha": "HEAD",
  "changed_files": [
    {"file_path": "src/auth.py", "change_type": "M", "lines_added": 12, "lines_deleted": 2}
  ],
  "commit_message": "fix token validation",
  "budget_ratio": 0.25,
  "repo_path": null,
  "execute": false
}
```

Defaults are `local/sample-app`, `HEAD`, an empty changed-file list, `Update application logic`, `0.25`, `null`, and `false`; however, an empty `changed_files` list is deliberately rejected with 422 because no real diff exists to score. `budget_ratio` must be within `[0.01, 1.0]`; line counts must be non-negative.

The response includes `commit_sha`, `decision_mode`, `abstained`, selected/total counts, `test_reduction_pct`, top confidence, aggregate epistemic uncertainty, reasons, selected node IDs, ranked per-test records, a Markdown summary, and optional execution output. `test_reduction_pct` is a count reduction, not measured wall-clock saving. Missing model/calibrator/policy artifacts return 503; unexpected pipeline failures return 500 with details kept in server logs.

## Explanation

`POST /api/v1/explain` accepts `test_id`, a numeric feature dictionary, and `top_k` (default 5, range 1–32). Missing canonical keys are converted to `0.0` in canonical order; callers should therefore use the complete [feature schema](feature_schema.md) rather than depend on silent fill.

The response contains predicted probability, base expected value, a rule-card risk level/reasons, and top positive/negative SHAP drivers. Attributions are computed against the first member named by the configured ensemble metadata, not an aggregate ensemble explanation. Missing or invalid ensemble metadata/member files return 503; initialization errors return 500.

## Calibration

`GET /api/v1/calibration` reads `reports/calibration_report.json`, produced by `python scripts/calibrate_model.py`. It returns the uncalibrated metrics, the selected calibrated method when one was selected, paired-difference intervals when recorded, reliability bins, fitted temperature when applicable, and explicit selection/measurement split labels.

There are no fabricated fallback metrics: a missing report returns 503 and an unreadable report returns 500. `calibrated` is null when the selection outcome is `uncalibrated`.

## Repositories and analytics

Repository creation requires `full_name` and `url`; `local_path` is optional in the request schema. Unknown repository IDs return 404. Detail counts are based on API queries capped at 1,000 records and should not be treated as unbounded database counts.

Analytics reports repository, commit, decision, fast-mode, abstention, and failure totals plus ten recent decisions. Averages are null when no decisions exist. `total_missed_failures` is null when no outcome has a verified full-suite comparison; `verified_outcomes` and `unverified_outcomes` state evidence coverage. Recent `test_reduction_pct` values are count ratios.

## GitHub webhook

The webhook validates `X-Hub-Signature-256` when a secret is configured. Invalid signatures return 401 and invalid JSON returns 400. `ping` returns `pong`; pull-request actions other than `opened`, `synchronize`, and `reopened` are ignored.

For actionable pull requests, the handler fetches changed files from GitHub. If the token/API/diff is unavailable, it returns 202 with `decision_mode: SAFE_FULL_SUITE` and does not score an invented diff. Configure `CONFTEST_GITHUB_TOKEN` for API access and a non-public `CONFTEST_GITHUB_WEBHOOK_SECRET` outside development.

## Health semantics

Both health paths return status, service, version, environment, database state, and process uptime. A failed database probe still returns HTTP 200 with `status: degraded` and `database: unreachable`; monitoring must inspect the body rather than relying only on the status code.
