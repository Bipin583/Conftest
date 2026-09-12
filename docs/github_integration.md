# GitHub Integration

ConfTest has two separate GitHub integration paths. Choose one based on where selection should run:

1. The checked-in **GitHub Actions workflow** runs inside a pull-request runner.
2. The **FastAPI webhook endpoint** receives signed events at `/api/v1/github/webhook`.

They share the selector but have different credentials, outputs, and failure modes.

## GitHub Actions workflow

[`.github/workflows/conftest.yml`](../.github/workflows/conftest.yml) runs for opened, synchronized, and reopened pull requests, and can also be started manually. Its steps are:

1. Check out full Git history and install Python 3.11 plus `.[dev]`.
2. Run `scripts/check_no_fabricated_labels.py`.
3. Run the complete `tests/` suite with coverage.
4. Run `scripts/select_tests.py` and write `reports/conftest_selection_report.json` with `--output-json`.
5. Run the selector again with `--execute` to execute the chosen tests.
6. On pull requests, post a summary using the job-scoped `GITHUB_TOKEN`.

The selector's maintained options are:

```text
--repo-path --commit-sha --budget --ensemble --calibrator
--policy-config --execute --persist-db --output-json
```

A minimal invocation is:

```bash
python scripts/select_tests.py \
  --repo-path . \
  --commit-sha HEAD \
  --budget 0.25 \
  --output-json reports/conftest_selection_report.json
```

The workflow grants `contents: read` and `pull-requests: write`. The write permission is used only for its report comment. Fork pull requests and repository policy can restrict comment permissions; selection and tests still run, but the comment step may not be able to publish.

`test_reduction_pct` is a reduction in selected test count. It must not be described as measured wall-clock savings unless execution timing was separately measured.

### Required artifacts

The runner must have access to:

```text
models/ensembles/5_seed_lgbm/
models/calibrator.joblib
models/policy_config.json
```

If a deployment stores artifacts outside Git, retrieve them through an authenticated, integrity-checked step before selection. Do not replace absent artifacts with sample scores.

## Webhook mode

Configure a repository webhook with:

- Payload URL: `https://your-domain.example/api/v1/github/webhook`
- Content type: `application/json`
- Secret: the exact value of `CONFTEST_GITHUB_WEBHOOK_SECRET`
- Event: Pull requests

Generate and store a high-entropy secret in the deployment's secret manager. Production startup rejects the public development placeholder. GitHub signs each body in `X-Hub-Signature-256`; ConfTest verifies the HMAC SHA-256 signature before processing it. Invalid signatures receive HTTP 401.

The endpoint acknowledges `ping` events and processes only pull-request actions `opened`, `synchronize`, and `reopened`. Other events or actions are ignored explicitly.

### Fetching changed files

Set `CONFTEST_GITHUB_TOKEN` when the service must fetch pull-request files from private repositories or avoid the unauthenticated GitHub API rate limit. The token should have only the repository read access required for pull-request metadata.

If changed files cannot be fetched, ConfTest does not select from an empty or guessed diff. It returns HTTP 202 with a conservative response:

```json
{
  "status": "abstained",
  "decision_mode": "SAFE_FULL_SUITE",
  "abstained": true,
  "reason": "Changed files for this pull request could not be retrieved, so no selection was made. Run the full suite. Set CONFTEST_GITHUB_TOKEN to enable diff-based selection."
}
```

The CI caller must treat this response as an instruction to run the full suite.

### CORS and transport

GitHub's webhook delivery is server-to-server and does not depend on browser CORS. `CONFTEST_CORS_ALLOW_ORIGINS` controls browser clients such as a separately hosted dashboard. Production requires an explicit JSON list and rejects `*`. Terminate TLS in front of the API; never expose an unsigned HTTP webhook over the public Internet.

See [Docker deployment](deployment_docker.md) for Compose secret and CORS setup.

## Report interpretation

The Actions comment reports the execution mode, selected and total test counts, test-count reduction, confidence, and epistemic uncertainty from the JSON selection output. `SAFE_FULL_SUITE` is an intentional abstention outcome.

Example values in screenshots or templates are illustrative unless they identify a committed artifact. The measured held-out operating point currently abstains on most commits; consult [Limitations](limitations.md) and the reports referenced there rather than treating a sample fast-selection card as typical.

## Troubleshooting

**Webhook returns 401.** Confirm GitHub and the service use the same raw secret, the `sha256=` signature header is forwarded unchanged, and no proxy modifies the request body.

**Webhook returns 202 and `SAFE_FULL_SUITE`.** Configure `CONFTEST_GITHUB_TOKEN`, verify repository access, and inspect API logs for GitHub response details. Continue running the full suite until diff retrieval succeeds.

**Workflow rejects an option.** Compare the YAML invocation with:

```bash
python scripts/select_tests.py --help
```

The JSON output option is `--output-json`; `--output-report` is not supported.

**PR comment is absent.** Confirm the event is a pull request, the JSON report exists, and repository policy permits `pull-requests: write` for that event.

**Selection falls back to the full suite.** Inspect the returned decision mode and policy rationale. Do not convert abstention into a failed build unless that is an explicit repository policy.

For endpoint schemas and status codes, see [API reference](api.md).
