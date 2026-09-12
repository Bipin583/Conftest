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

## Read-only AI failure analysis

[`.github/workflows/ai-failure-analysis.yml`](../.github/workflows/ai-failure-analysis.yml) is an optional privileged follow-up workflow. After `ConfTest Regression Test Selection CI` fails for a pull request, it downloads that run's GitHub-generated log archive, keeps bounded failure context, redacts credential-like values, requests an advisory diagnosis from Anthropic, and creates or updates one marked pull-request comment.

Before enabling it:

1. Create an Anthropic API key and store it in the repository's **Settings > Secrets and variables > Actions** as `ANTHROPIC_API_KEY`. Never place the key in source, workflow YAML, an issue, or a pull-request comment.
2. Merge the follow-up workflow and `scripts/analyze_ci_failure.py` into the repository's default branch. GitHub only triggers a `workflow_run` workflow when its definition exists on the default branch.
3. Keep the configured model as `claude-sonnet-5`, or review and test any deliberate model change in the trusted workflow definition.

The workflow has `actions: read`, `contents: read`, and `pull-requests: write`. It checks out only the default branch with persisted credentials disabled; it never checks out or executes the failed pull-request head. The write permission is used only to create or update the advisory comment. The original CI conclusion remains failed.

### Data and threat model

Pull-request code and all failed-run logs are untrusted. The analyzer reads ZIP members in memory without extraction, rejects unsafe member paths and non-regular entries, enforces compressed, per-file, file-count, uncompressed, and prompt-size bounds, and keeps only nearby failure diagnostics. It removes the configured GitHub and Anthropic secrets plus common token, authorization-header, assignment, URL-credential, and secret-query forms before calling `https://api.anthropic.com/v1/messages`.

The bounded redacted evidence, repository name, pull-request number, failed-run URL, and head SHA are sent to Anthropic. Logs can still contain project data that generic redaction does not recognize; do not enable the feature for repositories whose CI output must not leave GitHub. Fork logs are treated with the same hostile-data rules, and no pull-request cache or artifact is executed or trusted.

Model output is text only. It is never executed, applied, committed, or pushed, and it cannot change tests or the workflow result. Every suggestion requires developer inspection and approval. If downloading logs, parsing the archive, or calling Anthropic fails, the analyzer step fails rather than inventing a diagnosis.

Successful runs, cancelled runs, manual/non-pull-request runs, and failed runs without an associated pull request do not invoke Anthropic. Rerunning analysis updates the existing `<!-- conftest-ai-failure-analysis -->` comment instead of adding duplicates.

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
