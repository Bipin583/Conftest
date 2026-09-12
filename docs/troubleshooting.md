# ConfTest Troubleshooting

Start with the exact failing command, traceback, working directory, Python version, and relevant paths. Run commands from the repository root with the intended virtual environment active.

## Quick diagnostic sequence

```bash
python --version
python -c "import conftest; print(conftest.__file__)"
python scripts/select_tests.py --help
python scripts/evaluate.py
python scripts/check_no_fabricated_labels.py
```

Then run the narrowest relevant test. Do not delete databases, artifacts, or environments before preserving the error and confirming what the target contains.

## Import or command not found

Symptoms include `ModuleNotFoundError: conftest`, missing console dependencies, or imports resolving outside this checkout.

1. Confirm Python 3.11+ and the active interpreter.
2. Install the project from the repository root:
   ```bash
   python -m pip install -e ".[dev]"
   ```
3. Print `conftest.__file__` and confirm it points to this checkout.
4. Prefer `python -m pytest`, `python -m uvicorn`, and `python -m streamlit` so commands use the active interpreter.

On Windows PowerShell activate with `.\.venv\Scripts\Activate.ps1`; on POSIX use `source .venv/bin/activate`.

## Missing ensemble, calibrator, or policy

Typical failures name `models/ensembles/5_seed_lgbm`, `models/calibrator.joblib`, or `models/policy_config.json`.

- Confirm the configured path exists relative to the process working directory.
- Check `CONFTEST_ENSEMBLE_PATH`, `CONFTEST_CALIBRATOR_PATH`, and `CONFTEST_POLICY_CONFIG_PATH`.
- Keep artifacts from the same training run together.
- Generate missing artifacts through the documented [reproducibility workflow](reproducibility.md).
- Do not copy arbitrary thresholds into `.env` or use untuned constructor defaults as the shipped policy.

## LightGBM installation failure

Use a supported Python version and update packaging tools in the active environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If a wheel is unavailable, capture the complete build error and check the platform compiler/runtime prerequisites for the pinned dependency. Do not switch to an unreviewed similarly named package.

## Empty test discovery

If `total_count` is zero or expected tests are absent:

1. Run pytest collection in the target repository:
   ```bash
   python -m pytest --collect-only -q /path/to/repository
   ```
2. Confirm `--repo-path` is the repository root and tests follow pytest discovery rules.
3. Install the target repository's own test dependencies in the execution environment.
4. Inspect target pytest configuration, custom plugins, import errors, and collection exclusions.
5. Verify generated or dynamically registered tests can be collected non-interactively.

ConfTest cannot rank tests that pytest does not expose.

## Git diff or commit resolution problems

Confirm the target is a Git checkout and the requested ref exists:

```bash
git -C /path/to/repository rev-parse HEAD
git -C /path/to/repository show --stat --oneline HEAD
```

Use a full or unambiguous ref with `--commit-sha`. A shallow clone may not contain the parent required for a diff. Fetch the required history in authorized CI environments rather than accepting a fabricated fallback diff as evidence.

## Selection always falls back

`SAFE_FULL_SUITE` is a valid safety outcome. Inspect `reasons`, uncertainty, confidence, OOD limits, artifact compatibility, and changed-file size. Do not raise thresholds solely to make the dashboard show more reduction. Tune on validation data using `scripts/tune_policy.py`, then evaluate the frozen point on held-out data.

The shipped policy itself abstains on 97.81% of held-out commits, so frequent fallback is consistent with the committed experiment.

## Test execution fails

Selection can succeed while `--execute` fails because target tests are independent code.

- Re-run the exact selected node IDs with pytest in the same environment.
- Check imports, environment variables, services, filesystem permissions, working directory, and timeouts.
- Treat untrusted repositories as untrusted code and execute them only in isolation.
- In fallback mode, expect the full discovered suite to run.

Omit `--execute` to debug selection without launching tests.

## Database initialization or lock errors

Initialize explicitly:

```bash
python -m conftest.db.init_db
```

Confirm the parent directory is writable and `CONFTEST_DATABASE_URL` uses the intended path. For SQLite lock errors, stop duplicate API workers and selector processes using `--persist-db`, allow active transactions to finish, and retry. Do not delete `data/conftest.db`, `-wal`, or `-shm` files without a verified backup and explicit intent.

## API does not start

- Read the startup validation error before changing settings.
- In production, replace the public development webhook secret and wildcard credentialed CORS.
- Confirm all artifact paths and database permissions.
- Start locally with:
  ```bash
  python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000
  ```
- Probe `/health`, `/api/v1/health`, and `/openapi.json`.

If the port is occupied, identify the existing listener before choosing a new port or terminating a process.

## Dashboard metrics unavailable

The dashboard reads committed report artifacts. Run `python scripts/evaluate.py` to see which files are absent and which producer owns each one. Missing metrics should remain unavailable; do not create placeholder JSON or copy numbers from prose. Start the dashboard from the project root so relative report paths resolve.

## GitHub webhook returns 401

A 401 commonly means `X-Hub-Signature-256` does not match the configured secret.

- Confirm GitHub and ConfTest use the same non-placeholder secret.
- Preserve the raw request body for HMAC verification; JSON reserialization changes the signature input.
- Verify proxies forward the signature header unchanged.
- Rotate exposed secrets and update both ends together.

Do not log the secret or full authorization token.

## GitHub webhook returns 202 with full-suite advice

For supported PR events, ConfTest fetches changed files from GitHub. If that fetch fails or yields no trustworthy diff, the API returns a safe `SAFE_FULL_SUITE` recommendation with HTTP 202. Check repository identity, token permissions, rate limits, network access, and PR metadata. This is deliberate refusal behavior, not a diff to bypass.

## Docker service unhealthy

Run:

```bash
docker compose config
docker compose ps
docker compose logs conftest-api
docker compose logs conftest-dashboard
```

Check resolved `CONFTEST_DATABASE_URL`, secret/CORS configuration, volume permissions, artifact mounts, port conflicts, and health probe URLs. On systems without Docker, use the local startup commands; do not claim Compose was validated.

## Report or documentation validation fails

- Missing producer: verify every referenced `scripts/*.py` path exists.
- Unsupported flag: run the producer with `--help` and update the command.
- Provenance refusal: restore real labels or mark data synthetic; do not weaken the guard.
- Configuration drift: compare `.env.example` with `conftest.config.Settings`.
- Feature mismatch: compare against `FeatureExtractionPipeline.FEATURE_NAMES` exactly.
- Broken link: resolve it relative to the Markdown file containing it.

Run focused checks before the full suite:

```bash
python -m pytest tests/unit/test_documented_import_paths.py -q
python -m pytest tests/unit/test_producer_references.py -q
python -m pytest tests/unit/test_config.py -q
python -m pytest tests/ -q
```

## Getting a useful failure report

Include the command, exit code, traceback, operating system, Python version, current revision, whether execution/persistence was enabled, and non-secret configuration names. Redact tokens, webhook secrets, private repository contents, and credentials.
