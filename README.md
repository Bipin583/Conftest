# ConfTest

Confidence-calibrated, uncertainty-aware regression test selection for Python CI pipelines.

ConfTest ranks tests by calibrated failure risk when its ensemble is sufficiently certain. Otherwise, it abstains from subset selection and recommends or executes the full suite. The maintained implementation is under `src/conftest/`; the project also includes a FastAPI service, Streamlit dashboard, SQLite persistence, GitHub integration, and an evidence-producing evaluation pipeline.

## Measured status

The shipped operating point in `models/policy_config.json` has been evaluated on 183 held-out commits (86,469 commit-test rows):

| Metric | Observed result | Evidence |
|---|---:|---|
| Failure recall | 100.00% | `reports/baseline_comparison.csv` |
| Escaped commits | 0 of 183 | `reports/baseline_comparison.csv` |
| Test-execution reduction | 3.11% | `reports/baseline_comparison.csv` |
| Wall-clock reduction | 0.0% | `reports/economic_analysis.json` |
| Abstention rate | 97.81% | `reports/baseline_comparison.csv` |
| Validation ECE | 0.0390 -> 0.0242 | `reports/calibration_report.json` |
| Mean scoring latency, 50 tests | 3.345 ms | `reports/latency_benchmark.json` |

These are **observations on the committed experiment**, not a bug-free guarantee. The zero-escape result is achieved while the policy abstains on nearly every held-out commit, so the measured time saving is zero. A validation-selected higher-reduction point reached only 87.69% recall on held-out data and failed the project's 95% recall gate. Published artifacts also predate the row-bagging fix in the trainer. Read [limitations](docs/limitations.md) before quoting results.

## Quick start

Requirements: Python 3.11+, Git, and a checkout containing the committed model artifacts.

### POSIX shell

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
python scripts/select_tests.py --repo-path tests/sample_suite --commit-sha HEAD --budget 0.25
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
python scripts/select_tests.py --repo-path tests/sample_suite --commit-sha HEAD --budget 0.25
```

Add `--execute` to run the selected or fallback suite, `--persist-db` to record the decision, and `--output-json reports/local_selection.json` to save the response. See the [user manual](docs/user_manual.md) for artifact requirements and expected behavior.

## GitHub Actions

The pull-request workflow runs ConfTest and reports its selection result. An optional read-only follow-up can analyze failed CI logs with Anthropic and post one advisory pull-request comment. To enable it, add `ANTHROPIC_API_KEY` as a GitHub Actions repository secret and merge `.github/workflows/ai-failure-analysis.yml` into the default branch. The analyzer checks out no pull-request code, sends only bounded and redacted log evidence, and cannot execute suggestions or modify source. Every proposed fix requires human review. See [GitHub integration](docs/github_integration.md#read-only-ai-failure-analysis) for permissions, external-data handling, fork security, and activation details.

## Services

```bash
# API: http://127.0.0.1:8000/docs
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000

# Dashboard: http://127.0.0.1:8501
python -m streamlit run dashboard/app.py --server.port 8501
```

The API initializes the SQLite schema at startup. Production mode rejects the public development webhook secret and wildcard credentialed CORS. Follow the [operations guide](docs/operations.md). The complete template is `.env.example`; this synchronized block lists the runtime keys accepted by `conftest.config.Settings`:

```ini
CONFTEST_ENV=development
CONFTEST_DEBUG=true
CONFTEST_LOG_LEVEL=INFO
CONFTEST_GIT_SHA=
CONFTEST_API_HOST=127.0.0.1
CONFTEST_API_PORT=8000
CONFTEST_API_WORKERS=1
CONFTEST_DATABASE_URL=sqlite:///./data/conftest.db
CONFTEST_DB_ECHO=false
CONFTEST_ENSEMBLE_PATH=./models/ensembles/5_seed_lgbm
CONFTEST_CALIBRATOR_PATH=./models/calibrator.joblib
CONFTEST_POLICY_CONFIG_PATH=./models/policy_config.json
CONFTEST_DEFAULT_REPO_ROOT=./tests/sample_suite
CONFTEST_DEFAULT_BUDGET_RATIO=0.25
CONFTEST_CORS_ALLOW_ORIGINS=["http://localhost:8501","http://127.0.0.1:8501"]
CONFTEST_GITHUB_WEBHOOK_SECRET=development_secret_only_change_in_ci
CONFTEST_GITHUB_TOKEN=
CONFTEST_DATA_DIR=./data
CONFTEST_MODELS_DIR=./models
```

## How the system works

```text
Git commit or PR diff
  -> test discovery + diff/AST/dependency/history features
  -> 5-member LightGBM ensemble
  -> epistemic disagreement + post-hoc calibration
  -> tuned selective policy
       -> FAST_SELECTED: budget-matched ranked subset
       -> SAFE_FULL_SUITE: abstain and run/recommend every discovered test
  -> optional execution, SQLite persistence, API/dashboard/GitHub reporting
```

The canonical 32-column order is `src/conftest/features/pipeline.py::FEATURE_NAMES`; the readable reference is [docs/feature_schema.md](docs/feature_schema.md). Thresholds come only from `models/policy_config.json`, produced by `scripts/tune_policy.py`.

## Documentation

- [Documentation hub](DOCUMENTATION.md) and [complete index](docs/README.md)
- [User manual](docs/user_manual.md): install, select, execute, API, dashboard
- [Architecture](docs/architecture.md), [feature schema](docs/feature_schema.md), [API](docs/api.md), [database](docs/database.md)
- [Developer guide](docs/development.md): package layout, extension points, tests
- [Operations guide](docs/operations.md): configuration, security, deployment, backup
- [Reproducibility guide](docs/reproducibility.md): data-to-report evidence chain
- [Troubleshooting](docs/troubleshooting.md) and [limitations](docs/limitations.md)
- [Academic/research map](docs/README.md#academic-and-research-review)

## Reproduce and inspect results

```bash
python scripts/evaluate.py
python scripts/check_no_fabricated_labels.py
```

The first command lists each evaluation stage, its producer, and whether its artifacts are present. Run one stage with `python scripts/evaluate.py --run <stage>`; use `--all` only after reading the [reproducibility guide](docs/reproducibility.md), because cross-repository and continuous-learning stages retrain repeatedly.

## Development validation

```bash
python -m pytest tests/ -q
python -m ruff check src tests dashboard
```

The `Makefile` provides POSIX shortcuts, but the Python module commands documented above are portable to Windows. Test totals are deliberately not hard-coded here; run the suite for the current count.

## Scope and maturity

ConfTest is a research-grade Python RTS prototype, not a universal or formally verified safety system. Static analysis can miss dynamic behavior; new repositories are out of distribution; 13 training features are constant in the committed harvest; scoring latency excludes feature mining; and dollar figures are projections. Use full-suite fallback for decisions whose risk is not supported by local validation.

## License

MIT licensed.
