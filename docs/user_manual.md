# ConfTest User Manual

This guide covers local installation, non-executing selection, optional execution and persistence, and the API/dashboard entry points. ConfTest is a research-grade selector: a `SAFE_FULL_SUITE` decision is expected behavior, not an error, and measured recall is not a formal guarantee.

## Prerequisites

- Python 3.11 or newer.
- Git and a local Python repository whose tests can be collected by pytest.
- ConfTest's ensemble, calibrator, and tuned policy artifacts.
- Permission to execute the target repository's tests if using `--execute`.

The required default artifact paths are:

- `models/ensembles/5_seed_lgbm/`
- `models/calibrator.joblib`
- `models/policy_config.json`

If an artifact is absent, do not substitute an arbitrary file or threshold. Follow [reproducibility](reproducibility.md) to generate it or pass an explicitly validated path.

## Install on POSIX

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
```

## Install on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m conftest.db.init_db
```

Run commands from the repository root. Editable installation exposes the maintained package under `src/conftest/`.

## First non-executing selection

```bash
python scripts/select_tests.py \
  --repo-path tests/sample_suite \
  --commit-sha HEAD \
  --budget 0.25 \
  --output-json reports/local_selection.json
```

PowerShell accepts the same command on one line. The CLI resolves the target commit, discovers tests, extracts the canonical 32 features, scores candidates, and returns either:

- `FAST_SELECTED`: a risk-ranked subset limited by the requested budget.
- `SAFE_FULL_SUITE`: every discovered test, because a confidence, uncertainty, OOD, or artifact safety gate required fallback.

The JSON and log summary include the commit SHA, decision mode, abstention flag, counts, selected test IDs, ranking information, confidence, uncertainty, reasons, and any execution result.

## Selection CLI reference

| Option | Default | Meaning |
|---|---|---|
| `--repo-path PATH` | `.` | Target repository root. |
| `--commit-sha REF` | `HEAD` | Commit SHA or Git ref to analyze. |
| `--budget FLOAT` | `0.25` | Fraction of tests allowed in fast mode. |
| `--ensemble PATH` | `./models/ensembles/5_seed_lgbm` | Ensemble directory. |
| `--calibrator PATH` | `./models/calibrator.joblib` | Serialized calibrator. |
| `--policy-config PATH` | `./models/policy_config.json` | Tuned policy JSON. |
| `--execute` | off | Run the selected or fallback tests in a subprocess. |
| `--persist-db` | off | Store repository, predictions, and decision in SQLite. |
| `--output-json PATH` | empty | Write the complete outcome as JSON. |

Check the installed version directly with `python scripts/select_tests.py --help`. Do not use the obsolete `--output-report` flag.

## Execute tests safely

Review the target repository and selection output before enabling execution:

```bash
python scripts/select_tests.py --repo-path ./target-repo --commit-sha HEAD --execute
```

ConfTest runs target tests as local code. Use an isolated environment for untrusted repositories. In fallback mode, `--execute` runs the full discovered suite. Without `--execute`, selection is advisory and no tests are launched.

## Persist a decision

Initialize the schema, then add `--persist-db`:

```bash
python -m conftest.db.init_db
python scripts/select_tests.py --repo-path ./target-repo --persist-db --output-json reports/selection.json
```

The default database is `data/conftest.db`. Avoid multiple writers during maintenance; see [operations](operations.md#database-and-backups).

## Start the API

```bash
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000
```

Useful URLs:

- Health: `http://127.0.0.1:8000/health` and `/api/v1/health`
- OpenAPI UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Schema: `http://127.0.0.1:8000/openapi.json`

The API initializes the database schema at startup. Payloads and errors are documented in the [API reference](api.md).

## Start the dashboard

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Open `http://127.0.0.1:8501`. Dashboard headline values come from report artifacts. A missing artifact should produce an explicit unavailable/refusal state, not an invented metric.

## Interpret results

The shipped policy observed 100.00% failure recall and zero escaped commits on 183 held-out commits, while abstaining on 97.81% and producing 3.11% test-execution reduction but 0.0% measured wall-clock reduction. These values describe one committed experiment; they do not prove the same behavior on another repository or after retraining. Review [limitations](limitations.md) before operational use.

## Next steps

- Configuration and production safeguards: [operations](operations.md)
- Common failures: [troubleshooting](troubleshooting.md)
- GitHub automation: [GitHub integration](github_integration.md)
- Experimental regeneration: [reproducibility](reproducibility.md)
