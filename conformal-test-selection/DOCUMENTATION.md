# Conformal Test Selection — Technical Documentation

Detailed developer and architecture reference for the
**Confidence-Calibrated Test Selection with Conformal Guarantees** project.

This document complements the two others in the repository:

- **`README.md`** — project overview, headline results, quick start.
- **`CONFORMAL_IMPLEMENTATION_AUDIT.md`** — validity audit of the conformal claim.
- **`DOCUMENTATION.md`** (this file) — end-to-end technical reference: architecture,
  data flow, every CLI command, the full configuration schema, the serving contract,
  the HTTP API, the advisory CI integration, and a module-by-module API summary.

---

## Table of contents

1. [What the system does](#1-what-the-system-does)
2. [The three-layer method](#2-the-three-layer-method)
3. [End-to-end data flow](#3-end-to-end-data-flow)
4. [Repository layout](#4-repository-layout)
5. [Installation & environment](#5-installation--environment)
6. [Data and the reused corpus](#6-data-and-the-reused-corpus)
7. [Feature engineering](#7-feature-engineering)
8. [Pipeline stages & CLI reference](#8-pipeline-stages--cli-reference)
9. [Configuration reference](#9-configuration-reference)
10. [Serving contract — `SelectionPipeline.predict`](#10-serving-contract--selectionpipelinepredict)
11. [HTTP API](#11-http-api)
12. [Advisory CI integration & live PR extraction](#12-advisory-ci-integration--live-pr-extraction)
13. [Results & scope of the guarantee](#13-results--scope-of-the-guarantee)
14. [Testing](#14-testing)
15. [Module reference](#15-module-reference)
16. [Limitations & caveats](#16-limitations--caveats)
17. [References](#17-references)

## 1. What the system does

Large regression suites dominate CI/CD time and cost, yet most commits break
nothing. A model that skips tests it believes will pass saves money but is
dangerous on its own: a miscalibrated or overconfident model silently lets real
failures through, and the cost surfaces in production.

This project predicts, for each candidate test on a given change, the
probability that the test will fail, and then **selects which tests to run with a
distribution-free coverage guarantee** on how many real failures are still
caught. On the held-out test split it retains **95.2 % of failing tests while
running only 25.3 % of the suite**, a **74.7 % CI-cost reduction**.

The distinguishing contribution is the third layer — split **conformal
prediction** applied to test selection — which converts "the model is fairly
confident" into "at least 95 % of failing tests are selected, with 90 %
confidence," valid under the standard conformal assumption that calibration and
future commits are *exchangeable*.

## 2. The three-layer method

```
                 change + test features
                          │
   ┌──────────────────────▼───────────────────────┐
   │ Layer 1 — Gradient-boosted classifier          │  models/train.py
   │ XGBoost over 62 features → raw P(fail) score    │  → gbdt_model.pkl
   └──────────────────────┬───────────────────────┘
                          │ raw scores (rank well, not probabilities)
   ┌──────────────────────▼───────────────────────┐
   │ Layer 2 — Probability calibration               │  models/calibrate.py
   │ Platt (sigmoid) scaling, gated on ECE ≤ 8 %     │  → calibrator.pkl
   └──────────────────────┬───────────────────────┘
                          │ calibrated P(fail): 0.30 ≈ 30 % empirical failure
   ┌──────────────────────▼───────────────────────┐
   │ Layer 3 — Split conformal selection             │  models/conformal.py
   │ Order-statistic threshold with a PAC guarantee  │  → conformal_threshold.pkl
   └──────────────────────┬───────────────────────┘
                          │ selection threshold (probability floor)
                    run test ⇔ P(fail) ≥ floor
```

**Layer 1 — Gradient-boosted classifier (`models/train.py`).** XGBoost over 62
transformed features (41 raw columns) describing the change, each test's
history, and the change↔test relationship. Class imbalance (~5 % failing rows) is
handled with `scale_pos_weight` (derived automatically as
`n_negative / n_positive` when set to `auto`). Trees trained with imbalance
weighting rank well but their scores are *not* probabilities.

**Layer 2 — Probability calibration (`models/calibrate.py`).** Platt scaling
(a fitted sigmoid) maps the raw scores onto calibrated probabilities so a
predicted `0.30` corresponds to a ~30 % empirical failure rate. This is gated on
**Expected Calibration Error (ECE ≤ 8 %)** — calibration is a prerequisite for
the conformal layer to mean anything, because layer 3 turns a probability into a
threshold.

**Layer 3 — Split conformal selection (`models/conformal.py`).** Split conformal
prediction converts calibrated probabilities into a selection threshold (a
*probability floor*) with a coverage guarantee. Two flavours are configurable:

- **`marginal`** — coverage holds *in expectation* over calibration draws
  (order statistic at rank ⌈(n+1)(1−α)⌉).
- **`pac`** (default) — coverage holds *with probability ≥ confidence* over the
  calibration draw (Vovk 2012, training-conditional / PAC-style bound). The PAC
  rank is at least the marginal rank, so it is the more conservative rule.

The fitted rule is: **run a test when its calibrated `P(fail) ≥ probability floor`.**
On this corpus the floor is `0.038145`.

## 3. End-to-end data flow

```
data/splits (reused ConfTest corpus, 561,711 rows)   OR   data/raw (data/collect.py)
        │                                                        │
        │  data/features.py  build_features()  (harvest path)    │
        └───────────────────────────┬────────────────────────────┘
                                     ▼
     data/preprocess.py  preprocess()
       • adapt_external_frame() maps the mined corpus onto the schema
       • impute → scale (numeric) / impute → one-hot (categorical)
       • chronological, group-aware 70/15/15 split (never splits a commit)
                                     ▼
        data/processed/{train,val,test}.csv   +   models/preprocessor.pkl
                                     ▼
     models/train.py     → models/gbdt_model.pkl        + reports/training_report.json
     models/calibrate.py → models/calibrator.pkl        + reports/calibration_report.json
     models/conformal.py → models/conformal_threshold.pkl + reports/conformal_report.json
                                     ▼
     models/pipeline.py  SelectionPipeline.load()  (wires all four artefacts)
                                     ▼
     cli.py predict / evaluate      │      api/server.py (Flask)     │   examples/advisory_report.py (CI)
```

**Serving order matters.** The four fitted artefacts (preprocessor → model →
calibrator → selector) only mean something when applied in the *same order, to
the same columns* as during training. `SelectionPipeline` encapsulates that
sequence so both the CLI and the API call it identically:

```
raw records → feature frame → preprocessor → model → calibrator → selector
```

## 4. Repository layout

```
conformal-test-selection/
├── cli.py                  # 8 subcommands: collect · features · preprocess · train
│                           #   · calibrate · conformal · predict · evaluate
├── common.py               # config loading, logging, seeding, path resolution, joblib I/O
├── config.yaml             # single source of truth: every hyper-parameter, path, target
├── conftest.py             # pytest bootstrap (adds project root to sys.path)
├── pytest.ini              # pytest configuration
├── requirements.txt        # pinned baseline versions
├── CONFORMAL_IMPLEMENTATION_AUDIT.md
├── README.md
├── DOCUMENTATION.md        # this file
├── data/
│   ├── splits/             # → link to ../../data/splits (mined ConfTest corpus)
│   ├── collect.py          # optional GitHub Actions history harvester (PyGitHub)
│   ├── features.py         # 62-feature builder (change / history / cross families)
│   ├── preprocess.py       # impute · scale · one-hot · chronological 70/15/15 split
│   ├── live_extract.py     # live per-PR feature extraction for the advisory CI job
│   └── processed/          # train.csv · val.csv · test.csv (generated)
├── models/
│   ├── train.py            # XGBoost fit + metrics
│   ├── calibrate.py        # Platt scaling + ECE report
│   ├── conformal.py        # split-conformal threshold (marginal / PAC)
│   ├── pipeline.py         # SelectionPipeline: load all four layers and predict
│   └── *.pkl               # trained artefacts (committed, ~1 MB)
├── api/
│   └── server.py           # Flask serving: /predict · /predict/batch · /health · /metrics
├── examples/
│   ├── advisory_report.py  # CI advisory Markdown builder (non-gating)
│   └── sample_candidates.json  # committed fallback sample
├── reports/                # training / calibration / conformal / evaluation JSON
└── tests/                  # pytest suite (144 tests)
```

## 5. Installation & environment

```bash
cd conformal-test-selection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Version compatibility.** `requirements.txt` pins the versions the project was
specified against (xgboost 1.7.5, scikit-learn 1.3.0, pandas 2.0.0, numpy
1.24.0). The source is written against the *intersection* of those APIs and
current releases (xgboost 3.x, scikit-learn 1.9, pandas 2.2), so it runs
unmodified under either set. The committed `models/*.pkl` artefacts were produced
with the newer resolves; if you install the pinned set, regenerate them with the
pipeline commands (§8). Note the sklearn version-skew gotcha: a preprocessor
pickled under one scikit-learn minor may warn or fail to unpickle under another —
regenerate rather than mix.

## 6. Data and the reused corpus

The mined **ConfTest corpus** — 561,711 commit–test rows with a strict
chronological 70/15/15 split — is **reused, not regenerated**. `data/splits` is a
symlink to the parent ConfTest repository's `data/splits` (the 300 MB corpus is
not copied into this project).

To recreate the link:

```bash
# from conformal-test-selection/
ln -s ../../data/splits data/splits        # macOS/Linux
# Windows (PowerShell, admin):
#   New-Item -ItemType SymbolicLink -Path data\splits -Target ..\..\data\splits
```

If symlinks are unavailable, copy the three CSVs into `data/splits/`. To use a
corpus you harvest yourself instead, set `data.external_splits_dir: null` in
`config.yaml` and populate `data/raw/` with `data/collect.py`.

**Label & grouping.** The label column is `label_failed`; rows are grouped by
`commit_sha` so a single commit's tests are never split across train/val/test —
this is what makes the temporal split honest (no leakage of a commit's outcome
across the split boundary).

## 7. Feature engineering

The model consumes **41 raw modelled columns**, which the preprocessor expands to
**62 transformed features** (one-hot encoding of the two categoricals `repo` and
`mutant_operator` accounts for the difference). `data/features.py` groups them
into three families:

**Change (commit-level) features** — identical across every test for one change:
`lines_added`, `lines_removed`, `total_churn`, `files_changed`,
`commit_message_length`, `has_source_change`, `has_test_change`,
`has_config_change`, `diff_num_src_files`, `diff_num_test_files`,
`diff_has_python`, `diff_is_fix_commit`, `diff_is_refactor_commit`,
`diff_msg_word_count`.

**Test-history features** — computed from *strictly prior* observations only (no
lookahead): `execution_count`, `failure_rate_lifetime`, `recent_failures`,
`duration_mean`, `flakiness_score`, `hist_prior_failures`, `hist_has_ever_failed`,
`co_change_frequency`, `hist_changed_files_prior_mod_count`, `test_name_length`,
plus the AST structure of the test file (`ast_test_file_functions_count`,
`ast_test_file_classes_count`, `ast_test_file_imports_count`,
`ast_test_file_complexity`, `ast_test_is_parameterized`).

**Cross (change × test interaction) features** — how related the change is to the
test: `file_test_similarity`, `lexical_similarity`, `path_overlap`, `same_module`,
`dependency_distance`, `dep_is_direct_import`, `dep_name_heuristic_coupled`,
`dep_is_reachable`, `dep_max_reverse_dependencies`, `dep_test_total_out_degree`.

**Categoricals:** `repo`, `mutant_operator` (one-hot; an unseen category encodes
as all-zero).

Similarity primitives live in `features.py`: `tokenize_identifier`, `jaccard`,
`dice`, `path_overlap_ratio`.

### Live extraction (`data/live_extract.py`)

The offline builder needs a labelled mined-history CSV and cannot run on a bare
checkout. `data/live_extract.build_live_candidates()` fills that gap for the
advisory CI job: it turns a *real PR diff* + repository checkout into candidate
records in exactly the shape `SelectionPipeline.predict` consumes.

- Only **statically computable** columns are produced from the diff and the
  test-file AST.
- The **8 historical features** and `mutant_operator` are deliberately left
  **absent** so the fitted preprocessor median-imputes them, rather than being
  materialised to a fake literal (which would tell the completeness check a
  feature is "present" when it was never measured).
- It is **crash-proof**: every function degrades to a neutral value or `[]`;
  `build_live_candidates` never raises, so the advisory step cannot go red.
- Git is driven through the CLI (GitPython is not installed in the advisory job),
  using a `base...head` merge-base diff so the whole PR is seen.

> **Bug history:** `_is_test_path` originally substring-matched `"test"` over the
> whole path. Because this project lives under `conformal-test-selection/`, every
> file was misclassified as a test, `changed_sources` was always empty, and the
> advisory silently fell back to the committed sample. It now classifies by
> pytest filename convention (`test_*` / `*_test.py`) plus residence in a
> `test`/`tests` directory. A regression test pins the fix.

## 8. Pipeline stages & CLI reference

Every command reads `config.yaml` and writes a report under `reports/`. Reproduce
the whole pipeline from the reused splits:

```bash
python cli.py preprocess    # impute, scale, encode, chronological split → data/processed/
python cli.py train         # fit XGBoost              → models/gbdt_model.pkl,         reports/training_report.json
python cli.py calibrate     # Platt scaling + ECE gate → models/calibrator.pkl,         reports/calibration_report.json
python cli.py conformal     # PAC threshold            → models/conformal_threshold.pkl, reports/conformal_report.json
python cli.py evaluate      # score against targets    → reports/evaluation.json (exit 1 if any gate fails)
```

| Command | Purpose | Reads | Writes |
|---|---|---|---|
| `collect` | Optional: harvest CI history from GitHub Actions | `data/repos.txt`, GitHub API | `data/raw/*.csv` |
| `features` | Build the feature matrix from harvested history | `data/raw/test_history.csv` | `data/processed/features.csv` |
| `preprocess` | Impute, scale, one-hot, chronological group split | splits or `features.csv` | `data/processed/{train,val,test}.csv`, `preprocessor.pkl` |
| `train` | Fit the gradient-boosted model | `train`/`val` CSVs | `gbdt_model.pkl`, `training_report.json` |
| `calibrate` | Fit Platt scaling, report ECE (gate ≤ 8 %) | `val` CSV, model | `calibrator.pkl`, `calibration_report.json` |
| `conformal` | Fit the coverage-guaranteed threshold | calibration split | `conformal_threshold.pkl`, `conformal_report.json` |
| `predict` | Select tests for a change | JSON/CSV candidates | `--output` selection JSON (or stdout) |
| `evaluate` | Score the pipeline against `config.targets` | `test` CSV, all artefacts | `evaluation.json` (exit 1 on gate failure) |

### `predict`

`predict` takes a JSON or CSV of candidate test *records* (the feature columns
`features.py` produces), **not** a raw commit — feature building is a separate,
cacheable step.

```bash
python cli.py predict candidates.csv --output selection.json
# stdout:
#   Selected 45 of 120 tests (37.5%), saving 62.5% of CI cost.
#   Guarantee: at least 95% of failing tests selected, 90% confidence (PAC)
#     [RUN ] p=0.8123  repoX::tests/test_auth.py::test_login
#     [skip] p=0.0041  repoX::tests/test_docs.py::test_readme
```

### `evaluate`

Writes the *measured* column of the README results table from
`reports/evaluation.json` and marks each acceptance gate PASS/FAIL against
`config.targets`. `python cli.py evaluate --update-readme` regenerates the table
in place between the `RESULTS` markers so it never drifts from the JSON. The
command exits non-zero if any gate fails, so it doubles as a CI check.

## 9. Configuration reference

`config.yaml` is the single source of truth; no module hard-codes a value that
appears here. It is read through `common.load_config()` (cached).

| Block | Key | Meaning |
|---|---|---|
| `project` | `random_seed` | Global RNG seed (42) for reproducibility |
| `model` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | XGBoost hyper-parameters |
| | `eval_metric` | `logloss` |
| | `scale_pos_weight` | `auto` (= n_neg/n_pos from train split), a float, or `null` to disable |
| | `early_stopping_rounds`, `n_jobs` | Early stopping patience; `-1` = all cores |
| `calibration` | `method` | `platt` (sigmoid) or `isotonic` |
| | `cv_folds`, `ece_bins` | Calibration CV folds; reliability-diagram bin count |
| | `ece_threshold` | **Acceptance gate**: ECE must be ≤ this after calibration (0.08) |
| `conformal` | `coverage` | `1 − α`: fraction of *failing* tests to retain (0.95) |
| | `confidence` | `1 − δ`: confidence the coverage holds, PAC mode (0.90) |
| | `method` | `split` |
| | `guarantee` | `marginal` or `pac` (default) |
| `selection` | `threshold` | Fallback score cut-off when no conformal model is loaded (0.7) |
| | `min_tests` | Never hand CI fewer than this many tests (10) |
| | `max_tests` | Hard ceiling on a selected subset (1000) |
| | `min_feature_completeness` | Below this fraction of supplied features, a prediction is flagged `degraded` (0.30) |
| `data` | `external_splits_dir` | `data/splits` link, or `null` to use harvested `data/raw` |
| | `label_column`, `group_column` | `label_failed`, `commit_sha` |
| | `split_ratios` | `[0.70, 0.15, 0.15]` |
| `collection` | `repos_file`, `max_repos`, `max_runs_per_repo`, `request_delay_seconds`, `rate_limit_buffer` | GitHub Actions harvest controls |
| `artifacts` | `model_path`, `calibrator_path`, `conformal_path`, `preprocessor_path`, `reports_dir` | Artefact locations |
| `targets` | `recall` ≥ 0.92, `selection_rate` ≤ 0.40, `ece` ≤ 0.08, `coverage` ≥ 0.95, `cost_reduction` ≥ 0.60 | **Acceptance gates** checked by `evaluate` (goals, not measurements) |
| `logging` | `level`, `format` | Logging configuration |

## 10. Serving contract — `SelectionPipeline.predict`

`models/pipeline.py` wires the four artefacts and is the single scoring entry
point. `SelectionPipeline.load()` reads every artefact named in `config.artifacts`
and raises `PipelineError` (naming the command that produces the missing one) if
any is absent.

```python
from models.pipeline import SelectionPipeline
pipeline = SelectionPipeline.load()
result = pipeline.predict(
    [{"test_id": "tests/test_api.py::test_ok",
      "test_path": "tests/test_api.py",
      "changed_file_path": "src/api.py"}],
    min_tests=None,   # defaults to selection.min_tests
    max_tests=None,   # defaults to selection.max_tests
)
```

**Partial input is accepted and reported.** Absent modelled columns are filled
with `NaN` (not zero — zero would read as a real measurement) and median-imputed
by the preprocessor. Every response carries `feature_completeness`; a request
below `selection.min_feature_completeness` is flagged `degraded` rather than
silently answered.

**Response shape:**

```jsonc
{
  "predictions": [
    { "test_id": "...", "failure_probability": 0.7076,
      "selected": true, "margin": 0.6695 }      // margin = P(fail) − probability_floor
  ],
  "summary": {
    "n_candidates": 3, "n_selected": 2,
    "selection_rate": 0.6667, "cost_reduction": 0.3333,
    "feature_completeness": 1.0, "degraded": false,
    "budget_capped": false
  },
  "guarantee": {
    "type": "pac", "target_coverage": 0.95, "confidence": 0.90,
    "certified_coverage": 0.9501, "probability_floor": 0.038145,
    "class_conditional": true, "calibration_size": 4578,
    "scope": "per failing test (marginal over the failing class), not per commit",
    "assumptions": "Valid under exchangeability ...; distribution shift can weaken it.",
    "claim": "With probability 90% over the calibration draw, at least 95% of failing tests are selected.",
    "holds": true
  }
}
```

**`min_tests` / `max_tests` semantics.** These are operational overrides applied
*after* the conformal rule:

- `min_tests` **tops up** the selection with the next-riskiest tests. Adding
  tests can only help coverage, so it never voids the guarantee.
- `max_tests` can **remove** a test the rule selected. That *does* void the
  guarantee: the response sets `summary.budget_capped = true` and
  `guarantee.holds = false` with an explanatory `note`.

Helper functions in the module: `load_pipeline()`, `records_from_frame()`
(dataframe → records with `NaN`→`None`), `iter_batches()`.

## 11. HTTP API

`api/server.py` is a Flask service that loads the artefacts **once at import** and
reuses them (per-request loading would dominate latency). A load failure is not
fatal at start-up: `/health` reports `degraded` and the prediction routes return
`503`, so a container can come up and be diagnosed rather than crash-loop.

```bash
python -m api.server                 # or: python api/server.py --host 0.0.0.0 --port 5000
curl localhost:5000/health
```

| Route | Method | Purpose | Codes |
|---|---|---|---|
| `/predict` | POST | Select tests for one change | 200 / 400 / 503 |
| `/predict/batch` | POST | Select tests for several changes | 200 / 400 / 503 |
| `/health` | GET | Liveness + artefact readiness + guarantee summary | 200 / 503 |
| `/metrics` | GET | Most recent held-out evaluation | 200 / 404 |

**Request body (`/predict`).** Accepts `{"tests": [...]}`, a bare list, or a
single object. A top-level `change` block is merged into every test row (test-level
keys win), so a caller need not repeat change features per test. `min_tests` /
`max_tests` may be supplied at the top level.

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"change": { "lines_added": 120 },
       "tests": [ { "test_id": "tests/test_auth.py::test_login", "test_path": "tests/test_auth.py" } ]}'
```

**Batch (`/predict/batch`).** Expects `{"changes": [ ... ]}`. Each change is scored
independently, so one malformed entry returns an error for *that entry alone*; the
response includes a `batch_summary` (n_changes, n_succeeded, n_failed, aggregate
candidates/selected, cost_reduction).

**Safety limits:** `MAX_TESTS_PER_REQUEST = 20 000`, `MAX_CHANGES_PER_BATCH = 100`.
Errors return JSON (custom 404/405/500 handlers avoid leaking HTML or stack
traces). **The dev server is single-threaded and not hardened — front a real
deployment with gunicorn or waitress.**

## 12. Advisory CI integration & live PR extraction

The repository ships a GitHub Actions workflow,
`.github/workflows/conformal-advisory.yml`, that posts a **non-gating advisory**
comment on pull requests. It **selects nothing and skips no tests** — it reports
what the conformal rule *would* do, so its realized coverage can be watched on
real PRs before it is ever trusted to gate.

Why it can run on a bare checkout: `models/*.pkl` and `reports/*.json` are
committed (~1 MB). The 300 MB corpus is not, so `cli.py evaluate` cannot re-measure
in CI — instead the job surfaces the committed held-out metrics and runs a live
`predict`.

**How the advisory is built (`examples/advisory_report.py`):**

1. `_candidates()` prefers features extracted from the **real PR** via
   `data.live_extract.build_live_candidates()`, using the workflow-provided env
   `BASE_SHA` / `HEAD_SHA` / `REPO_NAME` / `GITHUB_WORKSPACE`.
2. If there is no diff, no changed source, or no discoverable test (e.g. a
   `workflow_dispatch` run or a bare checkout), it **falls back** to
   `examples/sample_candidates.json` so the job always produces output.
3. It scores with `SelectionPipeline.predict(min_tests=1)` (the pure conformal
   rule, not the smoke-set floor), renders a Markdown table (top-20 by risk), and
   writes it to `$COMMENT_PATH` / `$GITHUB_STEP_SUMMARY`.
4. It **never raises** out to the caller — an advisory step must not turn a build
   red.

**Out-of-distribution caveat.** When run live on *this* repo, the model is far
from its training corpus: the `repo` category is unseen (all-zero one-hot), there
is no `mutant_operator`, and the 8 historical features are median-imputed. The
advisory therefore prints a prominent warning that the absolute `P(fail)` values
and the PAC guarantee **do not transfer** to this repository — the ranking of
which tests relate to the change is still informative, but it should be read as a
*relative* ranking, not calibrated risk. See
`CONFORMAL_IMPLEMENTATION_AUDIT.md`.

**Workflow requirements.** `actions/checkout` uses `fetch-depth: 0` so both PR
endpoints, the merge-base, and `git log` (for prior-modification counts) are
reachable.

There is also a standard `.github/workflows/ci.yml` that runs the pytest suite on
every push and pull request.

## 13. Results & scope of the guarantee

Measured on the held-out test split (86,469 rows, 3,208 failing) by
`python cli.py evaluate`:

| Metric | Target | Measured | Status |
|---|---|---|---|
| Recall (failures caught) | ≥ 92 % | **95.23 %** | PASS |
| Selection rate | ≤ 40 % | **25.25 %** | PASS |
| Calibration error (ECE) | ≤ 8 % | **0.58 %** | PASS |
| Conformal coverage | ≥ 95 % | **95.23 %** | PASS |
| Cost reduction | ≥ 60 % | **74.75 %** | PASS |

**Baselines that matter.** *Cost-matched top-k* selects the same *number* of
tests ranked by probability and reproduces the conformal result exactly —
confirming conformal prediction is not a better *ranker*; its contribution is
choosing *how many* tests to run without peeking at labels. *Val-tuned threshold*
(pick the lowest threshold hitting 95 % recall on validation, then ship it)
under-delivers on the future test split — the gap is the finite-sample correction
conformal prediction adds.

**Scope, stated honestly.** The 95 % figure is *per failing test*: of the tests
that were going to fail, at least 95 % are selected. It is **not** a promise that
every failing test in a commit is caught. At the commit level the rule fully
catches **65.2 %** of failing pushes and catches *at least one* failing test in
**95.6 %** (`reports/conformal_report.json`). The guarantee is conditional on
*exchangeability* between the calibration split and future commits; because the
split is deliberately temporal, distribution drift can erode it, and the measured
coverage is evidence it held on this corpus, not a proof it holds everywhere. It
is additionally voided if `max_tests` drops a selected test (`holds: false`).

## 14. Testing

```bash
python -m pytest -q          # 144 tests
```

The suite covers feature construction (`test_features.py`), preprocessing
(`test_preprocess.py`), training (`test_train.py`), calibration
(`test_calibrate.py`), conformal threshold selection (`test_conformal.py`), the
serving pipeline (`test_pipeline.py`), live PR extraction (`test_live_extract.py`),
and shared utilities (`test_common.py`). `tests/conftest.py` and the top-level
`conftest.py` add the project root to `sys.path`.

## 15. Module reference

**`common.py`** — shared infrastructure. `load_config()` (cached YAML),
`setup_logging()` / `get_logger()`, `set_seed()`, `resolve_path()` (config-relative
→ project root), `save_artifact()` / `load_artifact()` (joblib). Exceptions:
`ConfigError`, `ArtifactError`.

**`data/features.py`** — `build_features()` runs the three extractors
(`extract_change_features`, `extract_test_features` — strictly-prior observations
only, `extract_cross_features`) and assembles the model-ready frame. Similarity
primitives: `tokenize_identifier`, `jaccard`, `dice`, `path_overlap_ratio`.
Exception: `FeatureError`.

**`data/preprocess.py`** — `preprocess()` runs impute → scale (numeric) / impute →
one-hot (categorical) and writes the split CSVs. `temporal_group_split()` splits
chronologically while keeping every row of a commit on one side.
`build_preprocessor()` assembles the column transformer. `adapt_external_frame()`
maps the mined corpus onto this project's schema. Exception: `PreprocessError`.

**`data/collect.py`** — optional GitHub Actions harvester. `collect()` walks
workflow runs and flattens JUnit/pytest reports into `(commit, test)` rows.
`parse_junit_xml`, `parse_pytest_log`, `collect_repo`, `read_repo_list`;
`RateLimiter` (politeness + quota block), `TestObservation`. Exception:
`CollectionError`.

**`data/live_extract.py`** — live per-PR feature extraction (§7).
`build_live_candidates()` (never raises), `pr_diff()`, `diff_level_features()`,
`ast_metrics()`, `discover_tests()`.

**`models/train.py`** — Layer 1. `train()` fits XGBoost, evaluates on all splits,
persists the model. `build_model()`, `classification_metrics()`, `load_split()`.
Exception: `TrainingError`.

**`models/calibrate.py`** — Layer 2. `calibrate()` fits on the validation split and
reports ECE before/after. `ScoreCalibrator` (`.fit` / `.transform`),
`expected_calibration_error()` (ECE, MCE, reliability bins),
`load_calibrated_scorer()`. Exception: `CalibrationError`.

**`models/conformal.py`** — Layer 3 (the contribution). `build_selector()` fits
thresholds from a held-out calibration split; `ConformalSelector` (`.select`,
`.to_dict`); `marginal_rank()`, `pac_rank()` (≥ marginal rank),
`nonconformity_scores()`, `selection_metrics()`, `fit_conformal()`,
`load_selector()`. Exception: `ConformalError`.

**`models/pipeline.py`** — `SelectionPipeline` (§10): `.load`, `.predict`,
`.probabilities`, `.guarantee_statement`. Helpers `load_pipeline`,
`records_from_frame`, `iter_batches`. Exception: `PipelineError`.

**`api/server.py`** — Flask app (§11): `create_app()`, route handlers, `main()`.

## 16. Limitations & caveats

- **Guarantee is per-failing-test and exchangeability-conditional**, not
  per-commit and not distribution-free across repos (§13). Monitor realized
  coverage in deployment.
- **Live advisory on this repo is out-of-distribution** — read as a relative
  ranking (§12).
- **`max_tests` voids the guarantee** when it drops a selected test.
- **`predict` consumes feature records, not raw commits** — feature building is a
  separate step (offline `features.py`, or `live_extract.py` for a live PR).
- **The Flask dev server is not production-hardened** — use gunicorn/waitress.
- **Pickle/scikit-learn version skew** — regenerate `.pkl` artefacts if you change
  the scikit-learn minor version rather than mixing.

## 17. References

- Machalica et al., *Predictive Test Selection*, Meta — arXiv:1810.05286
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* —
  arXiv:2107.07511
- Vovk, *Conditional validity of inductive conformal predictors* (2012) —
  PAC-style coverage

---

*Final Year B.Tech (CSE) major project. This document reflects the codebase as of
its last update; regenerate the results tables with `python cli.py evaluate
--update-readme` and keep this file in step with `config.yaml`.*
