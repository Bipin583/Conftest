# Confidence-Calibrated Test Selection with Conformal Guarantees for CI/CD Optimization

A machine-learning system that predicts which regression tests a code change is
likely to fail, runs only those, and does so with a **distribution-free
coverage guarantee** — valid under the standard conformal assumption that the
calibration and future commits are *exchangeable* — on how many real failures it
will still catch. On the held-out test split it retains **95.2% of failing tests
while running only 25.3% of the suite**, cutting CI cost by **74.7%**.

> Final Year B.Tech CSE Major Project. The novelty is the third layer: to our
> knowledge this is the first application of **split conformal prediction** to
> regression test selection, turning "the model is fairly confident" into "at
> least 95% of failures are caught, with 90% confidence."

---

## The problem

Large test suites dominate CI/CD time and cost. Most changes break nothing, so
running every test on every push wastes 30–50% of CI spend re-confirming green.
The obvious fix — a model that skips tests it thinks will pass — is dangerous
without a guarantee: a model that is *miscalibrated* or *overconfident* silently
lets real failures through, and you only find out in production.

## The approach — three layers

1. **Gradient-boosted classifier (`models/train.py`).** XGBoost over 62
   features describing the change, each test's history, and the change↔test
   relationship (dependency reachability, co-change frequency, lexical/path
   similarity, mutation-operator signals, AST counts). Class imbalance (~5%
   failing rows) is handled with `scale_pos_weight`.
2. **Probability calibration (`models/calibrate.py`).** Platt scaling so a
   predicted 0.30 means a 30% empirical failure rate. Gated on Expected
   Calibration Error (ECE ≤ 8%).
3. **Conformal selection (`models/conformal.py`).** Split conformal prediction
   converts calibrated probabilities into a selection threshold with a
   **PAC guarantee**: with ≥90% confidence over the calibration draw, at least
   95% of failing tests are selected. This is the layer that makes skipping
   tests safe rather than merely cheap. Two caveats keep the claim honest: it is
   a guarantee about *individual failing tests* (marginal over the failing
   class), not about catching *every* failing test in a commit — that per-commit
   figure is lower (see below); and like any conformal guarantee it assumes the
   calibration and future rows are *exchangeable*, an assumption a temporal or
   cross-project shift can weaken, so realized coverage should be monitored in
   deployment.

## Results

<!-- RESULTS:START -->

Measured on the held-out test split (86,469 rows, 3,208 failing) by `python cli.py evaluate`.

| Metric | Target | Measured | Status |
|---|---|---|---|
| Recall (failures caught) | &ge; 92% | **95.23%** | PASS |
| Selection rate | &le; 40% | **25.25%** | PASS |
| Calibration error (ECE) | &le; 8% | **0.58%** | PASS |
| Conformal coverage | &ge; 95% | **95.23%** | PASS |
| Cost reduction | &ge; 60% | **74.75%** | PASS |

Conformal guarantee in force: **PAC** -- with 90% confidence over the calibration draw, at least 95.01% of failing tests are selected (threshold: run a test when P(fail) &ge; 0.0381, fitted on 4,578 calibration failures).

| Strategy | Recall | Selection rate | Cost saving |
|---|---|---|---|
| **Conformal selection (this work)** | **95.23%** | 25.25% | 74.75% |
| run all | 100.00% | 100.00% | 0.00% |
| fixed threshold 0.5 | 8.60% | 0.74% | 99.26% |
| cost matched topk | 95.23% | 25.25% | 74.75% |
| val tuned threshold | 94.98% | 25.07% | 74.93% |

<!-- RESULTS:END -->

The two baselines that matter: **cost-matched top-k** selects the same *number*
of tests ranked by probability and reproduces the conformal result exactly —
confirming conformal prediction is not a better *ranker*; its contribution is
choosing *how many* tests to run without peeking at the labels. **val-tuned
threshold** is the honest competitor: pick the lowest threshold hitting 95%
recall on validation, then ship it — and watch it under-deliver on the future
test split. The gap it leaves is the finite-sample correction conformal
prediction adds.

The table above is regenerated in place (between the `RESULTS` markers) by
`python cli.py evaluate --update-readme`, so it never drifts from the numbers in
`reports/evaluation.json`.

### Scope of the guarantee

The 95% figure is *per failing test*: of the tests that were going to fail, at
least 95% are selected. It is **not** a promise that every failing test in a
commit is caught. At the commit level the rule fully catches **65.2%** of
failing pushes and catches *at least one* failing test in **95.6%**
(`reports/conformal_report.json`). The coverage guarantee is also conditional on
*exchangeability* between the calibration split and future commits; because the
split is deliberately temporal, distribution drift can erode it, and the 95.23%
measured on the held-out test split is evidence it held on this corpus rather
than a proof it holds everywhere. The guarantee is additionally voided if the
`max_tests` cap drops a selected test — the serving response reports this by
setting `holds: false`.

## Repository layout

```
conformal-test-selection/
├── cli.py                 # collect · features · preprocess · train · calibrate · conformal · predict · evaluate
├── common.py              # config loading, logging, path resolution
├── config.yaml            # single source of truth for every hyper-parameter and target
├── data/
│   ├── splits/            # -> symlink to ../../data/splits (mined ConfTest corpus, 561,711 rows)
│   ├── collect.py         # optional GitHub Actions history harvester (PyGitHub)
│   ├── features.py        # 62-feature builder (change / test-history / cross features)
│   ├── preprocess.py      # impute · scale · one-hot · chronological 70/15/15 split
│   └── processed/         # train.csv · val.csv · test.csv (generated)
├── models/
│   ├── train.py           # XGBoost fit + metrics
│   ├── calibrate.py       # Platt scaling + ECE report
│   ├── conformal.py       # split-conformal threshold (PAC)
│   ├── pipeline.py        # SelectionPipeline: load all three layers and predict
│   └── *.pkl              # trained artefacts
├── api/server.py          # Flask serving: /predict · /predict/batch · /health · /metrics
├── reports/               # training / calibration / conformal / evaluation JSON
└── tests/                 # pytest suite
```

## Installation

```bash
cd conformal-test-selection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data reuse

The mined ConfTest corpus (561,711 commit–test rows, strict chronological
70/15/15 split) is **reused, not regenerated**. `data/splits` is a symlink to
the parent ConfTest repository's `data/splits`. To recreate the link:

```bash
# from conformal-test-selection/
ln -s ../../data/splits data/splits        # macOS/Linux
# Windows (PowerShell, admin): New-Item -ItemType SymbolicLink -Path data\splits -Target ..\..\data\splits
```

If symlinks are unavailable, copy the three CSVs into `data/splits/` instead.
Set `data.external_splits_dir: null` in `config.yaml` to fall back to a corpus
harvested locally by `data/collect.py`.

## Usage

Reproduce the whole pipeline from the reused splits (each step reads
`config.yaml` and writes a report under `reports/`):

```bash
python cli.py preprocess          # impute, scale, encode, chronological split -> data/processed/
python cli.py train               # fit XGBoost              -> models/gbdt_model.pkl, reports/training_report.json
python cli.py calibrate           # Platt scaling + ECE gate -> models/calibrator.pkl, reports/calibration_report.json
python cli.py conformal           # PAC threshold            -> models/conformal_threshold.pkl, reports/conformal_report.json
python cli.py evaluate            # score against targets    -> reports/evaluation.json (exit 1 if any gate fails)
```

Select tests for a change. `predict` takes a JSON or CSV of candidate test
records (the feature columns `features.py` produces), not a raw commit — the
feature build is a separate, cacheable step:

```bash
python cli.py predict candidates.csv --output selection.json
# stdout:
#   Selected 45 of 120 tests (37.5%), saving 62.5% of CI cost.
#   Guarantee: at least 95% of failing tests selected, 90% confidence (PAC)
#     [RUN ] p=0.8123  repoX::tests/test_auth.py::test_login
#     [skip] p=0.0041  repoX::tests/test_docs.py::test_readme
```

Serve it:

```bash
python -m api.server            # Flask on :5000
curl localhost:5000/health
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
     -d '{"tests": [ { ...feature record... } ]}'
```

## Configuration

Every hyper-parameter, artefact path, and acceptance target lives in
`config.yaml`; no module hard-codes a value that appears there. The
**targets** block holds the acceptance gates (recall ≥ 0.92, selection ≤ 0.40,
ECE ≤ 0.08, coverage ≥ 0.95, cost reduction ≥ 0.60) that `cli.py evaluate`
checks — these are goals, not measurements. The **conformal** block selects the
guarantee flavour: `marginal` (coverage holds in expectation) or `pac` (holds
with probability ≥ `confidence`).

## Environment compatibility

`requirements.txt` pins the versions the project was specified against
(xgboost 1.7.5, scikit-learn 1.3.0, pandas 2.0.0, numpy 1.24.0). The source is
written against the intersection of those APIs and current releases (xgboost 3.x,
scikit-learn 1.9, pandas 2.2), so it runs unmodified under either set. Trained
`.pkl` artefacts in `models/` were produced with the newer resolves; regenerate
them with the pipeline commands above if you install the pinned set.

## Testing

```bash
python -m pytest -q
```

The suite covers feature construction, preprocessing, training, calibration,
conformal threshold selection, the serving pipeline, and shared utilities. CI
(`.github/workflows/ci.yml`) runs it on every push and pull request.

## Novelty & references

Confidence-based test selection is not new (Meta's Predictive Test Selection is
the reference system). What is new here is attaching a **conformal guarantee**
to the selection, so the recall promise is distribution-free and holds on unseen
future commits rather than resting on the assumption that the model stays
calibrated.

- Machalica et al., *Predictive Test Selection*, Meta — arXiv:1810.05286
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* — arXiv:2107.07511
- Vovk, *Conditional validity of inductive conformal predictors* (2012) — PAC-style coverage

## Team

Final Year B.Tech, Computer Science & Engineering. _Add team member names, roll
numbers, and guide here._

