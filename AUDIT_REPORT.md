# ConfTest — Audit, Fix and Finalize Report

**Date:** 2026-09-07
**Scope applied:** Critical + Important (per your decision). Cosmetic and minor issues were catalogued, not changed.
**Headline policy applied:** documentation restated to measured truth. No model was retrained, no threshold re-tuned, no metric re-measured — every number below is read from an existing artifact under `reports/`, or from a command run during this audit.

---

## Executive summary

The project is architecturally sound and its test suite is strong (700 tests, 78% statement coverage of `src/conftest`). The defects found were not in the machine learning plumbing — they were in the seams between the code and the claims: four documented entry points that did not exist, two documented import paths that did not exist, a report path that only resolved when the API was started from one directory, and a calibration payload that reported two different splits' numbers without saying which was which.

One finding is more serious than the rest and is the reason the ranker is weak. **The shipped ensemble's members hold 5, 5, 5, 6 and 7 boosting rounds — not the 150 the metadata advertises.** Early stopping fires almost immediately and the documentation (including a line I wrote earlier in this engagement) reported the configured ceiling as if it were the fitted model. That is now corrected, and it reframes the negative result: the 3.11% reduction at 100% recall is the measured behaviour of a *five-tree* ensemble, not evidence that these features cannot rank tests.

---

## 1. Critical issues fixed

### C1 — The shipped ensemble is 5–7 trees, documented as 150
- **Where:** `models/README.md:5`; `models/ensembles/5_seed_lgbm/ensemble_metadata.json` (`"n_estimators": 150`)
- **Why it matters:** `n_estimators` is the *budget*, and `best_iteration_` is what was kept. Verified directly:

  | member | seed | trees in booster |
  |---|---|---|
  | 1 | 42 | 7 |
  | 2 | 101 | 5 |
  | 3 | 2024 | 5 |
  | 4 | 777 | 5 |
  | 5 | 999 | 6 |

  A reader benchmarking against "a 150-tree LightGBM ensemble" is comparing against something 20–30× smaller. It also supplies the missing explanation for held-out PR-AUC 0.1393 against a 3.71% positive rate, and for why leave-one-group-out ablation *improves* the ranking when history or AST features are removed — a five-round fit has no capacity to use them.
- **Fix:** `models/README.md` now states the per-member tree counts, names the verification (`booster_.num_trees()`), says explicitly that 150 is "a budget the fit never spent", and identifies this as the first thing to change before concluding the features are inadequate.
- **Verify:** `python -c "import joblib,pathlib; [print(f.name, joblib.load(f).model.booster_.num_trees()) for f in sorted(pathlib.Path('models/ensembles/5_seed_lgbm').glob('member_*.joblib'))]"`

### C2 — Shipped ensemble metadata predates the row-bagging fix
- **Where:** `models/ensembles/5_seed_lgbm/ensemble_metadata.json` — no `subsample_freq` key, and absolute Windows paths in `member_files`
- **Why it matters:** `LGBMClassifier` ignores `subsample` unless `subsample_freq > 0`. The shipped members were therefore trained on 100% of rows each and differ only by seed-dependent tie-breaking, so the deep ensemble's spread understates epistemic uncertainty — and every published σ (mean 0.0114, p95 0.0345) inherits that. The absolute paths bake in one machine's checkout.
- **Fix:** documented in `models/README.md` rather than silently retrained (your no-retraining decision). The loader already handles both: it resolves members beside the metadata and emits a warning on every load of a pre-fix artifact — confirmed firing during this audit.
- **Verify:** load the ensemble and watch for `predates the row-bagging fix` in the log.

---

## 2. Important issues fixed

### I1 — Four documented entry points did not exist
`scripts/init_db.py`, `scripts/collect_data.py`, `scripts/train.py`, `scripts/evaluate.py` are named in the README, `DOCUMENTATION.md` and the CI recipe; none were present. A documented command that does not exist is a documentation defect that a reader discovers only after cloning.

Created as **thin delegators, not second implementations** — each rewrites `sys.argv[0]` and hands off via `runpy`, so `--help` prints the real script's own usage and there is exactly one set of flags to keep correct:

| new file | delegates to | why that target |
|---|---|---|
| `scripts/init_db.py` | `conftest.db.init_db.init_db` | the API and tests already initialise schemas in-process |
| `scripts/collect_data.py` | `scripts/collect_repository_data.py` | verbatim argv forwarding |
| `scripts/train.py` | `scripts/train_ensemble.py` | the 5-seed ensemble is what ships; the single-model trainer produces an artifact no shipping path loads |
| `scripts/evaluate.py` | 11 producers, by stage | see I2 |

### I2 — No way to see which script produced which published number
`scripts/evaluate.py` is a dispatcher rather than a twelfth evaluation. It maps all 20 artifacts under `reports/` to the one script that writes each, shows which are present, and forwards to a producer on request (`--run <stage>`, `--all`, with flags after `--`). This mirrors the discipline already in the code, where modules refuse to serve a number whose artifact is missing and name the producing command instead of guessing.

### I3 — Two documented import paths did not exist
`conftest.features.extractor` and `conftest.engine.abstention` are the names used throughout the docs; the implementations are `conftest.features.pipeline` and `conftest.models.policy` + `conftest.engine.selector_engine`. Added as **pure re-export shims** with no logic, plus `tests/unit/test_documented_import_paths.py` (7 tests) which pins that they re-export the *same objects* and fails if either shim ever grows a function of its own. The abstention shim's docstring refuses to name a default τ, because there isn't one — it points at `models/policy_config.json` and `is_tuned`.

### I4 — Calibration report path resolved against the working directory
- **Where:** `src/conftest/api/routes/calibration.py:22` — `REPORT_PATH = Path("./reports/calibration_report.json")`
- **Why it matters:** started from anywhere but the project root, the endpoint answered **503 "no calibration report"** and told the caller to re-run a measurement that was already on disk. The only CWD-relative path in the API.
- **Fix:** anchored to `PROJECT_ROOT` from `conftest.config`.
- **Verify:** `cd /tmp && PYTHONPATH=<repo>/src python -m uvicorn conftest.api.main:app --port 8011` then `curl :8011/api/v1/calibration` — confirmed returning 200 with real metrics during this audit (it would have 503'd before).

### I5 — The calibration payload reported two splits' numbers with nothing saying which
- **Where:** `src/conftest/api/routes/calibration.py`, `src/conftest/api/schemas.py`
- **Why it matters:** the response carried `uncalibrated.ece = 0.0415 → calibrated.ece = 0.0161` (held-out test) beside a `selection_reason` quoting "validation ECE 0.0242 (vs 0.0390 uncalibrated)". Both are correct and they come from different splits, but a dashboard reader comparing the payload against the documentation sees a contradiction.
- **Fix:** added `metrics_split` ("held-out test split") and `selection_split` (read from the report's `chosen_on`, i.e. "validation holdout").

### I6 — Local `.env` had drifted from `.env.example`
It still declared `CONFTEST_MODEL_PATH`, `CONFTEST_DEFAULT_RISK_TOLERANCE` and `CONFTEST_ABSTENTION_THRESHOLD=0.15` — all three removed from `config.py`, and the last one 7.5× the τ actually in force (0.02). Because `Settings` uses `extra="ignore"`, they were silently discarded. Five current keys (`CONFTEST_ENSEMBLE_PATH`, `CONFTEST_CALIBRATOR_PATH`, `CONFTEST_POLICY_CONFIG_PATH`, `CONFTEST_DEFAULT_REPO_ROOT`, `CONFTEST_CORS_ALLOW_ORIGINS`) were absent. Refreshed from the template; `.gitignore` now covers `.env.backup*` and `.env.local` with a comment explaining why. *(The old file held no real secrets — the webhook value was the public placeholder and the GitHub token was empty — so the backup was removed after verification.)*

### I7 — Invalid escape sequences in two dashboard files
`dashboard/app.py:133` and `dashboard/pages/2_📉_Confidence_Calibration.py:153` embedded LaTeX (`\{`, `\%`, `\hat`) in non-raw strings. These are `DeprecationWarning` today and a `SyntaxError` in a future Python. Converted to `rf"""` and `r"""`. A tree-wide sweep of `dashboard/`, `src/`, `scripts/` and `tests/` now reports **zero** files with invalid escape sequences; suite warnings dropped from 14 to 9.

### I8 — Test count wrong in seven documents
Every doc claimed `628/628`. The measured figure moved during the audit as files were added (628 → 684 → 693 → 700, the growth being parametrised producer-reference cases generated by the new files themselves). All seven sites now read `700/700` with the command and date: `DOCUMENTATION.md`, `README.md`, `RELEASE_NOTES.md`, `docs/DOCUMENTATION_DETAILED.md`, `docs/KTU_Phase1_Presentation_Deck.html`, `slides/viva_presentation.html`, `slides/viva_presentation.md`.

### I9 — CI already ran pytest (verified, no change)
`.github/workflows/conftest.yml:44` already executes `python -m pytest tests/ -v --cov=src/conftest --cov-report=term-missing`, alongside a fabricated-label rejection step. Checked rather than assumed; nothing to fix.

---

## 3. Findings reported, deliberately not fixed

These need re-measurement or are outside the agreed scope. Each is real.

| # | Finding | Evidence |
|---|---|---|
| F1 | 13 of 32 features are constant across the training split, **including all 12 diff/churn features**. `diff_churn_only` therefore scores ROC-AUC exactly 0.5000 with 0 informative features. Direct inspection after the audit traced this to the dataset protocol: every mutation truthfully represents one changed Python source line in one file and has no commit message. These fields are inert for this corpus; inventing varied churn or prose would fabricate inputs. | `scripts/build_real_dataset.py`; `data/processed/real_features_manifest.json`; `reports/ablation_study.json` |
| F2 | A `POST /api/v1/select` against the sample suite returned byte-identical `raw_score` 0.0447 for all 10 tests. That observation is consistent with a weak five-round model and limited varying signal, but it is not evidence that the Git diff extractor is broken. | reproduced during this audit |
| F3 | The pre-registered 95% recall floor is **not met**. `gate_met: false`. At the validation-picked τ, held-out recall is 87.69% [75.77, 95.81] with 15 escaped commits — an 8.33 pp generalization gap. | `reports/g5_recall_floor.json` |
| F4 | `src/conftest/groundtruth/bugsinpy_adapter.py` — 650 statements at **0% coverage**. The single largest untested module. | coverage run below |
| F5 | Time-reduction Cliff's δ is **negative against every subset baseline** (−0.906 to −0.998), while recall δ is strongly positive. Reporting only the recall effect sizes would be cherry-picking. | `reports/statistical_significance.json` |
| F6 | Your Phase 4 step 7 names `GET /api/commits/abc123/select` and Phase 1 expects `/predict`. Neither exists and neither is claimed by any project document. The real routes are `GET /health` and `POST /api/v1/select`; explanations are `POST /api/v1/explain`. | `app.openapi()` |

---

## 4. Components completed

- [x] `scripts/init_db.py` — created (delegator)
- [x] `scripts/collect_data.py` — created (delegator)
- [x] `scripts/train.py` — created (delegator)
- [x] `scripts/evaluate.py` — created (stage → producer dispatcher)
- [x] `src/conftest/features/extractor.py` — created (re-export shim)
- [x] `src/conftest/engine/abstention.py` — created (re-export shim)
- [x] `src/conftest/models/calibration.py` — already present, no change needed
- [x] `tests/unit/test_documented_import_paths.py` — created (7 tests)
- [x] Calibration route: path anchoring + split labelling
- [x] `.env` / `.gitignore` hygiene
- [x] Escape-sequence fixes across the dashboard
- [x] Test-count corrections in 7 documents

---

## 5. End-to-end run (your Phase 4 sequence)

| # | Step | Result |
|---|---|---|
| 1 | `pip install -r requirements.txt` | all satisfied (dry-run, exit 0) |
| 2 | `cp .env.example .env` | done; drift documented in I6 |
| 3 | `python scripts/init_db.py` | **exit 0** — all tables created, idempotent |
| 4 | `python scripts/collect_data.py --repo-path . --max-commits 25` | **exit 0** — 21 commits, `data_origin: REAL_GIT_MINED` |
| 5 | `python scripts/train.py --output-dir <scratch>` | **exit 0** — 5 members trained; **redirected to a scratch directory** so the shipped artifacts every published number was measured on were not overwritten (your no-retraining decision). This is the run that exposed C1. |
| 6 | `uvicorn` + `curl` | `/health` → 200 healthy, DB connected. `POST /api/v1/select` → `SAFE_FULL_SUITE`, `abstained: true`, reason names the threshold it failed (`0.0207 < tau_conf 0.1000`) — consistent with the measured 97.81% abstention rate. Error handling: 422 on out-of-range `budget_ratio`, 422 on empty `changed_files` with an explanatory message, 422 on malformed JSON, 404 on unknown route, **401 on an unsigned webhook**. |
| 7 | `streamlit run dashboard/app.py` | **HTTP 200**, `_stcore/health` 200 |
| 8 | `pytest tests/ -v --cov=src/conftest` | **693 passed** at the time of the coverage run, **78% total coverage**; **700 passed** on the final run after the new tests landed |
| 9 | `python scripts/evaluate.py` | **exit 0** — 11 stages, 20 artifacts, all present, every producer resolves |

**Final suite:** `700 passed, 9 warnings in 171.24s` (`python -m pytest tests/ -q`, 2026-09-07).

---

## 6. What this means for the paper

The honest framing is unchanged by this audit, and slightly strengthened by it. The safety half of the claim holds: **100.0% failure recall, 0 escaped commits across 183 held-out commits**. The savings half does not: **3.11% of executions, 0.0% wall-clock, at 97.81% abstention**, and the alternative operating point trades to 32.85% reduction at 87.69% recall with 15 escapes.

What C1 adds is a *credible mechanism*. Before this audit the negative result read as "these features do not support confident test ranking." It now reads as "a five-round gradient-boosted ensemble does not support confident test ranking, on features whose largest group is entirely constant." Those are different claims, and the second one is both more defensible and more actionable.

## 7. Recommended next steps, in order

1. **Formalize inert mutation features** (F1). Keep the canonical serving schema, but report that one-file/one-line/no-message mutation inputs make all 12 diff columns constant; do not invent variation.
2. **Re-examine early stopping** (C1) — the validation metric, `stopping_rounds`, and whether `scale_pos_weight` interacts badly with it at a 5% positive rate.
3. **Retrain with row bagging active** (C2) so the epistemic spread means what the paper says it means.
4. Only then re-run the policy sweep and the G5 gate. Steps 1–3 change the ranker; re-tuning thresholds before them measures the wrong model.
5. Add coverage for `bugsinpy_adapter.py` (F4).

Nothing in steps 1–4 was performed here, by your instruction. Every number in this report is traceable to a file in `reports/` or to a command shown above.
