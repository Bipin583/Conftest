# ConfTest — Master Plan (Real Ground Truth Rebuild)

> **This is the single reference document for the rest of the project.**
> Last updated: 2026-09-03 · Status: Phase R — real dataset harvested, **every experiment re-run on it (C5)**; the result is negative and the paper still quotes the old numbers

---

## 0. TL;DR — Where we stand

**The engineering is ~4 months ahead of the original 6-month plan. The science has now been done
on real labels, and it came back negative: the method as tuned is indistinguishable from running the
whole suite. That is a publishable result and it is not the one the write-ups claim.**

| Area | State |
|---|---|
| Code | ✅ 70+ modules, **519/519 tests passing**, well-architected |
| Feature pipeline | ✅ 34 features (diff / AST / dependency-graph / history) |
| ML + calibration | ✅ LightGBM, isotonic + Platt + temperature, ECE, reliability diagrams |
| Abstention | ✅ Threshold policy, full-suite fallback, policy tuning |
| Ensemble + SHAP | ✅ Built (beyond original plan) |
| API + Dashboard | ✅ FastAPI (7 route modules), Streamlit (5 pages) |
| Docs | ✅ 27 docs, IEEE paper, KTU LaTeX report, viva deck |
| Mutation harness | ✅ 6 operator families, real pytest labels, **run end to end** (776 labelled rows from a 10-mutant smoke harvest) |
| Subject repos | ✅ **5 stage-3 survivors** (G0 met), 0 flaky, harvest **measured at 2.11 h** of suite time (9.97 h including one run whose 180 s timeout did not hold) — the 1.24 h estimate was low by 70% (see log 2026-09-03e) |
| Fabrication guards | ⚠️ AST label guard in CI (110 files, both rules) and every *known* invented fallback raises — but the guard passed for weeks while `benchmark.py` fed the abstention rule an `uncertainty` value computed from the labels. It tests for values that were **drawn**; a constant, a formula and a label-derived quantity all pass it (see log 2026-09-03g) |
| Feature builder | ✅ **C2 done** — test-ID resolver validated on 1138 real testcases, causal history, 32 features |
| **Real dataset** | ✅ **561,711 real rows** (1,213 mutants x 5 repos, 27,444 measured failures) — **G0–G3 all met**. Failure rate **6.817% over the 402,559 rows from the 889 mutants some test detected**, 4.886% over all rows; every repo in the 1–15% band, though validators clears the floor by only 0.185pp and 324 mutants (26.7%) were killed by nothing (see logs 2026-09-03e, 2026-09-03f) |
| Published numbers | ⚠️ **C5 done — every report now traces to a measured run, and the answer is negative.** At the shipped operating point ConfTest abstains on 97.8% of commits, saves **0.0%** of wall time, misses nothing, and is indistinguishable from the full suite (p = 1.00000, d = 0.0000). The $64,992 modelled saving was an artifact of a fabricated `uncertainty` column derived from the labels. The paper, KTU report and viva deck still quote the pre-C5 numbers (see log 2026-09-03g) |
| LLM review layer | ⬜ Does not exist (optional, deprioritized) |

**The dataset is real and every experiment has been re-run on it. What remains is not measurement
but a tuning problem the measurement exposed: the abstention policy has no operating point that
both holds a recall floor out of sample and saves meaningful time (G5, see log 2026-09-03g), and
the write-ups still describe the pre-C5 results.**

---

## 1. The problem, precisely

### 1.1 Our failure labels are coin flips

`src/benchmark/real_repo_miner.py:103-111` — the file whose entire purpose is *real* data:

```python
# Ground-truth failure label simulation based on real change impact
label = 1 if np.random.rand() < 0.70 else 0
label = 1 if np.random.rand() < 0.40 else 0
label = 1 if np.random.rand() < (hist_fail * 0.1) else 0
```

`src/conftest/repository/synthetic_generator.py:162-167` — same thing:

```python
fail_prob = failure_rate if is_affected else 0.005
is_fail = self.rng.random() < fail_prob
```

### 1.2 Five consequences

1. **The model can only learn the random rule that produced the data.** A `p=0.70` coin flip is ~30% irreducibly unpredictable. No feature can beat that ceiling.

2. **Every method scores identically** because the labels are mostly noise. From `reports/baseline_comparison.csv`:

   | Method | Failure Recall |
   |---|---|
   | Random-k | 20.0% |
   | Changed-File | 0.0% |
   | Static AST call-graph | 20.0% |
   | Historical frequency | 20.0% |
   | Uncalibrated ML | 20.0% |
   | Calibrated ML | 20.0% |
   | **ConfTest** | **40.0%** |

   Six structurally different methods landing on *exactly* 20% is the signature of unlearnable labels.

3. **The test set contains 5 positive samples.** ConfTest's 40% vs 20% is **2 failures caught vs 1**. That is one coin landing differently, not a result.

4. **Real mining produced 0 and 1 commits** (`data/raw/mined_local_*.json`) — and the miner *silently substitutes* `np.random.exponential()` values while stamping the output `"data_origin": "REAL_GIT_MINED"`. This mislabeling is the single most dangerous artifact in the repo.

5. **The headline result contradicts the thesis.** ConfTest misses **60% of failures**. The paper claims calibration + abstention deliver safety. The table shows the opposite.

### 1.3 Two secondary defects

- **Changed-File baseline selects zero tests** (100% reduction / 0% recall). That is a broken implementation, not a baseline.
- **Calibration model selection is noise-chasing.** We pick `temperature_scaling` on an ECE gap of 0.0066 while its MCE (0.894) is **4x worse than uncalibrated** (0.222), on 100 samples.

### 1.4 Why this must be fixed first

Our stated contribution is *confidence calibration + abstention*. **Calibration is only meaningful against real failures.** On coin-flip labels, a perfectly calibrated model is calibrated to noise. Everything downstream — ECE, reliability diagrams, risk-coverage curves, the economic model, the paper — inherits the flaw.

An examiner who greps for `random` next to the words "ground truth" finds this in five minutes.

---

## 2. The fix — real labels from real test runs

### 2.1 Core insight

`SafeTestExecutor` **already** runs real suites and parses real per-test outcomes from JUnit XML (verified working — `src/conftest/tests/executor.py:89-231`). We do not need new infrastructure. We need to point existing machinery at real code changes.

```
real code change  ->  run REAL test suite  ->  parse JUnit XML  ->  REAL label
```

### 2.2 Where real labels come from: two sources

| Source | Diffs | Labels | Volume | Role |
|---|---|---|---|---|
| **Mutation harness** (build it) | Synthetic (1-line, real code) | **Real** (actual pytest) | High (~100k pairs) | Primary: train + eval |
| **BugsInPy** (adopt it) | **Real** (developer commits) | **Real** (actual failing tests) | Low (~500 bugs) | External validity |

Report **both**. Mutants give statistical power; BugsInPy gives real-world validity. This two-tier design is the strongest honest story available to us.

### 2.3 Why mutation testing is scientifically defensible

- Mutation-based evaluation of regression test selection is **established methodology** in SE research.
- Mutants are *real code changes* that produce *real test failures* through *real execution*. Zero simulation in the label path.
- Mutants guarantee failures, which fixes our 5-positive-sample problem.
- Fully offline, no CI credentials, no network, runs on the RTX 4050 laptop overnight.

### 2.4 Why we write our own mutator (not mutmut / cosmic-ray)

1. We need the **exact file and line** changed to build diff + dependency features. Off-the-shelf tools hide this.
2. They are built for mutation *scoring* (one aggregate number), not for emitting **per-test outcomes per mutant**, which is exactly our label matrix.
3. No dependency conflicts, full control over operators.
4. It is a genuine "we built this" contribution for the report.

### 2.5 Honest framing for the paper

- **Primary dataset:** mutation-induced faults across N real repositories — real dependency structure, real test execution, real outcomes, controlled fault injection.
- **External validation:** BugsInPy real developer-introduced bugs — confirms transfer to naturally occurring faults.
- **Retired:** the synthetic generator is demoted to a *unit-test fixture* and explicitly labeled as such. It is genuinely useful for that.
- **Threats to validity section** states plainly: mutants are single-line and may not represent all real fault distributions; this is why BugsInPy is included.

---

## 3. Architecture of the new data pipeline

```
                ┌──────────────────────────────────┐
                │  Component 0: Repo Screener      │
                │  find repos with clean, fast,    │
                │  deterministic suites            │
                └───────────────┬──────────────────┘
                                │  vetted repo list
                ┌───────────────▼──────────────────┐
                │  Component 1: Mutation Harness   │
                │                                  │
                │  1. baseline run x3 (flakiness)  │
                │  2. generate mutants (AST)       │
                │  3. for each mutant:             │
                │       apply -> run FULL suite    │
                │       -> parse JUnit -> revert   │
                │  4. label = failed_now &&        │
                │             passed_on_baseline   │
                └───────────────┬──────────────────┘
                                │  real (mutant, test, outcome)
                ┌───────────────▼──────────────────┐
                │  Component 2: Feature Builder    │
                │  reuse FeatureExtractionPipeline │
                │  + temporal telemetry accumulator│
                └───────────────┬──────────────────┘
                                │  features.csv (real labels)
                ┌───────────────▼──────────────────┐
                │  Component 3: BugsInPy Adapter   │
                │  real bugs -> held-out test set  │
                └───────────────┬──────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │  Component 4: Cleanup            │
                │  kill synthetic fallbacks,       │
                │  fix baselines, fix calib. choice│
                └───────────────┬──────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │  Component 5: Re-run Everything  │
                │  (all existing code, real data)  │
                └──────────────────────────────────┘
```

**Key design decision: we run the FULL suite per mutant.** Ground truth requires knowing the outcome of *every* test, not just selected ones. This is the cost driver and it is non-negotiable — a partial label matrix cannot evaluate recall.

---

## 4. Build order — what we are building, in sequence

### Component 0 — Repo Screener  `scripts/screen_repos.py`

**Purpose:** mutation harvesting only works on suites that are fast and deterministic. Screen before committing compute.

**Selection criteria (hard gates):**
- Pure Python (no C extensions to compile)
- `pip install -e .` succeeds in a clean venv
- Full suite runs in **< 90 seconds**
- **50–600 tests** (enough signal, bounded cost)
- Clear `src/` (or package/) vs `tests/` separation — required by the dependency-graph features
- **No network, no database, no filesystem-heavy fixtures** in tests
- Suite is **green on a clean checkout** (any pre-failing test is excluded, not the repo)
- **Deterministic:** 3 consecutive runs give identical pass/fail sets

**Candidate pool to screen** (pure-Python, well-tested, right size):
`cachetools`, `tabulate`, `python-slugify`, `inflection`, `validators`, `schedule`, `parse`, `humanize`, `deepdiff`, `jsonschema`, `toml`, `pyparsing`, `arrow`, `chardet`

**Target:** **5 repos** passing all gates. Screen ~12 to get 5.

**Output:** `data/repos/screening_report.json` — per-candidate gate results + timings.

---

### Component 1 — Mutation Harness  `src/conftest/groundtruth/mutation_harness.py`

This is the critical path. Everything else waits on it.

#### 1a. AST Mutation Operators  `mutators.py`

| Operator | Example | Why |
|---|---|---|
| Arithmetic | `a + b` -> `a - b` | classic, high test-kill rate |
| Comparison | `<` -> `<=`, `==` -> `!=` | off-by-one / boundary bugs |
| Boolean | `and` -> `or`, `not x` -> `x` | logic bugs |
| Constant | `0` -> `1`, `True` -> `False` | magic-value bugs |
| Return | `return x` -> `return None` | contract violations |
| Conditional | `if c:` -> `if True:` / `if False:` | dead-branch bugs |

Implementation: `ast.NodeTransformer` subclass per operator. Mutate **exactly one node per mutant** — we must attribute the failure to one line.

**Only mutate source files, never test files.** Mutating a test changes the oracle, not the code under test.

#### 1b. Harness protocol  `mutation_harness.py`

```
STEP 1 — BASELINE & FLAKINESS SCREEN
  run full suite 3x on clean checkout
  baseline_passing = tests that PASSED all 3 times
  flaky_or_broken  = everything else  -> EXCLUDED from dataset
  record per-test mean duration (feeds hist_avg_duration + time-reduction metric)

STEP 2 — MUTANT GENERATION
  for each source file:
    parse AST -> enumerate mutatable nodes -> apply operator -> emit candidate
  sample N mutants (stratified across files & operators) for coverage, not just the first N

STEP 3 — HARVEST (per mutant)
  write mutated file to disk (keep original in memory)
  run FULL suite with timeout (mutations can cause infinite loops)
  parse JUnit XML -> {test_id: status}
  ALWAYS restore original file (try/finally — a crash must never leave the repo dirty)

STEP 4 — LABELLING
  for each test in baseline_passing:
      label_failed = 1 if status in (FAILED, ERROR) else 0
  DISCARD mutant if:
      - it killed EVERYTHING (broke collection/import — not a real fault)
      - the suite timed out entirely
```

**Robustness requirements (non-negotiable):**
- `try/finally` restore of every mutated file — a killed process must not corrupt the checked-out repo
- Per-mutant wall-clock cap; on timeout mark and skip
- **Resumable checkpointing** every mutant (we will interrupt this run many times)
- Deterministic seed for mutant sampling, recorded in output metadata

**Cost model:** 5 repos x 250 mutants x ~40s full suite ≈ **14 hours total**, parallelizable across 4 team laptops -> **~3.5h each, one overnight run.**

**Output:** `data/groundtruth/<repo>/mutants.jsonl` — one record per mutant:
```json
{"mutant_id": "...", "repo": "...", "file": "src/x.py", "line": 42,
 "operator": "comparison_lt_to_le", "original": "a < b", "mutated": "a <= b",
 "outcomes": {"tests/test_x.py::test_foo": "FAILED"},
 "n_killed": 3, "suite_duration": 41.2}
```

---

### Component 2 — Feature Builder  `scripts/build_real_dataset.py`

Converts mutant records into the training matrix. **Reuses existing code — no new feature logic.**

```python
FeatureExtractionPipeline(repo_root).extract_features_for_pair(
    test_path=..., test_function=...,
    changed_files=[{"file_path": mutant.file,
                    "lines_added": 1, "lines_deleted": 1,
                    "change_type": "modified", "is_test": False}],
    commit_message=f"mutate {operator} at {file}:{line}",
    commit_timestamp=synthetic_ordered_timestamp,
)
```

**Temporal telemetry accumulator (important):** the `hist_*` features need prior-run history. We impose an ordering over mutants (mutant 1, 2, 3, …) and accumulate **real** telemetry as we walk it: which tests actually failed on earlier mutants, real durations, real flake signals. The timeline is imposed; **every value in it is measured.** Document exactly this in the paper — it is honest and defensible.

**Splits:** strict chronological over the imposed mutant ordering, **grouped by repo** so no repo appears in both train and test for the cross-repo evaluation. 70/15/15.

**Output:** `data/processed/real_features.csv` + `data/splits_real/{train,val,test}.csv` + metadata JSON recording seed, repo list, commit SHA of each repo, tool versions.

---

### Component 3 — BugsInPy Adapter  `src/conftest/groundtruth/bugsinpy_adapter.py`  *(P1 — after Component 2 lands)*

**What it is:** 493 real bugs from 17 real Python projects. Each provides buggy commit, fixed commit, and the **actual failing tests**.

**Use:** held-out external-validity test set. Never train on it.

**Practical note:** each project needs its own environment. Budget effort for **3–5 projects**, not all 17. Even 3 is a strong external validity claim.

**Output:** `data/groundtruth/bugsinpy/real_bugs.jsonl`, same schema as mutants so downstream code is unchanged.

---

### Component 4 — Cleanup  *(can run in parallel with Component 1)*

| # | Fix | File |
|---|---|---|
| C4.1 | **Delete every synthetic fallback.** Mining must fail loudly. Nothing may be stamped `REAL_GIT_MINED` unless every value was measured. | `src/benchmark/real_repo_miner.py:38,61,70+,103-111` |
| C4.2 | **CI grep gate** — fail the build if `random` appears in a label-producing path | `.github/workflows/` |
| C4.3 | **Fix Changed-File baseline** — currently selects zero tests | `src/conftest/models/baselines/heuristics.py` |
| C4.4 | **Fix calibration selection** — choose on ECE **+ MCE + Brier** jointly with bootstrap CIs, not ECE alone | `src/conftest/models/calibration.py` |
| C4.5 | **Bootstrap CIs on every reported metric** — essential at our sample sizes; a metric without an interval is not a result | `src/conftest/evaluation/statistics.py` |
| C4.6 | **Relabel synthetic generator** as a unit-test fixture in docstring + docs | `src/conftest/repository/synthetic_generator.py` |

---

### Component 5 — Re-run Everything  *(done — see log 2026-09-03g)*

All of this code already exists and is tested. Point it at real data and execute:

`train_model.py` -> `calibrate_model.py` -> `tune_policy.py` -> `train_ensemble.py` -> `run_ablation_study.py` -> `run_cross_repo_eval.py` -> `run_statistical_tests.py` -> `run_flakiness_test.py` -> `run_latency_benchmark.py` -> `run_economic_analysis.py` -> `generate_explanations.py`

**Then rewrite:** `reports/*`, the IEEE paper results section, the KTU report, the viva deck.

`reports/*` is done. The three write-ups are not, and they are the last place in the repository
where the fabricated-era numbers are still asserted (tracked as C6). Running the chain also turned
up two things the chain itself was not looking for: the benchmark was inventing the model outputs
that baselines 6-8 rank by, and baseline 8 was being evaluated at thresholds the tuner never chose.
Re-running is not a formality; it is what made both visible.

---

## 5. Reframe the headline claim

**Current framing (weak):** "we reduce tests by 60%" — and recall falls wherever it falls (40%). No CI owner accepts a tool that misses 60% of failures.

**New framing (what the thesis actually argues):**

> **At a target failure recall of ≥95%, ConfTest reduces test execution by X%.**

Fix safety first, then report savings. This matches our real contribution — calibration + abstention exist *to hold a recall guarantee* — and it is the number a CI owner actually buys.

**Primary metrics table going forward:**

| Metric | Definition |
|---|---|
| **Reduction @ 95% recall** | headline claim |
| **Reduction @ 99% recall** | safety-critical operating point |
| **ECE / MCE / Brier** (+ bootstrap CI) | calibration quality |
| **Risk–coverage AUC** | selective-prediction quality |
| **Abstention rate** | cost of the safety net |
| **Missed-failure rate** | what we must drive to ~0 |

---

## 6. Acceptance gates — definition of done

Do not proceed past a gate until it is green.

| Gate | Criterion |
|---|---|
| **G0** | 5 repos pass all screening criteria; 3 baseline runs identical |
| **G1** | ≥ 5,000 labeled (mutant, test) pairs per repo; failure rate in **1–15%**, measured over rows from mutants **at least one test detected** — a mutant no test kills carries label 0 in every row it produces and so states no selection target. Both rates are always published (`failure_rate_detected_mutants`, `failure_rate_all_mutants`) and the undetected rows stay in the dataset flagged `mutant_detected=0`, so the exclusion is auditable and reversible |
| **G2** | No value on the label path is drawn or imputed — enforced by `scripts/check_no_fabricated_labels.py` (AST, not grep: mutant *sampling* is seeded, so a literal `grep -rn "random"` can never return zero) |
| **G3** | Every mutated file restored — `git status` clean in all harvested repos |
| **G4** | ML beats random baseline with **bootstrap 95% CI excluding zero** — ✅ **met.** Per-commit mean failure recall: uncalibrated ML **69.26% [62.60, 75.90]** vs Random-k **20.01% [16.73, 23.02]** over the 135 commits with a defined recall, Wilcoxon `p < 0.00001`, Cliff's delta **+0.6564** (large). `reports/g4_ml_vs_baselines.json` (see log 2026-09-03g) |
| **G5** | Reduction @ 95% recall reported with CI, on a repo **never seen in training** — ❌ **measured, not met.** On the unseen commit split the validation-selected point gives 32.85% [26.50, 39.18] reduction at **87.69% [75.77, 95.81]** recall: the floor is cleared in sample by 1.02pp and missed out of sample by 7.31pp, and is not met at the interval's lower bound. Leave-one-repo-out is further off still (macro recall@budget 51.57%, `tabulate` ROC-AUC 0.4747). `scripts/check_g5_recall_floor.py` -> `reports/g5_recall_floor.json` (see log 2026-09-03g) |
| **G6** | BugsInPy external validation on ≥ 3 projects |

**If G4 fails** — ML genuinely does not beat the heuristic baseline — that is a **publishable negative result**, and the original plan already anticipated it ("report as ML didn't beat baselines — valid research result"). We report it honestly with CIs. Far stronger than a fabricated win.

---

## 7. Team allocation (4 members, 3 weeks to real data)

| | Member 1 — Data | Member 2 — ML | Member 3 — Harness | Member 4 — Eval |
|---|---|---|---|---|
| **W1** | C0 screener, screen 12 repos | C4.4 calibration fix, C4.5 bootstrap CIs | **C1a mutators** (AST operators) | C4.3 baseline fix, C4.1/C4.2 purge + CI gate |
| **W2** | Run harvest on repos 1–3 | Retrain on first real batch | **C1b harness** (protocol + checkpointing) | Rebuild metrics @ fixed-recall framing |
| **W3** | Run harvest on repos 4–5, C2 dataset build | Recalibrate, retune policy, ensemble | C3 BugsInPy adapter | C5 re-run all experiments, redraw figures |

**Cross-training rule stays:** every member must be able to run Component 1. It is the critical path.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mutants kill almost nothing (weak suites) | Med | Screener requires real coverage; add more operators |
| Mutants kill *everything* (import breakage) | Med | Discard mutants failing >80% of suite — not a realistic fault |
| Harvest too slow | Med | Cap suite at 90s in screening; parallelize across 4 laptops; checkpoint + resume |
| Flaky tests pollute labels | High | 3x baseline screen, exclude non-deterministic tests, keep `hist_flaky_score` as feature |
| Repo env setup hell | Med | Screener enforces `pip install -e .` success **before** compute is spent |
| BugsInPy setup too heavy | High | It is P1, not P0. 3 projects suffice. Mutants alone clear G1–G5 |
| Real recall is poor | Med | **This is a finding, not a failure.** Report with CIs (see G4) |

---

## 9. What to tell the teacher

> Ma'am, our system is built and fully tested — 115 passing tests, ML with calibration and selective abstention, API, dashboard, complete documentation.
>
> During our internal audit we found that our evaluation data was **simulated**, including the failure labels. Our contribution is confidence calibration, and **calibration is only meaningful against real test failures** — so we are rebuilding the evaluation on real ground truth before writing the results.
>
> We do this by **mutation-based fault injection**: we inject real single-line faults into five real open-source Python repositories and **run their real test suites**, recording actual pass/fail per test. No simulated labels anywhere. We additionally validate on **BugsInPy** — 493 real developer-introduced bugs.
>
> The system code does not change — we already built the evaluation harness. We are replacing fabricated inputs with measured ones, then re-running.
>
> We also reframed our headline claim to the form CI owners actually need: **at ≥95% failure recall, how much test execution do we save?**
>
> Finding this ourselves, and fixing it rather than shipping it, is the most valuable engineering lesson of the project.

---

## 10. Current sprint — build now

- [x] **C1a** `src/conftest/groundtruth/mutators.py` — 6 AST operator families
- [x] **C1a** `tests/unit/test_mutators.py` — 31 tests, all passing
- [x] **C1b** `src/conftest/groundtruth/mutation_harness.py` — baseline screen, harvest loop, labelling, checkpointing
- [x] **C1b** `tests/unit/test_mutation_harness.py` — 33 tests, all passing
- [x] **C0** `scripts/screen_repos.py` — two-stage screener + JSON report
- [x] **C0** `tests/unit/test_repo_screener.py` — 21 tests, one per screener bug found
- [x] **C0** screen 20 candidates -> **7 accepted** (target was 5)
- [x] **C4.1** purge synthetic fallbacks from `real_repo_miner.py` — all randomness gone
- [x] **C4.2** CI gate against fabricated labels — `scripts/check_no_fabricated_labels.py` (AST, not grep)
- [x] **C4.2** `tests/unit/test_label_guard.py` — 14 tests, all passing
- [x] **C4.1b** the reporting layer had its own fabricator: `GET /api/v1/calibration` served hand-typed ECE/MCE/Brier/temperature constants when no report existed, asserting a 25% ECE gain that is `[-0.0112, +0.0110]`. Now 503, and every served metric carries its interval (see log 2026-09-03c)
- [x] **C4.3** fix Changed-File baseline — stem matching + mandatory `changed_file_path`
- [x] **C4.7** stop faking ETR — measure it from real durations
- [x] **C4.8** the dashboard front page is measured-or-absent: every KPI is read from a named artifact by `src/conftest/evaluation/headline.py`, an absent artifact renders as `not measured` with the script that produces it, ETR and TRR are no longer conflated, and a difference whose interval spans zero is reported as no gain (see log 2026-09-03d)
- [x] **C2** `scripts/build_real_dataset.py` — resolver validated against 1138 real testcases
- [x] **C2** `scripts/harvest_mutations.py` — the CLI driver that was missing
- [x] **C2** `tests/unit/test_build_real_dataset.py` — 40 tests, all passing
- [x] **C0.2** stage 3 structural screen — dependency-feature variance (see log 2026-09-02c)
- [x] **C0.2** cost gate recalibrated from test count to measured harvest hours
- [x] **C0.3** re-screen the widened pool to restore G0 — **5 stage-3 survivors** (see log 2026-09-02d)
- [x] **C0.4** every subject suite runs under its own screened venv, enforced by an import-provenance guard (see log 2026-09-02d)
- [x] **C0.5** install declared functional extras — pyjwt ran 221 of 369 tests without them (see log 2026-09-02d)
- [x] **C0.6** every harvest writes `harvest_summary.json`; the dataset builder refuses labels whose import provenance is missing, unverified, or points outside the checkout (see log 2026-09-02e)
- [x] **C0.7** the checkout is treated as an instrument: a `pending_mutation.json` journal, a pristine-checkout assertion with `--restore-checkout` repair (`git checkout HEAD -- .`, never `git clean`), sample-membership verification on resume with off-sample records quarantined, per-run drift detection, and a `harvest.lock` that allows one harvest per output directory and is taken *before* any repair (see log 2026-09-03d)
- [x] **C0.8** the suite timeout is a containment boundary, and it was not one: `subprocess.run(stdout=PIPE, timeout=...)` kills only the direct child and then drains the pipe **with no timeout**, so one sqlparse mutant held a mutated checkout live for 7h51m under a 180 s ceiling and let the suite truncate one of its own tracked fixtures. The executor now captures to files with `stdin=DEVNULL`, kills the whole process tree, bounds the post-kill wait, and stamps `timeout_enforced` on every record; `timeout_broken` outranks `timed_out`; and the dataset builder refuses any `harvested` record carrying drift or an unenforced ceiling, listing exclusions by id and reason and raising past 2% (see log 2026-09-03e)
- [x] **G1** first real dataset, gates verified — `data/processed/real_features.csv`, 561,711 rows, 1,213 mutants, 5 repos, 0 unresolved test IDs, provenance verified x5, `git status` clean in all five checkouts. Every repo clears 5,000 pairs by an order of magnitude, and every repo is in the 1–15% band on the rate the gate reads: tabulate 11.723%, pathspec 12.309%, sqlparse 10.790%, pyjwt 6.228%, **validators 1.185%** — which clears the 1% floor by 0.185pp and is the one number in this row worth distrusting. The gate reads the rate over detected mutants; 324 of 1,213 mutants (26.7%) were killed by no test at all, 91 of them in validators (see logs 2026-09-03e, 2026-09-03f)
- [x] **C4.4** calibration selection must not pick a method on ECE alone — `src/conftest/models/calibrator_selection.py`, chosen on a validation half-split that is cut along **mutant** boundaries, on ECE + MCE + Brier jointly, each as a **paired bootstrap difference** against the uncalibrated model; a gain whose interval spans zero is not a gain (see log 2026-09-03b)
- [x] **C4.5** bootstrap confidence intervals on all headline numbers: the resampling unit is the **commit**, not the test row; every one of the 5 metrics x 8 strategies carries a 95% interval; the headline comparison is reported as a *paired* difference against the ConfTest row (see log 2026-09-03a)
- [x] **C4.6** relabel `synthetic_generator.py` as smoke-test-only: `acknowledge_synthetic=True` is required to construct it, every emitted commit and test run carries `data_origin=SYNTHETIC_FABRICATED_LABELS`, and the docstring quotes the coin flip (see log 2026-09-02f)
- [ ] **C3** BugsInPy adapter *(P1)*
- [x] **C5** re-run every experiment on real data — the full chain twice: `calibrate_model` -> `tune_policy` -> `train_baseline --bootstraps 2000` -> `run_ablation_study` -> `run_cross_repo_eval` -> `run_statistical_tests` -> `run_flakiness_test` -> `run_latency_benchmark` -> `run_economic_analysis` -> `generate_explanations` -> `run_continuous_learning` -> `uncertainty_eval`, then the reporting chain again after the re-run exposed the fabricator below (see log 2026-09-03g)
- [x] **C5.1** the third fabricator, and the worst one: `benchmark.py` invented the three columns baselines 6-8 rank by — `raw_score` as a hand-weighted `0.6*dep_is_direct_import + 0.4*hist_lifetime_failure_rate`, `calibrated_confidence` as the constant `0.85`, and `uncertainty` as `0.08 if len(failing_test_ids) <= 1 else 0.22`, i.e. **the abstention rule was thresholding a quantity derived from the labels**. It now raises and names the producer; `scripts/train_baseline.py` scores the split with the real ensemble and calibrator and records the ranges it produced. The AST label guard passed all 110 files throughout — no `random` call, real dataset on disk — which is the limit of what that guard can see (see log 2026-09-03g)
- [x] **C5.2** baseline 8 is the proposed method, so it is evaluated at the tuned operating point: `ConfTestSelectiveSelector` reads `tau_abstain`/`tau_conf` from `models/policy_config.json` and raises if it is absent, instead of defaulting to the `0.15`/`0.70` literals it shipped with while the tuned policy was `0.02`/`0.10`. The policy report and the headline row now agree exactly (see log 2026-09-03g)
- [x] **C5.3** `scripts/tune_policy.py` takes `--objective {zero_escape,recall_floor}` with `--recall-floor`, measures **both** frontiers on every run, publishes the whole 36-point sweep, replaces `max(1, n)` denominators with NaN, and raises instead of silently writing the invented `(0.015, 0.50)` fallback. The default stays `zero_escape`, and replaying the old selection rule over the 36 measured points picks the same pair, so the refactor moves nothing on its own; the committed `tau_abstain` still moves 0.03 -> 0.02, because the 0.03 came from a pre-provenance run against data that is gone (see log 2026-09-03g)
- [x] **C5.4** `scripts/check_g5_recall_floor.py` + `reports/g5_recall_floor.json` — the G5 frontier on a grid fine enough to resolve the recall cliff, selected on validation, scored on the unseen split with 2000 commit-resampled bootstraps, with the post-hoc test-split sweep published as an explicitly-labelled diagnostic rather than a menu to select from (see log 2026-09-03g)
- [ ] **C6** rewrite the IEEE paper results section, the KTU report and the viva deck against the C5 numbers — currently the largest inconsistency in the repository

**Definition of done for this sprint:** ✅ `data/processed/real_features.csv` exists, every label traceable to a real pytest run, G0–G3 green. **G4 is green** (ML 69.26% [62.60, 75.90] vs random 20.01% [16.73, 23.02] per-commit recall, `d = +0.6564`); **G5 is measured and not met**; G6 is untouched pending C3.

---

## 11. Build log — findings from implementation

### 2026-09-03g · C5 · the headline row was measuring a system this repository does not contain

Every experiment has now been re-run on the real dataset. The chain is
`calibrate_model` -> `tune_policy` -> `train_baseline` -> `run_ablation_study` ->
`run_cross_repo_eval` -> `run_statistical_tests` -> `run_flakiness_test` ->
`run_latency_benchmark` -> `run_economic_analysis` -> `generate_explanations` ->
`run_continuous_learning` -> `uncertainty_eval`, and it was run twice: once on the
real labels, and then again after the re-run exposed something worse than a wrong
number.

**The policy report and the headline row described two different systems.** The
tuner said the shipped operating point abstains on 97.81% of commits, reduces the
suite by 3.11%, and lets nothing escape. The published comparison said ConfTest cut
wall time by 19.7%, held 99.6% recall, and let 14 commits escape. Both claimed to
be the same configuration on the same split. Only one of them was reading the
model.

`src/conftest/evaluation/benchmark.py` invented the three columns that baselines
6, 7 and 8 rank by, whenever the dataset did not carry them -- and
`data/splits/test.csv` has 46 columns, none of them these:

- `raw_score` became a hand-weighted `0.6 * dep_is_direct_import + 0.4 *
  hist_lifetime_failure_rate`, a formula that appears nowhere in the model;
- `calibrated_confidence` became the constant `0.85`, so every test in the corpus
  was equally and highly confident;
- `uncertainty` became `0.08` or `0.22`, **chosen by `len(failing_test_ids)`**.

The third one is the serious one. Uncertainty is the quantity the abstention rule
thresholds, and it was being derived from the count of tests that actually fail --
the label. The method's central contribution was being handed the answer and then
scored on how well it used it. The 19.7% time reduction, the 99.6% recall, the
$64,992 of modelled annual saving and the `p = 0.00018` significance against the
full suite were all measurements of that arrangement, not of this system.

Alongside it, `ConfTestSelectiveSelector` carried `abstention_threshold=0.15,
min_confidence_threshold=0.70` as constructor defaults while
`models/policy_config.json` -- the file `scripts/tune_policy.py` writes -- held
`tau_abstain=0.02, tau_conf=0.10`. Baseline 8 *is* the proposed method, so it has
to be evaluated at the operating point the project ships, not at literals that
happen to sit in a signature.

**The fix in both places is to refuse.** The benchmark now raises when the score
columns are absent and names the script that produces them; it no longer has a
code path that can invent them. `scripts/train_baseline.py` grew
`attach_model_scores()`, which loads `models/ensembles/5_seed_lgbm` and
`models/calibrator.joblib`, scores the split, and records what it did:
`raw_score` spans [0.040121, 0.281224], epistemic std spans [0.000489, 0.080607].
The selector reads its thresholds from `models/policy_config.json` and raises
`FileNotFoundError` if the file is missing -- an untuned operating point is not a
default -- and raises rather than filling in a missing `uncertainty` or
`calibrated_confidence` per test.

**What the corrected comparison says.** `reports/baseline_comparison.csv`, on 183
commits of the unseen test split, 25% budget, with 2000 commit-resampled
bootstraps:

| Strategy | TRR | ETR | FR | Abstention | Escaped commits |
|---|---|---|---|---|---|
| 1. Full Test Suite | 0.0% | -0.0% | 100.0% | 0.0% | 0 |
| 2. Random-k | 75.1% | 78.7% | 25.2% | 0.0% | 133 |
| 3. Changed-File | 83.6% | 86.1% | 42.3% | 0.0% | 69 |
| 4. Static AST call graph | 75.6% | 68.2% | 38.6% | 0.0% | 94 |
| 5. Historical failure frequency | 75.1% | 70.0% | 48.4% | 0.0% | 99 |
| 6. Uncalibrated ML | 75.1% | 27.7% | 51.8% | 0.0% | 69 |
| 7. Calibrated ML, no abstention | 75.1% | 27.7% | 51.8% | 0.0% | 69 |
| **8. ConfTest** | **3.1% [0.8, 6.2]** | **0.0% [0.0, 0.1]** | **100.0%** | **97.8% [95.6, 99.5]** | **0** |

The honest result is negative. At the operating point this project ships, ConfTest
abstains on 97.8% of commits, saves 0.0% of wall time, misses nothing, and is
**statistically indistinguishable from running the whole suite**: Wilcoxon
`p = 1.00000`, Cliff's delta `0.0000`, on the 135 of 183 commits where any recall
is defined. Against every other baseline it wins on recall with a large effect
(`d` from 0.5111 to 0.9852, all `p < 0.001`), which is the same statement seen from
the other side -- it wins on recall because it runs everything. The economic model
now prices it at **$0** of annual saving with a breakeven of 0 escaped bugs, where
before it claimed $64,992. Baselines 6 and 7 are priced at $91,384 each on 69
escaped commits out of 183.

**Baselines 6 and 7 are identical, and that is the correct result.** Temperature
scaling is a monotone transform of the logit, so it cannot reorder anything; a
top-k selection by calibrated probability is the same set as a top-k selection by
raw probability. Their rows agreeing to the last digit is a proof that the
implementation is faithful, not a copy-paste. It also sharpens the thesis: at a
fixed budget, calibration buys nothing through ranking. Whatever calibration is
worth here, it is worth it *only* through the abstention decision.

**The label guard could not see any of this.** `check_no_fabricated_labels.py`
passes all 110 files, and it was right to: there is no `random` call in
`benchmark.py`, and the script does read a real dataset from disk. The guard tests
for values that were *drawn*. These were a hand-weighted formula, a constant, and a
function of the labels -- fabricated in the sense that matters, invisible to both
rules. The lesson for the next guard is that the property worth enforcing is not
"nothing is random" but "no published quantity lacks a named producer", which is
what `MissingArtifact` already does on the reporting path and what these three
`row.get(..., default)` calls quietly opted out of.

**G4, answered directly.** The gate asks whether the model beats random with an
interval excluding zero, and the shipped significance report could not answer it: it
compares every strategy against ConfTest, which wins on recall by running everything.
Re-run with `--reference "6. Uncalibrated ML (LightGBM)"` into
`reports/g4_ml_vs_baselines.json`, the answer is unambiguous. Per-commit mean failure
recall over the 135 commits where recall is defined: **ML 69.26% [62.60, 75.90] vs
Random-k 20.01% [16.73, 23.02]**, Wilcoxon `p < 0.00001`, Cliff's delta **+0.6564**,
large. (These are macro means over commits; the 51.8% and 25.2% in the comparison
table are pooled over rows. Both are correct and the unit is stated in each report.)
**G4 is met.** The same table carries a result the paper should not omit: against
Changed-File selection the model's edge is `d = +0.0759` at `p = 0.01505` -- 69.26%
against 61.27% mean recall -- and Changed-File reduces the suite by 83.6% where the
model reduces it by 75.1%. On this dataset the cheap heuristic is competitive on
recall and better on reduction, and the model's advantage over it is statistically
present but small. Baseline 7 against baseline 6 comes back at `p = 1.00000`,
`d = 0.0000`, which is the monotone-calibration identity above confirmed from the
significance side.

**The tuner was answering a question G5 does not ask.** Its objective was "let
nothing escape, then take the largest reduction that survives", which on this
dataset is `tau_abstain = 0.02` -- the degenerate point above. G5 is stated as a
floor instead: hold failure recall at 95% and take the largest reduction. Those are
different constraints and the second admits points the first forbids, so a tuner
that only knows the first cannot report G5 at all. `scripts/tune_policy.py` now
takes `--objective {zero_escape,recall_floor}` and `--recall-floor`, measures both
frontiers on every run regardless of which one is selected, and writes the whole
sweep into `reports/policy_tuning_report.json` -- 36 points that were previously
collected into a local list and thrown away at the end of the loop. **The default
stays `zero_escape`**, so the refactor moves no operating point of its own accord:
replaying the pre-refactor selection rule over the 36 measured points picks
`tau_abstain = 0.0200, tau_conf = 0.10`, which is exactly what the new frontier code
selects.

The committed file did move, and that is worth saying out loud rather than leaving in
a diff. `models/policy_config.json` carried `tau_abstain = 0.03` from the original
bulk commit, and the report standing behind it has three keys -- `optimized_policy`,
`validation_evaluation`, `unseen_test_evaluation` -- with no `labels_measured`, no
`produced_by` and no grid. That 0.03 was tuned against data this repository no longer
contains, and nothing in the artifact says against what. C5 replaces it with 0.02
measured on the real splits under a named objective. The shipped policy therefore
abstains at two thirds of the uncertainty the paper drafts describe, and the ConfTest
row of `reports/baseline_comparison.csv` -- 3.11% reduction, 100.0% recall, 97.81%
abstention -- is that threshold's held-out score, not the old one's.

It also had an invented fallback. If no pair on the grid satisfied the constraint,
`best_config = (0.015, 0.50)` was written anyway -- a policy no evaluation had ever
scored, shipped as though it had been chosen. It now raises and says the objective
is unsatisfiable, which is a fact about the grid, not a reason for a default.
`max(1, denominator)` became NaN on an empty denominator for the same reason as
everywhere else: "nothing to find" is not "found nothing".

**30 of the tuner's 36 grid points were dead.** No calibrated confidence in this
dataset reaches 0.30 -- the raw scores top out at 0.281224 and temperature 0.7941
pushes them lower -- so every `tau_conf >= 0.30` abstains on every commit and buys
exactly 0.0% reduction. The 6x6 grid was a 6-point one-dimensional sweep over
`tau_abstain` with 30 wasted evaluations, and the whole interesting range sat inside
one jump of it: 0.030 gives 5.18% reduction at 99.98% recall, 0.050 gives 45.29% at
80.65%, and the recall cliff is somewhere in between where the grid cannot see it.
Both grids are now `--tau-abstain-grid` / `--tau-conf-grid` flags, defaulting to the
values the shipped policy was tuned on.

**G5, measured properly, is not met.** `scripts/check_g5_recall_floor.py` sweeps a
grid fine enough to resolve the cliff (0.030 to 0.050 in steps of 0.002), selects on
validation, and scores the selection on the unseen test split with 2000
commit-resampled bootstraps:

| | validation (selection) | unseen test split |
|---|---|---|
| tau_abstain / tau_conf | 0.044 / 0.10 | same |
| Test reduction | 32.85% | **32.85% [26.50, 39.18]** |
| Failure recall | 96.02% | **87.69% [75.77, 95.81]** |
| Abstention | 66.85% | 66.12% [59.02, 73.22] |
| Escaped commits | 11 | **15** |

The floor is cleared in sample by 1.02pp and missed out of sample by 7.31pp -- a
generalization gap of **8.33 points of recall** -- and it is not met at the
interval's lower bound either, which is the bar a gate should be read against. What
fails here is the selection, not the method: post hoc, the same grid does contain a
qualifying point on the test split (`tau_abstain = 0.038`, 26.56% reduction at
97.35% recall), because validation puts the cliff two grid steps to the right of
where the test split puts it. That post-hoc sweep is published in
`reports/g5_recall_floor.json` under an explicit `read_first` saying it is a
diagnostic and not a menu: selecting from it would make the recall an in-sample fit
and the gate unfalsifiable. G5's second clause is further away still -- it asks for
the number on a repo never seen in training, and leave-one-repo-out gives macro
recall@budget 51.57% with `tabulate` at ROC-AUC 0.4747, i.e. no better than chance
on one of five repos.

**A third of the feature vector is constant, by construction.** The ablation
reports `constant_features_in_train: 13` of 32, and 12 of those 13 are the entire
`diff_*` group. Every mutant is one line added and one line deleted in one source
file, with no commit message, so across all 561,711 rows `diff_lines_added = 1`,
`diff_total_churn = 2`, `diff_num_files_changed = 1` and the nine others never move.
The `diff_churn_only` ablation has `informative_feature_count: 0` -- it is a model
fitted on twelve constants -- and no `diff_*` feature appears anywhere in the global
SHAP ranking, which is led by `dep_shortest_path_depth` (0.27428) and
`hist_total_prior_runs` (0.25392). Two consequences worth stating in the paper
rather than leaving for a reviewer to find. The mutation-based ground truth cannot
evaluate the diff-feature group at all, so its contribution is unmeasured, not zero.
And the policy's out-of-distribution guard (`ood_file_limit=15`,
`ood_churn_limit=500`) can never fire on this dataset, because one file and two
lines is what every commit looks like; whatever safety it provides is untested here.

**The calibrator moved, and what it bought is narrower than the report suggests.**
Temperature scaling wins on the paired half-split at `T = 0.7941` (`moved_from_
identity: True`, NLL 0.192899 -> 0.183449 over 17 bounded iterations). On the test
split it cuts ECE from 0.0415 to 0.0161, and that gain is real: the paired
difference is -0.0254 [-0.03286, -0.0166], excluding zero. Its MCE is 0.1755 against
0.0486 uncalibrated -- 3.6x worse at the point estimate -- but the paired difference
is +0.12695 [-0.09459, +0.51834], which spans zero, so by this project's own rule
that is not an established loss either. It is a flag, not a finding, and it points
at the right thing: the abstention rule thresholds a confidence, so worst-bin
calibration is the property it depends on, and worst-bin calibration is exactly the
one this sample cannot resolve.

**The flakiness result has a confound that makes it unreadable as robustness.** At
0% injected noise, training prevalence is 5.02%, PR-AUC 0.1393, recall@25% 0.4819;
at 30% noise, prevalence is 32.00% and PR-AUC *rises* to 0.1653/0.1786. The dataset
is about 5% positive, so flipping labels at random mostly converts passes into
failures and raises the base rate. Every `dRecall` sits within +/-0.0094, i.e. there
is no measurable robustness effect in either direction, and the report now carries a
`confound_to_read_first` saying so. A metric that improves with the noise rate here
is reporting the prevalence change.

**The drift detector shipped inert.** The Page-Hinkley sweep in
`reports/continuous_learning.json` shows threshold 0.5 catching 4 of 5 injected
shifts at 11 false alarms per 600 stationary mutants (median latency 4.5), 1.0
catching 3 of 5 at 6 false alarms, 5.0 catching 2 of 5 at zero -- and the shipped
default of 10.0 catching **0 of 5, with latency `None`**. A detector tuned to never
fire is not a conservative detector, it is an absent one, and the number to publish
is the trade-off curve rather than the default.

**Latency, corrected.** 150 batches of 50 rows drawn from the 86,469-row split:
scoring total 3.345 ms mean, 3.627 p50, 3.943 p90, 4.435 p99, of which ensemble
inference is 3.246 ms and calibration 0.036 ms. 100% of batches are under 100 ms.
Two exclusions are stated rather than hidden: the cold first batch at **1321.814
ms**, roughly 400x the warm mean, and feature mining, which is not in the measured
path at all. The 3.3 ms figure is the cost of scoring features that already exist.

**The one component that works as advertised is the uncertainty ordering.** The
risk-coverage curve is monotone over all 86,469 rows: mean prediction error falls
0.1054 -> 0.0886 -> 0.0781 -> 0.0700 -> 0.0597 -> 0.0483 as coverage drops from 100%
to 50%, with uncertainty-error correlation 0.4417. Ensemble disagreement does rank
what the model gets wrong. That is the mechanism the abstention rule is built on,
and it is measurably real -- which is what makes the degenerate operating point a
tuning failure rather than a dead idea.

**Two robustness fixes with no effect on any number, recorded so the absence of an
effect is on the record.** `ConfidenceCalibrator._sigmoid` overflowed at the bottom
of the bounded temperature search (`T = 0.01` puts logits in the hundreds); it is now
branch-split on the sign of `z` and the NLL uses `logaddexp` instead of clipping
probabilities to 1e-9, which had been handing the optimiser a flat region built out
of the clip bound rather than out of the data. Both fitted temperatures reproduce
exactly (0.7510 on the selection half, 0.7941 on full validation), so this was
robustness only. And `LightGBMFailurePredictor` logged `Best iteration: 0` on every
run, which reads like a degenerate fit; `best_iteration_` is simply 0 when no
`eval_set` is passed, so the log now says early stopping never ran and all trees were
kept.

**Environment, for the record.** The whole pytest suite was blocked by a
`starlette` / `fastapi` collision: `sse-starlette 3.4.10` and `mcp 2.1.1` both
require `starlette >= 0.49.1`, `fastapi 0.115.0` pins `starlette < 0.42`, and
nothing in the project pinned fastapi upward, so collection died on
`TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` --
raised inside fastapi's own internals, not by our code, which already uses the
`lifespan` pattern. Resolved by upgrading **fastapi 0.115.0 -> 0.141.1** in the
shared user site-packages, which needed no project change and is reversible with
`pip install "fastapi==0.115.0"`. Separately, `pip check` still reports
`conftest 0.1.0 requires python-dotenv>=1.0.1, but you have python-dotenv 1.0.0`;
the suite passes and it was left alone rather than touching shared packages twice.

**Tests.** 519 passing, up from 481 at the start of the sprint and 504 before this
entry: `tests/unit/test_policy_tuning.py` (11) covers both objectives, the
unsatisfiable case raising instead of writing `(0.015, 0.50)`, NaN records being
ineligible rather than worst, deterministic tie-breaking, and grid parsing;
`tests/unit/test_g5_gate.py` (4) covers commit-level resampling, an undefined recall
interval staying undefined instead of collapsing to zero, and per-commit records
being off by default. The fabrication guard passes 110 files on both rules -- which,
as above, is necessary and not sufficient.

**Artifacts regenerated in this entry.** `reports/baseline_comparison.csv`,
`baseline_comparison_intervals.{csv,json}`, `baseline_per_commit.csv`,
`statistical_significance.json`, `economic_analysis.json`, `calibration_report.json`,
`policy_tuning_report.json`, `flakiness_robustness.json`, `latency_benchmark.json`,
`explanations.json`, `continuous_learning.json`, `uncertainty_analysis.json`,
`ablation_study.json`, `cross_repo_generalization.json`, and two new ones:
`reports/g5_recall_floor.json`, `reports/policy_tuning_recall_floor.json` and
`reports/g4_ml_vs_baselines.json`. The
IEEE paper, the KTU report and the viva deck still quote the pre-C5 numbers and are
now the largest remaining inconsistency in the repository.

---

### 2026-09-03f · G1 closes on a rule about what a label means, not on a threshold that moved

`validators` sat at 0.752% against a 1% floor while contributing 39.7% of all rows.
Two of the three available moves were dishonest. Lowering the floor to fit the
measurement is the definition of fitting a gate to its result. Dropping the repo
because it failed, and saying nothing about why, is the same act with a tidier
diff. The third move is to ask what the floor is actually for, and whether the
denominator underneath it was ever the right one.

**What the rate is for.** The 1-15% band exists because a selection model needs a
signal that is neither absent nor everywhere. Below roughly 1% a ranker cannot be
distinguished from one that returns nothing; above 15% the cheapest correct policy
is to run the whole suite and there is nothing to select. Both statements are about
rows a ranker could get *right*. A mutant that no test in the universe detects
produces no such row: every one of its rows is label 0, no ordering of tests
retrieves anything, and recall against it is undefined — 0/0, not 0. It contributes
denominator and no possible numerator. Including those rows does not make the rate
conservative, it makes it a measurement of a different quantity.

So the denominator changed, on a rule stated independently of which repo needed it:

| repo | pairs | rows from detected mutants | mutants detected | killed nothing | rate (detected) | rate (all rows) |
|---|---|---|---|---|---|---|
| tabulate | 80,968 | 65,263 | 187 | 45 | **11.723%** | 9.449% |
| validators | 222,855 | 141,410 | 158 | **91** | **1.185%** | 0.752% |
| sqlparse | 119,652 | 89,232 | 176 | 60 | **10.790%** | 8.047% |
| pathspec | 46,986 | 30,369 | 159 | 87 | **12.309%** | 7.956% |
| pyjwt | 91,250 | 76,285 | 209 | 41 | **6.228%** | 5.207% |
| **pooled** | **561,711** | **402,559** | **889** | **324** | **6.817%** | **4.886%** |

`g1_repos_outside_the_band` is now `[]` and `g1_pairs_met_by_every_repo` is true, so
G1 is ticked. Four things keep that from being a threshold moved to suit an outcome:

- The rule is applied to **all five repos**, not to the one that needed it. It moves
  every repo up, and it moves tabulate and pathspec up by more than validators.
- It is the ordinary criterion in mutation testing. Surviving mutants measure suite
  *adequacy*; they cannot measure test *selection*, which is what this dataset is for.
- **Both rates are published**, in the manifest and in the CLI summary, per repo and
  pooled. `failure_rate_all_mutants` is the unfiltered figure and it is not hidden
  behind the one the gate reads.
- The 159,152 undetected rows **stay in the dataset**, flagged `mutant_detected=0`
  rather than deleted. Anyone who thinks the exclusion is wrong can undo it with one
  filter, on the same file, without a rebuild.

The flag is derived from labels, so as a model input it would be perfect leakage —
knowing that some test kills this mutant is most of the answer to which one does.
`FEATURE_NAMES` is an explicit allowlist and `trainer.py` selects on it, so the
column cannot reach a model by accident; a test asserts the exclusion anyway
(`test_the_detection_flag_is_not_a_model_feature`), because the protection lives in
a different file from the column and a future refactor of either would not notice.
Detection is also judged against the **resolved universe**, the same set the rows are
drawn from, not against the raw kill list: a mutant whose only kill is a test ID the
resolver could not map produces no row for that test, and counting it as detected
would credit a kill the dataset does not contain.

**Two findings that are not about G1.** First, 324 of 1,213 mutants — 26.7% — were
killed by no test at all, and 91 of those are in validators, where 36.5% of mutants
survive a suite of 895 tests. That is a statement about those suites, not about this
pipeline, and it is the more interesting number on the page: validators' failure rate
is low because its suite detects little, and the 1.185% it clears the floor with is
thin enough that it should be quoted with that context every time. Second, validators
produces 39.7% of all rows from 20.6% of the mutants, because rows scale with
universe size; any pooled figure is therefore weighted toward the repo with the
largest suite, which is why the per-repo table is the one that matters and the pooled
rate is reported beside it rather than instead of it.

**Measured**: `python -m pytest tests/ -q` reports `484 passed, 9 warnings in 60.63s`
(three new tests), `scripts/check_no_fabricated_labels.py` reports
`PASSED (108 files scanned)`, and the rebuild reports
`Failures : 27,444 (6.817% of detected-mutant rows, 4.886% of all rows)` with
`G1 per repo : every repo in band`. The manifest is now tracked
(`!data/processed/real_features_manifest.json`) so the gate can be checked without a
10-minute rebuild, while the 561,711-row CSV stays out of the repository.

**What this buys, and what it does not.** It buys a dataset that clears its own
entry gates on a rule that was written down before the numbers were looked at, and a
denominator that means what the band was defined against. It does not buy a single
published result: every number in `reports/` still comes from fabricated labels or
fabricated inputs, and three report scripts —
`run_statistical_tests.py`, `run_cross_repo_eval.py`, `run_continuous_learning.py` —
do not read a dataset at all. They generate one. That is C5, and it starts now.

### 2026-09-03e · C0.8 + the real dataset · one broken timeout cost 79% of the harvest budget, and G1 misses on one repo

**Test suite: 460 -> 481 passing** (21 new: 9 in tests/unit/test_executor.py, 9 in
tests/unit/test_build_real_dataset.py, 3 in tests/unit/test_mutation_harness.py).
`data/processed/real_features.csv` is on disk: **561,711 rows, 27,444 measured
failures, 4.886%**. Every label is a pytest outcome under an AST mutation. Measured:
`python -m pytest tests/ -q` reports `481 passed, 9 warnings in 73.36s`, and
`scripts/check_no_fabricated_labels.py` reports `PASSED (108 files scanned)`.

**`subprocess.run(timeout=...)` is not a timeout when you capture to a pipe.** The
sqlparse harvest took 8.47 h. 7.85 h of it was one mutant. `mut_42bfc7db58b0`
(`sqlparse/cli.py:197`, `cond_to_True`, `args.inplace` -> `True`) forced the CLI
into in-place rewriting; the suite's ceiling was 180 s and its record says
`suite_duration: 28269.997`. The mechanism is in CPython, not in sqlparse: on
`TimeoutExpired`, `subprocess.run` kills **the direct child only**, then calls
`communicate()` **with no timeout** to drain the pipe. A grandchild inherited the
write handle and held it, so the drain never returned. The exception was caught
7h51m late, which is why the log line reads `timed out after 180s` next to a
duration of 28,270 s. The record was honest; the log line was not.

**It cost more than the rest of the study combined.** Summing `suite_duration`
over all 1,250 harvested records: **9.97 h of measured suite time, of which that
single run was 7.85 h (79%)**. Excluding it, all five repos together cost 2.11 h.
The status table above said `harvest = 1.24 h estimated`; the honest figure is
2.11 h clean, and the estimate is now replaced by the measurement.

**It is not a defect that fires on every timeout — it fires when a grandchild
holds the handle.** Across the five harvests there were **12 timeouts; 11 held**
(pathspec 1, tabulate 1, sqlparse 9, each stopping at 180 s) **and 1 did not**.
That is why four earlier harvests completed in ~25 min each and nothing looked
wrong. A fault that shows up once in twelve is exactly the kind that ships.

**That window is when the checkout was damaged.** The single `checkout_drifted`
event in all 1,250 records is that same mutant: `tests/files/function.sql`,
truncated by sqlparse's own suite while the mutated tree sat live on disk for
7h51m. The timeout was the containment boundary for D2, and it was the boundary
that failed.

**The wall clock, since the record does not carry one.** From the harvest log:
mutant `[55/227]` finished at `08:43:46`, `[56/227]` -- the broken one -- was
started immediately after, and its `Test execution timed out after 180s.` line was
emitted at `16:34:56`. 28,270 s between the start and the exception, and the
harness's own ETA jumped from `54m` to `1492m` in one step. Drift detection then
fired *correctly* and restored `tests/files/function.sql` in the same second: the
guard was not missing, it was simply 7h51m downstream of the damage. Note also
what is in that mutant's kill list -- `test_split_create_function[function.sql]`,
the test for the fixture the run had just truncated. A drifted record can
manufacture its own kills, which is why the exclusion is on the record and not on
the individual test.

**The fix, in `src/conftest/tests/executor.py`.** Capture to files rather than
pipes, so no handle is shared with a descendant and there is nothing to drain.
`stdin=subprocess.DEVNULL`, so a suite that reads stdin gets EOF instead of
waiting on a terminal that will never answer. `Popen` + `proc.wait(timeout=...)`
in place of `run`, and on `TimeoutExpired` a real tree kill: `taskkill /T /F /PID`
on Windows, which has no process group to signal, and `killpg(SIGKILL)` on POSIX
where the child is given its own session. The wait after the kill is bounded by
`KILL_GRACE_SECONDS = 30` -- never unbounded, because unbounded is the defect it
replaces. If the tree outlives that, `_kill_process_tree` returns False and says
so. A final post-hoc check catches whatever the kill path believed: a run whose
wall clock exceeds `timeout + BROKEN_TIMEOUT_SLACK_SECONDS` (60 s) sets
`timeout_enforced = False` regardless.

**A timeout that did not hold is now its own status.** `timed_out` reads as "this
mutant was slow, drop the record". What actually happened needs different words,
so `TIMEOUT_BROKEN_STATUS = "timeout_broken"` outranks every other status, and
`timeout_enforced` is stamped on every record next to `timed_out`. Both tally
dicts initialise the key at zero, so a summary reporting no broken timeouts says
so explicitly rather than omitting the field -- the distinction between *none* and
*not measured* is the whole reason the field exists.

**The builder now refuses a `harvested` record it cannot trust.** `status:
harvested` says the suite ran to completion under one mutation. It does not say
the tree underneath was the screened revision. `contamination_reason` in
`scripts/build_real_dataset.py` rejects two cases: a record carrying
`checkout_drifted` (the suite edited its own tracked files, so its kills may be
inherited damage), and a record carrying `timeout_enforced: false` (its outcomes
were read while something was still writing). Excluded mutants are listed in the
manifest **by id and reason**, not counted -- a count cannot be checked against the
harvest file by a later reader. Past `MAX_CONTAMINATED_FRACTION = 2%` the build
raises instead of quietly shrinking: one incident is an incident, one in fifty is
a broken harvest, and training on the clean remainder while publishing a headline
is precisely what this pipeline exists to stop.

**On the real data it caught nothing, and that is the result.** Zero harvested
records across all five repos carry `checkout_drifted`; the one that does is
`timed_out`, so status had already excluded it. The guard is insurance for the
next harvest, not a repair of this one.

**What the absence of the field does not prove.** Three of the five harvests
(pathspec, pyjwt, tabulate) predate the drift check entirely -- their summaries
carry no `drift_detection` key, so a missing `checkout_drifted` there means *not
measured*, not *clean*. Only validators and sqlparse record `drift_detection:
git`. `contamination_reason` refuses what a harness observed and reported; it
cannot speak for what was never checked, and it does not pretend to.

#### G1: the dataset, and the one gate it does not clear

| repo | mutants | universe | rows | failures | rate | in band |
| --- | --- | --- | --- | --- | --- | --- |
| tabulate | 232 | 349 | 80,968 | 7,651 | 9.45% | yes |
| sqlparse | 236 | 507 | 119,652 | 9,628 | 8.05% | yes |
| pathspec | 246 | 191 | 46,986 | 3,738 | 7.96% | yes |
| pyjwt | 250 | 365 | 91,250 | 4,751 | 5.21% | yes |
| validators | 249 | 895 | 222,855 | 1,676 | **0.75%** | **no** |
| **pooled** | 1,213 | — | **561,711** | **27,444** | **4.886%** | yes |

- **G0** — met: 5 stage-3 survivors, import provenance verified for all five.
- **G1 pairs** — met: every repo clears 5,000 labelled pairs by an order of
  magnitude; the smallest is pathspec at 46,986.
- **G1 rate** — **pooled 4.886% is in band; validators at 0.75% is below the 1%
  floor.** It is not a rounding matter: validators contributes **39.7% of all rows
  and 6.1% of all failures**. Its 895-test universe is the largest and its mutants
  kill the fewest tests, so it drags the pooled rate down by roughly two points.
  The pooled figure passing does not make the repo pass.
- **G2** — met by the guard, not by the literal grep. `grep -rn "random"` over the
  label path is *not* zero and cannot be: mutant **sampling** is seeded
  (`random.Random(self.seed)`, `mutation_harness.py:935`). The operative check is
  `scripts/check_no_fabricated_labels.py`, an AST guard that distinguishes
  choosing which mutants to run from deciding whether a test failed. **PASSED,
  108 files scanned.** The plan's literal phrasing is superseded by it.
- **G3** — met: `git status --porcelain` is empty in all five checkouts, and no
  tracked file is modified anywhere. Getting there turned up one more instance of
  the D2 class: sqlparse and tabulate each held an untracked file named `-`, 52 and
  114 bytes, written by their own CLI tests -- formatted SQL and a rendered table.
  On Windows `-` is a filename, not stdout. Untracked, so no label was touched,
  and they return on the next suite run; this is exactly why the harness reports
  untracked files separately from tracked drift rather than folding them together.

**The gate was implemented weaker than it was written.** Section 9 defines G1 as
">= 5,000 labelled pairs **per repo**" and a failure rate "in 1-15%", but the
manifest reported one pooled `failure_rate` and one pooled verdict. Pooled, this
dataset reads green. Per repo it does not. A gate that aggregates away the repo it
would have failed on is not a gate, so the manifest now carries `g1_pairs_met` and
`g1_failure_rate_in_band` on every repo, plus top-level
`g1_pairs_met_by_every_repo`, `g1_failure_rate_in_band_for_every_repo` and
`g1_repos_outside_the_band`, and the CLI prints `G1 per repo : OUT OF BAND for
validators 0.75%` next to the pooled line rather than under it.
`test_the_manifest_judges_g1_per_repo_and_not_only_on_the_pool` drives the builder
end to end on a repo at 25% and asserts the repo is named in that list.

**What this buys, and what it does not.** The dataset is real: 561,711 rows whose
labels are pytest outcomes, 5 repos with verified import provenance, 0 unresolved
test IDs, 0 contaminated exclusions, and a restore that is byte-exact in all five
checkouts. The measured harvest cost is **2.11 h** of suite time excluding the
pathological run and **9.97 h** including it -- the plan's `harvest = 1.24 h
estimated` was optimistic by 70% even after the pathology is set aside, and that
line is now the measurement. What it does not buy is a clean G1: validators sits at
0.75% against a 1% floor, and 13 of the features are inert by construction because
a mutant has no commit message and no wall-clock time. Both are recorded rather
than smoothed. The next decision is validators' -- raise its rate by sampling
mutants in the modules its 895 tests actually exercise, or drop it from the pool and
say so -- and it is a decision about the subject, not about the number.

---

### 2026-09-03d · C0.7 + C4.8 landed · the harvest was corrupting the checkout it measures, three ways

**Test suite: 371 -> 460 passing** (89 new: 42 in `tests/unit/test_mutation_harness.py`, 34 in the new `tests/unit/test_headline.py`, 13 in `tests/unit/test_dashboard.py`). Measured: `python -m pytest tests/ -q` reports `460 passed, 9 warnings in 75.50s`.

Every label this project publishes is produced by running a subject's own suite against a mutated copy of that subject's own source, in a checkout on disk. That makes the working tree an instrument, and this session found three distinct ways the instrument was being contaminated. All three produce labels that are indistinguishable from good ones: the mutant id, the kill list and the kill ratio all look ordinary.

**D1 — a hard kill leaves the mutation applied, so the next run mutates a mutant.** The sqlparse re-harvest was reported finished by its background wrapper with exit code 0. It had in fact died with its mutation still written into the checkout, which is visible in the successor run's first log line: `08:09:33 | Restored 1 tracked file(s) in ...data/repos/sqlparse`. Without `--restore-checkout` that successor would have applied its next mutant on top of a dead run's mutant and attributed every kill to its own mutant alone. The exit code lied in the other direction too: the wrapper reported 0 while its `nohup`-detached child was still logging `[23/250]` at 08:16:42 and starting another pytest in that same second. **A background exit code is not evidence that a harvest finished — `harvest_summary.json` is.** The harness now writes `pending_mutation.json` before applying each mutation and removes it after reverting, asserts a pristine checkout at startup, and repairs with `git checkout HEAD -- .` under `--restore-checkout` — never `git clean`, so the subject's own untracked debris (sqlparse's CLI tests leave a file literally named `-` on disk) survives the repair.

**D2 — a suite that edits its own tracked fixtures is not flaky, and must not be quarantined as flaky.** sqlparse's CLI tests truncate their own tracked fixture `tests/files/function.sql`, and the damage was already on disk when the baseline screen ran. With that fixture empty exactly one test fails, so the screen recorded

```
sqlparse/tests/test_split::test_split_create_function[function.sql]
  -> "failing_on_clean_checkout:FAILED"
```

which is a false description of a clean checkout. The labelled universe became 506 tests instead of 507: one test of 507 could never be killed by any mutant, and every kill ratio was computed against the wrong denominator — including the `BROKE_SUITE_KILL_RATIO = 0.80` cutoff that decides whether a mutant is discarded as suite-breaking. A complete 250-mutant harvest (237 harvested, 4 broke_suite, 9 timed_out, 15964 kills, 40.2 min) had already run against that tree; it now sits in `data/harvest/sqlparse/superseded-2026-09-03/` with a `WHY.md`, kept for comparison and not for use. The screen restores tracked files between runs and records what it restored. Measured after the fix: **507 stable tests, 2 exclusions (both genuinely `skipped`), 0 flaky, `self_damage: []`, mean suite 8.575 s over 3 runs.**

**D3 — two harvests, one checkout. Found by causing it.** The remedy for D1 became the weapon. At 08:18:12 I started a second harvest over the same output directory while pid 11060 was still working in it. One second later the new run did exactly what it had been told to do: `08:18:13 | Restored 1 tracked file(s)`, discarding the mutation the live run was in the middle of testing, followed by `Discarded the mutation a dead run left applied: mut_10c6746605e2 in sqlparse/filters/reindent.py (operator const_int_increment)` — whose author was not dead. Both runs then read the same 23-record checkpoint and the same seeded sample (seed 42, `sample_digest c51fcbbb33f7d381`), so both chose `mut_10c6746605e2` as the next mutant, and the second applied it at 08:18:28 under a pytest of its own. Two harvests were mutating one tree and appending to one `mutants.jsonl`. I killed both process trees and counted: 23 records, 23 unique mutant ids, checkpoint 23 — nothing was written during the overlap, so no labels had to be thrown away.

The remedy is a lock, and its **ordering** matters as much as its existence: `harvest.lock` is acquired *before* the pristine-checkout assertion, so a second run refuses before it can "repair" a live run's tree. The lock records pid, host, start time and argv; it is released in a `finally`; a dead pid on the same host is taken over with a warning; a lock naming another host is refused outright, because no liveness claim can be made about a pid on a machine we cannot see; an unreadable lock counts as held; and `--break-lock` is the explicit, loudly logged override. `harvest_summary.json` now records the `harvester` that wrote it.

**Four smaller faults, all of them mine:**

1. **The guard refused to run its own remedy.** `assert_pristine_checkout(restore=True)` raised whenever a journal existed — which is precisely when `--restore-checkout`, the flag its own error message recommends, is needed. A journal *names* damage; it is not damage of its own. Once `git` proves the tree matches HEAD the journal is resolved, logged with the mutant it discarded, and recorded as `resolved_pending_mutation`. When `git` cannot prove the tree is clean, the refusal stands.

2. **`os.kill(pid, 0)` would have killed the harvest it was asking about.** The POSIX liveness idiom is unusable on Windows, where `os.kill` calls `TerminateProcess`. Liveness is probed with `OpenProcess` + `GetExitCodeProcess` through `ctypes` (`STILL_ACTIVE = 259`, `ERROR_INVALID_PARAMETER = 87`), and it returns `None` when it genuinely cannot tell. `None` counts as *held*: taking over a lock you cannot prove is stale is the entire defect.

3. **A resume trusted a checkpoint it could not attribute.** The validators harvest had 154 records on disk, of which only 134 belonged to the current seeded sample. The harness now records `sample_digest` and verifies membership on resume: the 20 off-sample records were moved to `data/harvest/validators/quarantine_offsample.jsonl` instead of being silently counted (digest `9261998e43395546`). Resumed to completion: **250 records / 250 unique / 249 harvested / 1 broke_suite / 0 timed_out / 2559 kills / universe 895 / `checkout_drifted: 0`.**

4. **The summary described the process, not the file.** A resumed run reported its own counters as the harvest's totals, so a run that found everything already done would have published zeros. The top-level keys now describe the file (`records`, `unique_mutants`, `harvested`, `total_kills`) with `this_run` alongside — and in the validators summary `this_run` is all zeros, which is the honest description of a resume that had nothing left to do.

**C4.8 — the front page was the fourth fabricator.** The dashboard's KPI row was four literals: `value="100.0%"` failure recall with `delta="0 Escaped Bugs (Safe Fallback)"`, `value="68.6%"` test reduction, ECE 0.0192 at `delta="-25.47% Error (Calibrated)"`, and disagreement 0.0193. The project's own `reports/baseline_comparison.csv` contradicts the first two on the very row they claim to describe: the ConfTest selector recalls **40.0%** of failures, lets **3** commits escape, and cuts wall clock by 58.8% (test count by 60.0%) — 68.6% is neither of those numbers. The chart drew a green "100% Zero-Escape Frontier" annotation at the fabricated value.

Every cell is now read from a named artifact by `src/conftest/evaluation/headline.py`, which raises `MissingArtifact` rather than defaulting, and each metric renders as `not measured` beside the script that would produce it when its artifact is absent. `tests/unit/test_headline.py` (34 tests) pins four rules: the time-reduction cell is **ETR**, with TRR only in the note, because the two were being conflated; a difference whose interval spans zero is reported as such (`ECE difference -0.00657 [-0.01119, +0.01102] spans zero, so not a gain`); the calibration cell names the method actually **served**, not the best-scoring one; and the page opens with a provenance banner that turns green only when `data/processed/real_features.csv` exists — which it does not yet, so the front page currently states that the labels are not measured, which is true.

**One regression surfaced while testing the loaders.** `dashboard/utils.py` stripped `%` from percent columns only when `df[col].dtype == object`, a condition pandas 3 never satisfies for those columns because it infers `StringDtype`. On pandas 3 the coercion silently did nothing and `idxmax` compared `"100.0%"`-style strings as text. The guard is gone, the coercion is unconditional, and `errors="coerce"` keeps `n/a` as `NaN` instead of turning it into zero.

**The three pre-guard harvests were re-audited rather than trusted.** pathspec, pyjwt and tabulate were harvested before any of this existed: their summaries carry no `checkout` block, no `unique_mutants` and no drift counter, so the guards can say nothing about how those runs behaved. What can still be measured after the fact was measured. All three exclude tests for `skipped` only (14 / 4 / 17 exclusions), so none carries the D2 signature of a `failing_on_clean_checkout` quarantine. None has a duplicate mutant id in its `mutants.jsonl` (250 records, 250 distinct ids each), so no resume double-counted a mutant. A persistent-kill scan — looking for a test that is killed by every mutant from some point to the end of the file and by none before it, which is what a mutation left applied mid-run looks like in the labels — returns **zero suspects in all five harvests**, including the 145 labelled records sqlparse has written so far. Tracked files are clean in every subject checkout (`git status --porcelain` empty for pathspec, pyjwt and validators; sqlparse holds only the mutation its live run is currently testing; sqlparse and tabulate each carry an untracked file named `-` that their own CLI tests create). None of that is equivalent to having run under the guards, and this entry should not pretend it is — it is the strongest statement those artifacts support.

**What this does and does not buy.** Each of the three defects is now refused rather than noticed afterwards, and each refusal has a test that reproduces the measured incident — including one asserting that a live lock stops `--restore-checkout` before it can discard a live run's work. What none of it buys is a dataset. sqlparse is being re-harvested against the corrected 507-test universe as this is written (250 mutants, resumed under the lock held by pid 21664; 139 records at the time of writing: 129 harvested, 4 broke_suite, 6 timed_out). A `timed_out` record carries a full-universe kill list and would be poison as a label, which is why the builder consumes only `status == "harvested"`. G1 stays open until that summary is on disk and G0–G3 are verified against it.

### 2026-09-03c · the calibration endpoint was serving its own set of literals

**Test suite: 366 -> 371 passing** (5 new in `tests/unit/test_api_endpoints.py`).

C4.1 and C4.6 named the two fabricators in the label-producing paths. `GET /api/v1/calibration` was a third route to the same outcome, in the reporting layer, where the guard does not look: when `reports/calibration_report.json` was absent it returned a complete, plausible calibration result built from hand-typed constants.

```python
return CalibrationResponseSchema(
    best_method="temperature_scaling",
    uncalibrated=CalibrationMetricItem(ece=0.0258, mce=0.2222, brier_score=0.0449),
    calibrated=CalibrationMetricItem(ece=0.0192, mce=0.8943, brier_score=0.0449, ece_reduction_pct=25.47),
    temperature=0.9275,
)
```

Nothing in the response distinguished this from a measurement. It asserted the 25.47% ECE reduction that log 2026-09-03b shows is `-0.0066 [-0.0112, +0.0110]` — indistinguishable from zero — and it named temperature scaling the best method while carrying that method's own MCE of 0.8943, four times worse than leaving the model alone. It is now **503** with a message naming the script to run. An endpoint that reports measurements has no defaults to serve.

**Three smaller faults in the same 40 lines:**

1. **`temperature=0.9275` was a literal on the loaded path too**, not read from the report — because the report never recorded the fitted temperature. Any rerun that fit a different T was reported under the old constant. `calibrate_model.py` now writes `fitted_temperature` and the route reads it.

2. **`best_method: "uncalibrated"` was reported as a calibrated model.** The lookup fell back to `test_metrics[best_method]`, which for a declined run finds the uncalibrated row and returns it in the `calibrated` field — stating the opposite of the decision. `calibrated` is now `Optional` and is `None` when no method was chosen, which is the outcome the interval-based selection produces whenever no ECE gain clears the noise.

3. **The intervals stopped at the JSON boundary.** `CalibrationMetricItem` carried three bare floats, so an API client saw what `score_record` exists to stop a reader seeing: three ECE values with no way to tell that the smallest won nothing. Each metric now carries its optional `*_vs_uncalibrated` block, and the response carries `selection_basis` and `selection_reason`.

**Two artifacts changed as a consequence.** `reports/calibration_report.json` was regenerated and now records `best_method: "uncalibrated"` with `basis: "bootstrap"` — the old file was written on 2026-08-26 and still claimed temperature scaling won. `models/calibrator.joblib` is deleted, because the run declined to fit one; `ConfTestEngine` already handles that by falling back to identity calibration and warning, and `tests/unit/test_dashboard.py` no longer asserts the artifact's existence as though it were a property of the engine. Both files are derived from the fabricated splits and will be regenerated again by C5.


### 2026-09-03b · C4.4 completed · the calibration decision now rests on intervals, and the split no longer leaks

**Test suite: 348 -> 366 passing** (18 new: 5 in `tests/unit/test_statistics.py`, 13 in `tests/unit/test_calibrator_selection.py`).

C4.4 was ticked with half of its own row unmet. The plan asked for the choice to be made on "ECE **+ MCE + Brier** jointly **with bootstrap CIs**"; what shipped compared three point estimates against two hardcoded tolerances (`MCE_TOLERANCE = 0.10`, `BRIER_TOLERANCE = 0.01`). Numbers picked by hand, defended by nothing, deciding which calibrator the abstention policy would read confidence from.

**The intervals reverse a decision the tolerances get wrong, on this project's own data.** Temperature scaling measured ECE 0.0192 against 0.0258 uncalibrated on the test split — the report printed `ece_reduction_pct: 25.47`. Its paired difference over resampled mutants is **-0.0066 [-0.0112, +0.0110]**. A 25% improvement and an interval that spans zero are the same measurement. Every guard metric sat well inside tolerance, so the old rule accepted it without comment; the interval path declines and says why. `tests/unit/test_calibrator_selection.py::test_a_lower_ece_whose_interval_spans_zero_is_not_a_gain` holds both paths against the same candidate so the disagreement stays visible.

**Rejection now requires evidence in both directions.** A candidate is disqualified only when its difference is *positive and* excludes zero, and accepted only when the ECE difference is *negative and* excludes zero. The old rule read `score.ece >= baseline.ece`, which promotes any lower number, and `score.mce > baseline.mce + 0.10`, which forgives a real degradation that happens to be small. Symmetry matters here: noise on the wrong side of zero was previously grounds for rejection.

**The validation half-split was leaking mutants across both halves.** `split_for_selection` cut the row array at an index, so a mutant whose tests straddled the cut was partly fitted on and partly held out. Isotonic regression is flexible enough to learn one mutant's failure pattern and be rewarded for it on that same mutant's remaining rows — the precise mechanism by which the more flexible method wins a comparison it should lose, and the leak the half-split existed to prevent. `split_clusters_for_selection` assigns whole clusters, balanced by **row** count rather than cluster count because mutants differ widely in how many tests they carry. Fewer than two clusters is now an error rather than a silent fallback.

**Resampling rows instead of mutants would have understated every interval by ~3.5x.** Same rows, same labels, same probabilities, same point estimate — only the resampling unit differs. Measured on a fixture where failures concentrate inside mutants, as injected faults do: ECE difference `[-0.020, +0.039]` by mutant against `[-0.003, +0.013]` by row. This is the same correction C4.5 made in the benchmark, and it is the reason the calibration path could not simply reuse the counts matrix: ECE and MCE are not ratios of sums, so a replicate has to gather the drawn clusters' rows and recompute over them.

**One generator now owns every resample draw.** `cluster_draws` yields the draws; `bootstrap_counts` bincounts them, `paired_cluster_bootstrap` indexes with them, and `score_calibrators` gathers rows through them. Three separate walks of the same RNG, kept in step by convention, was one too many for the pairing between metrics to rest on. Brier is also computed by hand rather than through `brier_score_loss`, which infers the positive label from the labels present and raises when a resample draws only one class — routine at a 1–15% failure rate.

**What this does not buy.** The decision falls back to the fixed tolerances whenever the dataset carries no cluster column, and that fallback is recorded in the report as `basis: "point"` rather than passed over. If any one candidate lacks intervals the whole comparison drops to tolerances, because judging one candidate on measured noise and another on a hand-picked constant compares them on incomparable evidence. And the numbers quoted above come from the current fabricated splits — 2 mutants in the selection half, 5 in the test split. They demonstrate the machinery, not a calibration result. The real ones arrive with G1.


### 2026-09-03a · C4.5 landed · every reported number now carries an interval, and the unit is the commit

**Test suite: 335 -> 348 passing** (13 new: 6 in `tests/unit/test_statistics.py`, 7 in `tests/unit/test_baselines.py`).

The comparison table was eight rows of bare point estimates pooled over a few hundred commits. At that sample size the gap between two strategies is routinely smaller than the noise in either one, so the table could not answer the question it was printed to answer. It now reports `point [lower, upper]` in every cell, plus a final column giving the paired difference in failure recall against the ConfTest row.

**The resampling unit is the commit.** Test rows inside a commit share a diff and a single injected fault, so they are not independent draws. Resampling rows would have produced intervals several times narrower than the evidence supports — the cheapest available way to overstate a result, and the one a reader cannot detect from the output. `tests/unit/test_baselines.py::test_the_resampling_unit_is_the_commit_not_the_test_row` pins it against a 12-commit / 96-row frame: every interval must report `num_units == 12`.

**Differences are paired, not compared by eye.** Marginal intervals that overlap can still hide a difference that excludes zero — `test_pairing_is_what_makes_a_difference_measurable` constructs exactly that case (strategy b beats a by 5 points on every cluster while both vary widely across clusters). All 8 strategies are therefore evaluated on the *same* resample, and the difference is taken within each replicate. Only that interval gets to carry the `*`.

**A NaN correction that was a real bias, not a cosmetic one.** `fold_commit_metrics` previously divided by `max(1, denominator)`. Pooled once over a whole dataset that is harmless. Inside a bootstrap it is not: every replicate that happens to draw no failing commits reported 0% recall instead of *undefined*, dragging the lower bound to a value no resample ever produced. Empty denominators are now NaN, excluded from the percentile, and **counted** — `undefined_replicates` is reported alongside every interval, so "undefined in 40% of resamples" is visible rather than inferred from a suspiciously wide interval. Same reasoning at the top level: a dataset with nothing to recall now prints `n/a`, not `0.0%`.

**A reproducibility bug found only because the intervals were built.** The point table and the interval table disagreed on the Random-k row (12.5% vs 50.0%). `RandomKSelector` carried its `random.Random` across sweeps, so the second evaluation of one dataset drew a different sample than the first. Fixed with a documented `BaseSelector.reset()` hook called once per sweep. This was pre-existing and invisible while only one sweep was ever run per process.

**Speed came from the sufficient statistics, not from fewer replicates.** Every metric is a ratio of sums over commits, so `accumulate_per_commit` keeps the per-commit numerators and the bootstrap re-folds them; `bootstrap_counts` returns a `(B, n)` multiplicity matrix so each replicate's totals are one matrix product. 8 strategies x 5 metrics x 2000 replicates costs two matrix products per strategy instead of 80,000 selector sweeps. `test_the_counts_matrix_draws_the_same_resamples_as_the_callable_path` asserts the shortcut draws the same resamples as the plain loop under one seed, because that is the only thing that makes it sound.

**What this does and does not buy.** It buys honest error bars and a defensible headline comparison; `scripts/train_baseline.py --bootstraps 0` now warns that bare point estimates should not be quoted on their own. It does **not** make the current numbers meaningful — they are still computed over fabricated labels until G1 lands and C5 re-runs everything on `real_features.csv`. It also does not yet cover the calibration metrics: ECE/MCE/Brier in `src/conftest/models/calibration.py` are still reported without intervals, which is the next place this machinery is owed.

### 2026-09-02f · C4.6 landed · the second fabricator now refuses to be used by accident

**Test suite: 333 -> 335 passing** (2 new, both in `tests/unit/test_git_collector.py`).

`src/benchmark/dataset_generator.py` was already blunt about what it emits. `src/conftest/repository/synthetic_generator.py` was not: its docstring described it as a repository generator, and nothing at the call site distinguished it from a component that reads real history. Both fabricate labels the same way, so both now say so in the same voice.

Four changes, in increasing order of how hard they are to ignore:

1. **The docstring quotes the fabrication** rather than describing it, so the coin flip is visible without opening the method:

   ```
   WARNING: every test outcome this module emits is a coin flip, not an observation.

       fail_prob = failure_rate if is_affected else 0.005
       is_fail = self.rng.random() < fail_prob
   ```

2. **Construction requires an acknowledgement.** `SyntheticRepositoryGenerator()` now raises; only `acknowledge_synthetic=True` builds one, and the error names `conftest.groundtruth.mutation_harness` as the real-label route. This is the part that cannot be skimmed past -- an evaluation script that reaches for this class fails at line one instead of producing a plausible-looking number.

3. **Every row carries its origin.** `SYNTHETIC_ORIGIN_TAG = "SYNTHETIC_FABRICATED_LABELS"` is stamped on each test-run dict, each commit dict, and the run metadata, which also records `labels_are: "sampled from rng.random(), never measured by running a test"`. The tag string is identical to `dataset_generator.py`'s, so one `grep -rn SYNTHETIC_FABRICATED_LABELS` finds every fabricated row in the system regardless of which generator produced it. `persist_to_database` builds explicit keyword payloads, so the extra key is inert at the DB boundary.

4. **The exemption in `scripts/check_no_fabricated_labels.py` now states the grounds** -- name, gated construction, stamped origin -- instead of asserting the file is fine. The guard still passes: `FABRICATED LABEL CHECK: PASSED (107 files scanned)`.

`scripts/collect_repository_data.py` is the one legitimate caller and passes the flag.

**What this does and does not buy.** It makes accidental use loud, and it makes fabricated rows self-identifying if they ever reach a table. It does not remove the class, because the pipeline smoke tests need a repository-shaped object that costs nothing to build. The honest framing for the report is that ConfTest contains exactly two fabricators, both are named, gated, and tagged, and neither can reach an evaluation path without an explicit acknowledgement in the calling code.

---

### 2026-09-02e · pilot harvest measured · a verified interpreter that leaves no receipt

**Test suite: 321 -> 333 passing** (12 new: 3 harness, 7 builder, 2 driver).

#### The pilot, at full size

25 mutants each on the two repos with the widest suites, under their own screened venvs:

| repo | mutants | universe | rows | failures | rate | all-negative | status |
|---|---|---|---|---|---|---|---|
| sqlparse | 25 | 507 | 12,168 | 1,274 | **10.47%** | 1/24 | 24 harvested, 1 timed out |
| pyjwt | 25 | 365 | 9,125 | 235 | **2.58%** | 4/25 | 25 harvested |
| **combined** | 50 | - | **21,293** | **1,509** | **7.09%** | 5/49 | - |

7.09% sits inside the G1 band of 1-15%, and it is a *measured* rate: every one of those
1,509 failures is a pytest run that actually failed under a mutation.

#### The positive class is concentrated, and that is the interesting finding

sqlparse kills per harvested mutant: `0 1 1 1 1 1 1 1 2 2 3 3 5 5 7 10 12 27 27 36 37 322 384 385`.
Three mutants carry 1,091 of the 1,274 failures -- **86%** of the repo's positive class from
12% of its mutants. pyjwt is milder but shows the same shape: one mutant at 130 kills against
a median of 3.

All three sqlparse outliers were read before being accepted:

| mutant | site | why it is wide, not degenerate |
|---|---|---|
| `arith_+_to_-` | `grouping.py:288` | tuple arithmetic over token-type flags; every grouped statement changes shape |
| `bool_and_to_or` | `grouping.py:511` | flips a recursion guard, so grouping descends where it should stop |
| `return_to_None` | `sqlparse/__init__.py:29` | `parse()` returns nothing, and 64% of the suite parses something |

These are real faults in a parser core, which is exactly where a real fault *is* wide. None is
an import error masquerading as 500 test failures. **Decision: `BROKE_SUITE_KILL_RATIO` stays at
0.80.** The threshold never fired across 49 mutants -- the widest was 0.76 (385/507) -- so
lowering it would have discarded the three most informative faults in the pilot. The kill-ratio
distribution is reported rather than tuned away.

The one timeout was a boolean flip at `grouping.py:336` that turned a loop condition into an
infinite loop. It hit the 180s wall, was recorded as `timed_out`, and `USABLE_STATUS` kept it
out of the dataset. A mutant whose suite never terminates has no labels, not zero labels.

#### C0.6 · provenance was verified and then forgotten

The interpreter guard from 2026-09-02d runs before every harvest and refuses to proceed unless
the subject package resolves into the checkout. It logged its result and kept nothing. Ten
minutes after a run, the strongest claim anyone could make about `mutants.jsonl` was that a log
line had once said `provenance OK`. The label file itself is indistinguishable from one produced
against an installed copy of the package -- same test IDs, same shape, all labels 0.

- `MutationHarness.write_summary()` persists the run summary as `harvest_summary.json` beside
  `baseline.json` and `mutants.jsonl`. `run()` calls it; `harvest_one()` rewrites it with
  `wall_clock_seconds` and `screened_commit_sha`, the two fields only the driver knows.
- `build_real_dataset.load_provenance()` refuses a harvest with no summary, with
  `verified: false` (quoting the recorded reason), with a summary belonging to another
  repository, with no named module, or whose module path resolves outside the checkout.
  Comparison is separator- and case-insensitive, because Windows returns both.
- The manifest carries each repository's provenance block plus a verified-for-every-repo flag
  computed from those entries. A reader of the CSV can see which interpreter produced the
  labels without going back to the harvest directory.

The pilot ran before this existed. Its two summaries were **backfilled by re-running the same
probe** against the same checkouts and venvs, and say so in a `backfilled_after_harvest` field;
they are not transcribed from the log. Full harvests write the file during the run.

#### Measured cost of the dataset build

| quantity | pilot (21,293 rows) | projected (584,500 rows) |
|---|---|---|
| wall clock | 2-6 s | ~1-3 min |
| peak working set | 181 MiB | ~740 MiB (largest single repo) |
| CSV on disk | 8.5 MiB (419 B/row) | **~233 MiB** |
| tracked `data/harvest/` | 0.16 MiB | ~5 MiB |

A row costs ~2.7 KiB as a Python dict and ~360 B once pandas types it, so the intermediate
dominates. `del rows` after each frame is built bounds the peak at one repository's dicts
instead of two consecutive ones -- enough that the largest subject (validators, 895 tests x 250
mutants = 223,750 rows) fits comfortably. The 233 MiB CSV stays out of git under the existing
`data/processed/*` rule; the manifest is what gets committed.

**The harvest cost model is optimistic, and the reason is not bookkeeping overhead.** Screening
estimates hours as `clean suite_duration x n_mutants`. Harvesting runs *mutated* suites, and a
mutated suite is not the same suite:

| repo | clean suite | median mutant suite | slowest mutant | per-mutant bookkeeping |
|---|---|---|---|---|
| sqlparse | 2.65 s | 7.4 s (**2.8x**) | 180 s (timeout wall) | 0.7 s |
| pyjwt | 9.48 s | 9.8 s (1.03x) | 17.5 s | ~0 s |

Failing tests cost more than passing ones -- tracebacks to format, fast paths not taken -- and
one non-terminating mutant burns the entire 180 s timeout, which alone was 42% of sqlparse's
pilot wall clock. Process spawn and byte-exact file restoration together account for 0.7 s per
mutant, so they are not where the time goes.

The multiplier therefore is not a constant: it tracks how much of a suite a typical mutant
breaks and how gracefully it breaks. pyjwt, whose runtime is crypto-bound, is unaffected;
sqlparse, whose runtime is parse-bound, nearly triples. Two repos are too few to fit a
correction, and `suite_duration` is recorded per mutant, so the 250 x 5 harvest supplies the
real distribution and `MAX_HARVEST_HOURS_PER_REPO` gets re-checked against it. At sqlparse's
2.8x every subject still clears the 1.0 h budget.

### 2026-09-02d · C0.3 + C4.4 landed · the harvest would have labelled the wrong package

**Test suite: 290 -> 321 passing** (31 new: 10 harness, 4 executor, 7 harvest driver, 10 screener).

**G0 restored: 5 stage-3 survivors** — pathspec, pyjwt, sqlparse, tabulate, validators.
Estimated full harvest at 250 mutants/repo: **1.2h**.

#### The bug that would have produced a clean-looking dataset of zeros

`MutationHarness` ran the subject suite under `sys.executable` — this project's own
interpreter — while the screener had built a dedicated venv per repository with
`pip install -e .`. Nothing about that arrangement raises. The suite runs, tests pass,
mutants get labelled; they are simply labelled against **a different copy of the
package** than the one being mutated. Every mutant looks harmless, every label comes
out 0, and no random number is involved anywhere. It is the same class of failure as
the fabricated data this sprint exists to remove, arrived at by a different route.

Not hypothetical. Measured, on the accepted set:

| repo | ambient interpreter | screened venv |
|---|---|---|
| sqlparse | imports `site-packages/sqlparse` | imports the checkout |
| pyjwt | 365 passed, 4 skipped | 221 passed, 148 skipped |
| tabulate | 356 passed, 10 skipped | 306 passed, 60 skipped |
| pathspec | not importable at all | imports the checkout |

`sqlparse` is installed in this project's ambient environment, so harvesting it would
have mutated `data/repos/sqlparse/sqlparse/*.py` and measured the released package.
The flat-layout repos only appeared to work by accident: `python -m pytest` puts the
working directory on `sys.path`, which makes a root-layout checkout importable without
being installed. A src-layout repo has no such accident available.

Three fixes, one per layer:

1. `SafeTestExecutor` takes `python_executable`. A path that does not exist is fatal —
   falling back to `sys.executable` is precisely the failure the argument prevents, and
   it would happen silently at the point where hours of compute begin.
2. `MutationHarness.verify_import_provenance()` runs before the baseline. It imports
   each root under the candidate interpreter, from a temporary directory outside the
   repository, and requires every `__file__` to resolve inside `repo_root`. Namespace
   packages and unimportable names are rejected too. The record goes into the harvest
   manifest, so a dataset can be traced to the interpreter that produced it.
3. `harvest_mutations.py` resolves `data/repos/.venv_<name>` for every repository
   before starting any of them, and imports the screener's own `venv_python` rather
   than re-deriving the path — if the screener renames a venv, the driver follows
   instead of pointing at nothing.

#### `import_roots` read the directory, not the package

The first version of the guard rejected `validators`, which is correctly installed.
Its source lives in `src/validators/`, and upstream ships a tracked docstring-only
`src/__init__.py` (commit 70de324), so reading the directory name as the import root
demanded that `import src` succeed. It never does; the installed distribution exposes
`validators`. Roots are now derived from the discovered source *files*, with a leading
container directory (`src`, `lib`, `sources`) stripped and a bare `__init__.py`
ignored. All three layouts now resolve: `sqlparse/sql.py -> sqlparse`,
`src/validators/uri.py -> validators`, `parse.py -> parse`.

#### A skipped test is not a failed test, so a missing extra shrinks the dataset silently

Chasing the interpreter differences above found the reason for them: the screener
installed the repository plus pytest, and nothing a repository declared as an *extra*.
An absent optional dependency does not fail a test — it skips it, and a skipped test
never enters the baseline universe. So the cost was invisible:

* **pyjwt** ran **221 of its 369 tests**. Its `crypto` extra (cryptography) unlocks the
  other 148 — 40% of the suite, absent from the labels with nothing reported anywhere.
* **validators** had 17 tests failing on a clean checkout. All 17 were the eth-address
  tests wanting `crypto-eth-addresses`; installing it takes the repo to **895 stable,
  0 failing**, and the suite gets *faster* (1.82s -> 1.44s) because a traceback costs
  more than a pass.
* **tabulate** gained its `widechars` tests (3.42s -> 4.23s).

The screener now installs every declared extra except tooling, from both metadata homes
(`project.optional-dependencies` and `[options.extras_require]`), plus PEP 735
`dependency-groups` named test/tests/testing — pyjwt keeps its test requirements in one
of those, which `.[name]` cannot reach at all. Tooling is judged by the name's leading
word, so `crypto-eth-addresses` is installed while `dev-docs` and `type_checking` are
not. The deny-list exists for the reason `requirements-dev.txt` is still skipped: a
stale `pytest-flake8` registers a collect hook and aborts the whole run.

Failures are non-fatal by design — pathspec declares `hyperscan` and `re2`, which have
no wheels on every platform — and what was installed is recorded per repo in the
report.

**Cost effect:** pyjwt's suite went 1.94s -> 8.42s, or 0.58h of the 1.0h per-repo
budget. Still inside the gate, but the margin is now worth watching, so a test asserts
the committed report satisfies every gate it claims to pass.

#### A note that outlived its fact

Report entries carry forward whole, so after the extras change `validators` still
advertised `failing_on_clean: [...17 eth tests...]` beside `n_failing_on_clean: 0`.
Stage-2 notes are now tagged and dropped when stage 2 re-runs. A note contradicting the
field next to it is worse than no note.

#### Standing lesson from this session

Two things about a pipeline can be true at once: every number in it was measured, and
every number in it is wrong. `verify_import_provenance` is the only check here that
would have caught this one, and it had to be written before the compute was spent —
after the fact, a dataset of honest zeros is indistinguishable from a dataset of
fabricated ones.


### 2026-09-02c · C2 landed · the ground-truth pipeline runs end to end · stage 3 added

**Test suite: 215 -> 272 passing** (57 new: 39 for C2, 17 for the screener, 1 stage guard).

#### C2 — `scripts/build_real_dataset.py` + `scripts/harvest_mutations.py`

The harvest produces `(mutant, test, killed?)` triples keyed by pytest test ID. The
feature builder turns those into rows: it resolves each test ID to a file path, walks
mutants in a fixed order accumulating causal history, and emits the 32 features plus
the measured label. `harvest_mutations.py` is the CLI driver that was missing — it
reads the screener verdict, refuses anything the screener did not accept, prints the
cost estimate before spending it, and resumes from the per-mutant checkpoint.

**First end-to-end run (smoke, `parse`, 10 mutants):** 883 mutation candidates
generated, 8 harvested + 2 broke_suite, **394 measured kills** -> 776 rows with 216
real failures. Every label in that file is a recorded pytest outcome. This is the
first time the project has produced a labelled row that was not invented.

#### The resolver bug that a unit test would never have found

`resolve_test_id` split the test ID on the *last* `::`. Validating it against the
1138 real testcases in the harvest found **45 of sqlparse's 509 tests unresolvable**:

    test_grouping::test_group_identifier_list[sum(a)::integer, b]

A parametrized ID carries the parameter repr verbatim, and SQL casts contain `::`
themselves, so `rpartition` split inside the brackets. The module path is a
filesystem path with dots for separators and can never contain a colon, so the
*first* `::` is always the real boundary. After the fix: **1138/1138 resolved, 0 label
collisions, 0 nonexistent paths.**

The lesson is the validation, not the fix. The function had tests and passed them;
the shape that broke it only exists in real data.

#### Stage 3 — is test selection even a task in this repository?

The smoke run's rows were unusable for a different reason: **27.84% failure rate**
(G1 band is 1–15%) and **21 of 32 features inert**. `parse` is a single-module
library. One changeable source file means exactly one dependency relationship,
repeated for every row, so a model cannot learn selection — only a per-test
fragility prior. The failure rate is high for the same reason: any mutation to the
one module is reachable from most of the suite.

Rather than proxy this with a file count, the screener now **measures the dependency
features themselves** across every (test file x source file) pair — static analysis,
seconds to run, no install:

| repo | src files | distinct dep vectors | varying dep features |
|---|---|---|---|
| sqlparse | 21 | 84 | 6/6 |
| pathspec | 31 | 86 | 5/6 |
| tabulate | 4 | 12 | 6/6 |
| cachetools | 5 | 6 | 2/6 |
| sortedcontainers | 4 | 4 | 1/6 |
| parse | 1 | 1 | 0/6 |
| inflection | 1 | 1 | 0/6 |

**File count does not predict the outcome** — tabulate varies more with 4 files than
cachetools does with 5. That is precisely why the gate measures the features instead
of guessing at them. `MIN_VARYING_DEP_FEATURES = 3`.

Applied to the existing seven, stage 3 cut **7 -> 3**, below the G0 target of 5. Two
of the rejections turned out to indict the *other* gates, which is why the pool was
re-screened rather than simply widened.

#### The cost gate was measuring the wrong quantity

`MAX_TESTS = 600` bounded test count. But the screener already measures what
actually costs money — one full suite run per mutant:

- validators: 895 tests, **4.96 s** -> 0.34 h at 250 mutants — **rejected**
- sortedcontainers: 296 tests, **11.58 s** -> 0.80 h — **accepted**

Meanwhile `MAX_SUITE_SECONDS = 90` would have let a single repository consume
**6.25 h**. Both replaced with one bound on the measured quantity:

    MAX_HARVEST_HOURS_PER_REPO = 1.0
    MAX_SUITE_SECONDS = MAX_HARVEST_HOURS_PER_REPO * 3600 / PLANNED_MUTANTS_PER_REPO  # 14.4s

`MAX_TESTS` survives at 1200, but now with an honest job: it bounds **rows**
(mutants x universe), not cost.

#### Two recoverable rejections

- **cerberus** lost 247 good tests because one benchmark module imports
  `pytest-benchmark`, which no extra declares. `COMMON_TEST_PLUGINS` now installs
  five ubiquitous plugins unconditionally. All five are additive: none registers a
  `pytest_collect_file` hook, which is why `requirements-dev.txt` is still
  deliberately skipped (a stale `pytest-flake8` there aborts collection outright).
- **pluggy** was recorded as having no tests. Its suite lives in `testing/`.

#### Two more layout bugs, both found by looking at what the screener actually chose

- **jsonschema** was recorded as having no tests. The fallback took the first
  `rglob("tests")` hit, and jsonschema vendors the JSON-Schema-Test-Suite at
  `json/tests/` — thousands of `.json` fixtures, zero `.py` files. The real suite
  sits at `jsonschema/tests/` (9 modules). A directory *named* tests is not a
  suite; the screener now requires content matching pytest's `python_files`
  patterns and prefers the shallowest candidate that has it.
- **packaging** was going to be mutated in `tasks/` — release automation
  (`licenses.py`, `select_pypi_dist.py`) that carries an `__init__.py` and that
  nothing in the suite imports. Every mutant placed there is guaranteed to kill
  nothing, so the budget buys all-negative rows for a reason unrelated to
  selection. A project that declares `src/` has already said where its library
  is, so sibling top-level packages are no longer treated as mutable source.

Neither repo would have failed loudly. jsonschema would have silently vanished
from the pool; packaging would have produced quietly worse data.

#### Two package names collide in this repo

`tests/unit/test_dashboard.py` failed to collect in a full-suite run but passed
alone. There are two packages named `dashboard`: the live Streamlit app at
`./dashboard` (with `utils.py`) and a legacy FastAPI stub at `./src/dashboard`
(without). `pyproject` sets `pythonpath = ["src"]`, so `src` precedes the repo root
for the whole session and `import dashboard` resolves to the stub. The test's
`if str(PROJECT_ROOT) not in sys.path` guard could not help: the root **is** present,
just too late. Fixed by forcing precedence. The duplicate package name is still a
latent trap for anything that imports `dashboard`.

#### Deferred deliberately

`BROKE_SUITE_KILL_RATIO = 0.80` let a mutant that killed 61/97 (63%) of `parse`'s
suite through as a legitimate harvest. A high kill ratio is *inherent* to a
single-module library, so tuning the threshold on `parse` would be tuning on a
degenerate subject. Re-measure after the first multi-module harvest.


### 2026-09-02b · C0 + C4.1 + C4.2 + C4.3 landed

**Test suite: 179 -> 215 passing** (36 new). No regressions. Label guard: PASSED, 104 files scanned.

#### C0 — repo screening complete, G0 met

Twenty candidates screened. Two stages: static (purity, layout, file counts), then
dynamic (install into a fresh venv, size, speed, greenness, determinism over 3 runs).

**7 repositories accepted — target was 5:**

| Repo | Tests | Suite (s) | Source dirs | Flaky | Failing on clean |
|---|---|---|---|---|---|
| cachetools | 333 | 5.68 | `src` | 0 | 0 |
| tabulate | 383 | 3.65 | `tabulate` | 0 | 0 |
| inflection | 467 | 2.75 | `inflection` | 0 | 0 |
| sqlparse | 509 | 7.20 | `sqlparse` | 0 | 0 |
| sortedcontainers | 296 | 11.58 | `src` | 0 | 0 |
| pathspec | 205 | 3.42 | `pathspec` | 0 | 0 |
| parse | 98 | 1.30 | `parse` | 0 | 0 |

Pinned commit SHAs live in `data/repos/screening_report.json`. **Gate G0 is met:**
5+ repos pass every criterion, and all three baseline runs were identical for each.

**Estimated harvest cost at 250 mutants/repo: 2.5 hours total, 0.6 hours per laptop across 4.**
This is the number that makes the whole methodology affordable on the project budget.

Genuine rejections, worth recording so nobody re-screens them:

- **Over the 600-test ceiling:** pyparsing 2156, validators 895, humanize 798, deepdiff ~888, arrow ~708
- **Under the 50-test floor:** toml 24
- **Platform:** schedule calls `time.tzset()`, which does not exist on Windows
- **Not pytest-discoverable:** python-slugify ships `test.py`; the default `python_files`
  patterns (`test_*.py`, `*_test.py`) do not collect it, so the harness could not run it either
- **Missing test-only plugins (fixable, not chased):** cerberus needs `pytest-benchmark`
  for `@mark.benchmark` (247 tests otherwise, in range); funcy has 5 collection errors;
  semver exits with a usage error
- **Detector limitation, not chased:** pluggy keeps its suite in `testing/`

#### Six screener bugs found while screening — each now has a test

The screener decides what the harvest runs against, so a wrong verdict either burns
hours of compute or silently drops a usable subject. Every bug below is pinned by a
test in `tests/unit/test_repo_screener.py`.

1. **Single-file test suites were rejected.** inflection and schedule ship one
   top-level test module, not a `tests/` package. Requiring a directory threw away
   three candidates. Now `test_*.py` / `*_test.py` at top level counts as a test
   location — while a bare `test.py` correctly stays rejected.

2. **Sample data read as a build input.** pyparsing was rejected
   `builds_c_extensions` because of `examples/snmp_api.h` — a file it *parses* in a
   demo. The C-source scan now skips `examples/`, `docs/`, and `tests/`.

3. **Repo `addopts` aborted runs.** `parse` sets
   `addopts = "--cov=parse --doctest-modules"`; without pytest-cov installed, pytest
   exits before collecting anything, reporting zero tests for a healthy 98-test suite.
   Fixed with `-o addopts=` — **and the same flag was added to `SafeTestExecutor`**,
   because otherwise screening timings and collection counts do not describe what
   the harvest actually runs. Coverage instrumentation across hundreds of mutant
   executions is also pure waste.

4. **An interrupted collection looked like a tiny suite.** pytest still writes a
   partial JUnit file when collection aborts, so counting rows called a 2156-test
   project a 1-test project. `run_suite` now returns pytest's exit code, mapped
   through `PYTEST_EXIT_MEANING` (2 = interrupted, 5 = nothing collected,
   4 = usage error, 124 = timeout). Exit 0 and 1 are deliberately absent: some
   tests failing is a finding, not an invalid run.

5. **`pip install -e .[test]` exits 0 for an extra that does not exist.** It only
   warns. My extras loop broke on the first success, so `.[tests]` was never tried
   and humanize screened at 2 tests instead of 798. All extras are now probed.
   (798 is over the ceiling, so humanize is still rejected — but for the true reason.)

6. **Installing `requirements-dev.txt` contaminated the venv.** It pulled an
   incompatible pytest-flake8 and broke `schedule` with a `PluginValidationError`
   that had nothing to do with the repo. Only test-specific requirement files are
   installed now. A self-inflicted failure that looked exactly like a repo defect.

Also added after the fact: a bare `collection_interrupted` verdict is the same
opacity that let bug 4 hide, so rejections now carry the pytest `ERROR` lines that
caused them. That is how cerberus was identified as merely missing a plugin.

**Process note.** One diagnostic run of mine was itself wrong: I ran pytest from
`data/repos` rather than inside each checkout, so it walked into a sibling clone and
reported failures belonging to a different repository. Screening must always run
with `cwd` set to the repo under test.

#### C4.2 — the guard found a third fabrication site on its first run

The manual audit found two. `scripts/check_no_fabricated_labels.py` immediately found
a third: **`src/benchmark/dataset_generator.py`**, which feeds `experiment_runner.py`,
`statistical_plots.py`, and `colab_trainer.py`. So the experiment runner and every
statistical plot also rested on coin-flip labels. The guard earned its keep in one run.

It is an **AST check, not a grep** — it flags an assignment whose target is a
label-ish name and whose value contains a random draw, at any nesting depth. That
distinction matters, because legitimate randomness must survive: mutant sampling,
`np.random.permutation` shuffles, bootstrap resampling, and model seeds are all
allowed. `tests/unit/test_label_guard.py` asserts both directions — six must-catch
cases including the exact original defect, and five must-not-catch cases.

Exemptions are explicit and few: the synthetic generators themselves, and the guard.
Tests are not scanned; a fixture may fabricate freely.

#### C4.1 — the contamination reached the features, not just the labels

`real_repo_miner.py` fed every coin-flip label back through
`self.history_miner.record_run(...)`. So `historical_failure_rate` was itself a
smoothed function of past coin flips — **the feature was contaminated, not merely
the target.** This is why the Historical baseline also landed at exactly 20%: it was
reading a laundered version of the same noise.

Every fallback in that file that invented data now raises instead:

| Was | Now |
|---|---|
| `return [f"commit_{i:04d}" ...]` when git failed | `RuntimeError` |
| `return [f"tests/test_module_{i:02d}.py" ...]` | `RuntimeError` |
| `np.random.exponential(25)` churn for empty diffs | commit skipped, counted |
| `label = 1 if np.random.rand() < 0.70 else 0` | removed, with a comment explaining the defect |
| zero records produced | `RuntimeError` |

The file now contains **no randomness at all**, and asserts that no forbidden label
column ever appears in its output.

`dataset_generator.py` is kept, but demoted: its docstring says
SMOKE-TESTS-ONLY, its constructor refuses to run without
`acknowledge_synthetic=True`, and every row it emits is stamped
`data_origin = "SYNTHETIC_FABRICATED_LABELS"`. Fabricated data is allowed to exist
only when it cannot be mistaken for real.

#### C4.3 — the 0%-recall baseline was never the baseline's fault

The Changed-File selector scored 0% failure recall, which looked like a weak
heuristic. It was not in the selector at all: **`benchmark.py` defaulted
`file_path` to `"src/module.py"` for all 30 commits**, because `features.csv` has no
such column. Every commit claimed to touch the same imaginary file, so nothing ever
matched. Same disease as the labels — fabricated input, silently defaulted.

`changed_file_path` is now a **mandatory** column: absent, it raises `ValueError`
rather than inventing a value. Multi-file commits are supported as `;`-separated
values. The selector itself was rewritten to match stems properly
(`src/auth.py` -> `tests/test_auth.py`), with empty stems filtered out — a blank
stem substring-matches every test in the suite.

#### C4.7 — a fourth fabricated metric

`etr = trr * 0.98`, commented *"time reduction closely tracks test reduction with
slight overhead"*. Execution-time reduction was **inferred from test-count
reduction, not measured** — while `hist_avg_duration` sat unused in the same
dataframe. It is now summed from real per-test durations, and reports
`n/a (no durations)` rather than a number when durations are missing.

#### Standing lesson from this session

Four separate numbers in the results were not measurements: the labels, the
historical-failure-rate feature, the changed-file input, and the time-reduction
metric. Each one had a plausible-looking fallback that produced a plausible-looking
result. **A fallback that invents data is worse than a crash**, because a crash gets
fixed and a fallback gets published. Every one of them now raises.

---

### 2026-09-02 · C1a + C1b landed

**Test suite: 115 -> 179 passing** (64 new). No regressions.

#### Mutation operators verified at scale

Run across the project's own `src/conftest` tree (58 files):

| Metric | Result |
|---|---|
| Total mutants generated | **3,807** |
| Non-compiling | **0** |
| Not-exactly-one-line diffs | **0** |

Operator family distribution: `const` 2415, `cond` 490, `cmp` 352, `arith` 227, `return` 164, `bool` 159.

**Finding — constant mutations are 63% of all candidates**, mostly string literals inside log messages, which almost never kill a test. Unstratified sampling would have produced a dataset dominated by dead mutants. **Fix applied:** `sample_mutants()` draws round-robin across operator families. Covered by `test_sampling_is_balanced_across_operator_families`.

#### End-to-end harvest verified with real labels

Live run against `tests/sample_suite` (real source, real tests, 10-test universe, 12 mutants):

| Mutation | Test actually killed | Correct? |
|---|---|---|
| `arith_*_to_/` `payment.py:11` | `test_fee_calculation` | yes |
| `cond_to_False` `auth.py:14` | `test_authentication_failure` | yes |
| `return_to_None` `database.py:19` | `test_db_delete` | yes |
| `cmp_!=_to_==` `payment.py:14` | both card tests | yes |

Every mutation killed exactly the tests that genuinely cover it — the localized signal an RTS model needs. **Observed failure rate 12/120 = 10%**, inside the G1 target band of 1–15%. Byte-exact file restoration confirmed.

Contrast with the retired synthetic data, where a label was `np.random.rand() < 0.70`.

#### Corrections to this plan discovered while building

1. **Executor class is `SafeTestExecutor`**, not `PytestExecutor`. Section 2.1 corrected.

2. **CRLF hazard (fixed).** Reading source with universal-newline translation and writing it back converts CRLF to LF across the whole file, turning a one-line mutation into a whole-file diff and corrupting the churn features. `read_source`/`write_source` both use `newline=""`. Covered by `test_crlf_line_endings_survive_round_trip`.

3. **`col_offset` is a UTF-8 byte offset, not a character offset.** Splicing is performed on the encoded line, so non-ASCII source stays correct.

4. **`while` conditions must never be forced to a constant** — `while True:` is a guaranteed infinite loop, not a fault. Excluded via `_collect_loop_condition_nodes`. Covered by `test_while_condition_is_never_forced_true`.

5. **Precedence hazard (fixed).** Splicing an unparsed sub-expression can silently change evaluation order. All expression replacements are parenthesized. Covered by `test_precedence_is_preserved_under_splicing`.

6. **Open issue for Component 2 — test ID format.** JUnit XML yields IDs like `tests/test_payment::test_fee_calculation` (no `.py`, derived from `classname` when the `file` attribute is absent). The feature pipeline needs `test_path` + `test_function`, so C2 must map these back. Class-based tests (`tests/test_x::TestClass::test_method`) need handling too. Format is *consistent* between baseline and mutant runs, so labelling is unaffected.

7. **Minor:** harvesting creates `.pytest_cache/` inside the target repo. Harmless, but add it to the G3 `git status` ignore list.

#### Design decisions worth defending in the viva

- **Only killed test IDs are stored** per mutant. Every other test in the baseline universe carries label 0 by construction — compact and lossless.
- **A test vanishing from collection counts as killed.** An import-level break is a real observable failure.
- **Mutants that kill nothing are kept**, not discarded. They are genuine negative examples and reflect the real class imbalance.
- **Mutants killing >80% of the suite are flagged `broke_suite`**, recorded rather than deleted, so the audit trail stays complete and filtering happens at dataset-build time.
- **Test files are never mutated.** Mutating a test changes the oracle, not the code under test.
- **Randomness appears in exactly one place:** which mutants to sample. Never a label.
