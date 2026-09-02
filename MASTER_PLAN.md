# ConfTest — Master Plan (Real Ground Truth Rebuild)

> **This is the single reference document for the rest of the project.**
> Last updated: 2026-09-02 · Status: Phase R — ground-truth pipeline built, harvest next

---

## 0. TL;DR — Where we stand

**The engineering is ~4 months ahead of the original 6-month plan. The science is at zero.**

| Area | State |
|---|---|
| Code | ✅ 70+ modules, 277/277 tests passing, well-architected |
| Feature pipeline | ✅ 34 features (diff / AST / dependency-graph / history) |
| ML + calibration | ✅ LightGBM, isotonic + Platt + temperature, ECE, reliability diagrams |
| Abstention | ✅ Threshold policy, full-suite fallback, policy tuning |
| Ensemble + SHAP | ✅ Built (beyond original plan) |
| API + Dashboard | ✅ FastAPI (7 route modules), Streamlit (5 pages) |
| Docs | ✅ 27 docs, IEEE paper, KTU LaTeX report, viva deck |
| Mutation harness | ✅ 6 operator families, real pytest labels, **run end to end** (776 labelled rows from a 10-mutant smoke harvest) |
| Subject repos | ✅ **5 stage-3 survivors** (G0 met), 0 flaky, harvest = 1.24 h estimated |
| Fabrication guards | ✅ AST label guard in CI; every invented fallback now raises |
| Feature builder | ✅ **C2 done** — test-ID resolver validated on 1138 real testcases, causal history, 32 features |
| **Real dataset** | ⬜ **Pilot green (21,293 real rows, 7.09% failures); the 250x5 harvest is what remains** |
| Published numbers | ❌ **Still rest on fabricated labels until C5 re-runs them** |
| LLM review layer | ⬜ Does not exist (optional, deprioritized) |

**We are not behind. The machinery for real labels now exists and is tested; what remains is
to run it, build the dataset, and re-run every experiment on top of it.**

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

`src/conftest/repository/synthetic_generator.py:123-128` — same thing:

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

### Component 5 — Re-run Everything

All of this code already exists and is tested. Point it at real data and execute:

`train_model.py` -> `calibrate_model.py` -> `tune_policy.py` -> `train_ensemble.py` -> `run_ablation_study.py` -> `run_cross_repo_eval.py` -> `run_statistical_tests.py` -> `run_flakiness_test.py` -> `run_latency_benchmark.py` -> `run_economic_analysis.py` -> `generate_explanations.py`

**Then rewrite:** `reports/*`, the IEEE paper results section, the KTU report, the viva deck.

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
| **G1** | ≥ 5,000 labeled (mutant, test) pairs per repo; failure rate in **1–15%** |
| **G2** | `grep -rn "random"` over the label path returns **zero** hits |
| **G3** | Every mutated file restored — `git status` clean in all harvested repos |
| **G4** | ML beats random baseline with **bootstrap 95% CI excluding zero** |
| **G5** | Reduction @ 95% recall reported with CI, on a repo **never seen in training** |
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
- [x] **C4.3** fix Changed-File baseline — stem matching + mandatory `changed_file_path`
- [x] **C4.7** stop faking ETR — measure it from real durations
- [x] **C2** `scripts/build_real_dataset.py` — resolver validated against 1138 real testcases
- [x] **C2** `scripts/harvest_mutations.py` — the CLI driver that was missing
- [x] **C2** `tests/unit/test_build_real_dataset.py` — 40 tests, all passing
- [x] **C0.2** stage 3 structural screen — dependency-feature variance (see log 2026-09-02c)
- [x] **C0.2** cost gate recalibrated from test count to measured harvest hours
- [x] **C0.3** re-screen the widened pool to restore G0 — **5 stage-3 survivors** (see log 2026-09-02d)
- [x] **C0.4** every subject suite runs under its own screened venv, enforced by an import-provenance guard (see log 2026-09-02d)
- [x] **C0.5** install declared functional extras — pyjwt ran 221 of 369 tests without them (see log 2026-09-02d)
- [x] **C0.6** every harvest writes `harvest_summary.json`; the dataset builder refuses labels whose import provenance is missing, unverified, or points outside the checkout (see log 2026-09-02e)
- [ ] **G1** first real dataset, gates verified
- [x] **C4.4** calibration selection must not pick a method on ECE alone — `src/conftest/models/calibrator_selection.py`, chosen on a validation half-split
- [ ] **C4.5** bootstrap confidence intervals on all headline numbers
- [ ] **C4.6** relabel `synthetic_generator.py` as smoke-test-only, like `dataset_generator.py`
- [ ] **C3** BugsInPy adapter *(P1)*
- [ ] **C5** re-run every experiment on real data

**Definition of done for this sprint:** `data/processed/real_features.csv` exists, every label traceable to a real pytest run, G0–G3 green.

---

## 11. Build log — findings from implementation

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
