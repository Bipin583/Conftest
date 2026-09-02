# ConfTest — Master Plan (Real Ground Truth Rebuild)

> **This is the single reference document for the rest of the project.**
> Last updated: 2026-09-02 · Status: Phase R starting

---

## 0. TL;DR — Where we stand

**The engineering is ~4 months ahead of the original 6-month plan. The science is at zero.**

| Area | State |
|---|---|
| Code | ✅ 70+ modules, 115/115 tests passing, well-architected |
| Feature pipeline | ✅ 34 features (diff / AST / dependency-graph / history) |
| ML + calibration | ✅ LightGBM, isotonic + Platt + temperature, ECE, reliability diagrams |
| Abstention | ✅ Threshold policy, full-suite fallback, policy tuning |
| Ensemble + SHAP | ✅ Built (beyond original plan) |
| API + Dashboard | ✅ FastAPI (7 route modules), Streamlit (5 pages) |
| Docs | ✅ 27 docs, IEEE paper, KTU LaTeX report, viva deck |
| **Data / labels** | ❌ **100% fabricated — invalidates every reported number** |
| LLM review layer | ⬜ Does not exist (optional, deprioritized) |

**We are not behind. We have one fatal flaw to fix, then we re-run everything we already built.**

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
- [ ] **C0** `scripts/screen_repos.py` — screener + report
- [ ] **C0** screen 12 candidates -> select 5
- [ ] **C4.1** purge synthetic fallbacks from `real_repo_miner.py`
- [ ] **C4.2** CI grep gate against `random` in label paths
- [ ] **C4.3** fix Changed-File baseline
- [ ] **C2** `scripts/build_real_dataset.py`
- [ ] **G1** first real dataset, gates verified

**Definition of done for this sprint:** `data/processed/real_features.csv` exists, every label traceable to a real pytest run, G0–G3 green.

---

## 11. Build log — findings from implementation

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
