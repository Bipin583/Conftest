# Conformal Code Audit

A strict, evidence-based audit of whether the conformal-prediction and coverage
claims made in `DOCUMENTATION.md` (and `README.md`) are actually implemented
correctly in the code and reflected by the shipped artefacts.

- **Scope:** `models/conformal.py`, `models/calibrate.py`, `data/preprocess.py`,
  `models/train.py`, `cli.py` (evaluate path), and `reports/*.json`.
- **Method:** line-by-line reading of the implementation, hand-recomputation of
  the order-statistic ranks, and cross-checking every headline number against the
  committed report JSONs and `data/processed/split_summary.json`.
- **Rules honoured:** no code was modified; no result was invented; every claim
  cites a file, function, and line.

## 1. Executive Verdict

**The core conformal-prediction and coverage claims are correctly implemented and
internally consistent. No P0 defect was found.** The split-conformal quantile,
the PAC (training-conditional) tolerance-region rank, the class-conditional
(Mondrian) restriction to failing rows, and the selection rule are all textbook-
correct, and every headline number in `DOCUMENTATION.md` / `README.md`
reproduces from the committed artefacts and re-derives by hand.

Independently recomputed and confirmed:

- Marginal rank `k = ⌈(n+1)(1−α)⌉ = ⌈4579 × 0.95⌉ = 4351` — matches
  `reports/conformal_report.json` `variants.marginal.rank = 4351`.
- PAC rank (smallest `k` with `BetaInv(δ; k, n+1−k) ≥ 1−α`, `n=4578, α=0.05,
  δ=0.10`) `= 4369`, certified coverage `Beta.ppf(0.10, 4369, 210) = 0.95014` —
  matches `rank = 4369`, `certified_coverage = 0.9501368491`.
- `probability_floor = 1 − threshold = 1 − 0.9618546515 = 0.0381453485` — matches.
- Empirical **test-split** coverage `0.9523067 ≥ 0.95` on 3208 failures / 86469
  rows — clean out-of-sample confirmation.
- Feature counts `39 numeric + 2 categorical = 41 raw → 62 transformed`
  (`39 + 5 repo + 18 mutant_operator`) — matches `split_summary.json`.
- Corpus size `391825 + 83417 + 86469 = 561711` rows — matches.

The findings below are **caveats and write-up precision issues (P1/P2), not
correctness bugs.** They cluster around three honest-but-under-stated points: the
conformal quantile is taken on the same split the Platt calibrator was fit on
(mild training-conditional optimism); the val-tuned-threshold baseline is much
closer to the conformal rule than the narrative implies; and the shipped numbers
come from the *reuse* path, which trusts the upstream corpus's temporal split
rather than performing it in this repo. All three are already partially disclosed
in code docstrings; the fix is to surface them in `DOCUMENTATION.md`.

**Bottom line:** the contribution is real and the implementation is sound. The
project's unusual candour (it reports MCE next to ECE, states the exchangeability
caveat, and marks `holds:false` when `max_tests` voids the guarantee) largely
survives scrutiny. What it oversells is the *size* of the empirical win over a
naive tuned threshold, not the *validity* of the guarantee.

## 2. Evidence Table

| Claim (DOCUMENTATION.md / README.md) | Source of truth | Verdict |
|---|---|---|
| Marginal quantile rank = 4351 | `conformal.py:marginal_rank` L101; `conformal_report.json` `variants.marginal.rank` | ✅ Confirmed (hand-recomputed) |
| PAC rank = 4369, certified 0.95014 | `conformal.py:pac_rank` L131-159, `_achieved_pac_coverage` | ✅ Confirmed (hand-recomputed) |
| Probability floor 0.038145 | `conformal.py:build_selector` (`1−threshold`) | ✅ Confirmed |
| Test coverage 0.9523 (≥0.95) | `conformal_report.json` `test_metrics.empirical_coverage` | ✅ Confirmed |
| Selection rate 0.2525 / cost reduction 0.7475 | `conformal_report.json` `test_metrics` | ✅ Confirmed |
| ECE after ≈ 0.58% | `calibration_report.json` `test.ece_after=0.005795`; `cli.py:cmd_evaluate` L353 | ✅ Confirmed (value); ⚠ see §4 |
| Per-commit full/any catch 65.2% / 95.6% | `conformal_report.json` `commit_full_catch_rate`/`commit_any_catch_rate` | ✅ Confirmed |
| 41 raw → 62 transformed features | `split_summary.json` `n_numeric=39,n_categorical=2,n_model_inputs=62` | ✅ Confirmed |
| cost-matched top-k reproduces conformal | `evaluation.json` baselines; `cli.py:_baselines` | ✅ Confirmed (exact) |
| val-tuned threshold "under-delivers" | `evaluation.json` recall 0.94981, shortfall 0.000187 | ⚠ Directionally true, magnitude oversold (§6) |
| This repo performs the temporal 70/15/15 split | `preprocess.py:preprocess` L411-423 (reuse path) | ⚠ Reuse path trusts corpus split (§5) |
| Calibration lowers ECE | `calibrate.py` module docstring L24-25 vs `calibration_report.json` | ❌ False on test artefacts (§4, P2) |

## 3. Conformal Math Audit

**Non-conformity score.** `ConformalSelector.select` (`conformal.py` L225-235)
computes `scores = 1.0 - p` and returns `scores <= self.threshold`, i.e. it
selects a test iff `p ≥ 1 − threshold = probability_floor`. For a "select the
high-risk failing tests" objective, `s = 1 − P(fail)` is the correct
non-conformity score: a failing test is *conforming* when its predicted failure
probability is high, so its score is low, so it falls under the threshold. ✅

**Marginal quantile.** `marginal_rank(n, alpha)` (`conformal.py` L101):
`k = math.ceil((n + 1) * (1.0 - alpha))`, clipped to `n`. This is the standard
split-conformal order statistic. For `n=4578, α=0.05`:
`⌈4579 × 0.95⌉ = ⌈4350.05⌉ = 4351`. Report `variants.marginal.rank = 4351`,
`threshold = 0.9608406416`. ✅ Exact.

**PAC / training-conditional rank.** `pac_rank(n, alpha, delta)`
(`conformal.py` L131-159) binary-searches for the smallest `k` such that
`beta_dist.ppf(delta, k, n + 1 - k) >= 1 - alpha`, returning `None` if
unattainable. This is the Vovk (2012) tolerance-region result: the coverage of a
threshold placed at the `k`-th order statistic of `n` exchangeable calibration
scores is `Beta(k, n+1−k)`-distributed, and its `δ`-quantile is the
`(1−δ)`-confidence lower bound on coverage. Requiring that lower bound to reach
`1−α` yields the PAC guarantee "with prob. ≥ 1−δ over the calibration draw,
coverage ≥ 1−α". The binary search is valid because the beta quantile is
monotone in `k`. For `n=4578, α=0.05, δ=0.10`: `k=4369`, and
`_achieved_pac_coverage(4578, 4369, 0.10) = Beta.ppf(0.10, 4369, 210) = 0.95014`.
Report: `rank=4369`, `certified_coverage=0.9501368491`, `confidence=0.9`. ✅
Direction check: PAC rank (4369) ≥ marginal rank (4351), as it must be — the
training-conditional guarantee is strictly more demanding. ✅

**Class-conditional (Mondrian) construction.** `build_selector`
(`conformal.py`) restricts the calibration scores to failing rows via
`mask = labels == 1` before taking the quantile, and `n_calibration = 4578`
equals the failing-row count, not the full val size. This correctly yields a
guarantee that is *marginal over the failing class* — i.e. a bound on recall over
failures — which is exactly what `guarantee_statement` reports as
`"scope": "per failing test (marginal over the failing class), not per commit"`
(`pipeline.py` L289). The per-commit numbers (§6) are honestly reported as lower.
✅ The scope claim is accurate and the code matches it.

**PAC-unattainable fallback.** `build_selector` falls back to the marginal rank
with a loud `LOGGER.error` if `pac_rank` returns `None`. Non-silent degradation. ✅

**One real theoretical caveat (P1, not a bug):** split-conformal validity
requires the calibration scores to be exchangeable with future scores *under a
predictor fixed independently of the calibration set*. Here the final scoring
function is `base_model ∘ calibrator`; the Platt `calibrator` was fitted on the
val split (`calibrate.py` L298), and `fit_conformal` then takes the conformal
quantile on that **same** val split (`conformal.py` `fit_conformal`,
`calibration_split="val"`). The val non-conformity scores are therefore mildly
in-sample for the calibrator, so the *certified* 0.95014 carries a small
optimistic bias. The `fit_conformal` docstring explicitly acknowledges and
defends this (a 2-parameter sigmoid over ~83k rows induces negligible optimism),
and — decisively — the **empirical coverage is measured on the untouched test
split** (0.9523), which is genuine out-of-sample evidence. So the deployed rule
is validated cleanly; only the theoretical certified number inherits the bias.
This belongs in `DOCUMENTATION.md`'s guarantee section, where it is currently
absent.

## 4. Calibration Audit

**ECE/MCE definition.** `expected_calibration_error` (`calibrate.py` L144-211)
uses equal-width bins over `[0,1]`, `np.digitize` with interior edges, clips bin
indices to `[0, n_bins-1]`, and returns the sample-weighted mean of
`|mean_predicted − observed|` (ECE) and the worst-bin value (MCE). This is the
standard binned ECE/MCE. Default `n_bins = 15` (from config `ece_bins`). ✅

**Headline is measured on TEST, not val.** `cmd_calibrate` sets `ece_before` /
`ece_after` from the test split (`calibrate.py` L324-325), which is the honest
choice — reporting the val figure would be optimistic since the calibrator was
fitted there. ✅ `cli.py:cmd_evaluate` (L353) then *reads* `ece_after` from
`calibration_report.json` rather than recomputing it, so the evaluation and
calibration reports cannot disagree. ✅

**FINDING P2 (code docstring, false invariant).** The `calibrate.py` module
docstring (L24-25) shows an example asserting `report["ece_after"] <
report["ece_before"]`. On the shipped artefacts this is **false**: test
`ece_before = 0.004979548`, `ece_after = 0.005795282` — calibration *slightly
worsened* aggregate test ECE. The cause is benign: the raw XGBoost scores are
already near-perfectly calibrated in the dominant low-probability bin
(the bulk of the 86469 rows sit far below the 0.038 floor), so a 2-parameter
Platt map cannot improve the aggregate and marginally perturbs it. Both figures
are ≪ the 0.08 gate, so no *claim* breaks — but the docstring states an invariant
the code does not guarantee. Per the audit rules I did **not** edit the code;
this is filed as a P2 doc fix. Note `DOCUMENTATION.md` itself only claims "ECE
≤ 8%" and "≈0.58%", both true, so the user-facing docs are not wrong here — only
the in-code example is.

**MCE persists in the tail (disclosed, not a defect).** Test-after
`MCE ≈ 0.457` (`calibration_report.json`): the high-confidence bins remain badly
miscalibrated (a bin predicting ~0.76 observes ~0.30). These bins are sparse and
sit far above the 0.038 operating floor, so they never affect selection. The code
deliberately reports MCE alongside ECE and the docstring (L17-21) explains
exactly this, so the miscalibration is surfaced rather than hidden. ✅ Worth one
line in `DOCUMENTATION.md`: "calibration is strong where the selector operates
and weak in the sparse high-confidence tail; this is why MCE is reported."

**Calibrator design.** `ScoreCalibrator` (Platt = `LogisticRegression(C=1e10)`
on `_safe_logit(scores)`) is fitted on held-out val with the base model frozen
(`calibrate.py` L52-59, L298), which is the correct setup for preserving the
conformal exchangeability argument. `CalibratedClassifierCV` is only a
cross-check and is off by default. ✅

## 5. Data Split / Leakage Audit

**Harvest path split is correct.** `temporal_group_split` (`preprocess.py`
L128-183) orders commits by their earliest timestamp, cuts at the configured
count ratios, and keeps every row of a commit on a single side of the boundary.
No commit straddles a split, so no change-level or diff-level feature can leak
across the train/val/test boundary. The split is chronological *and* group-aware
by construction. ✅ The reported partition is consistent:
train 849 commits / 391825 rows / 5.02% fail, val 181 / 83417 / 5.49%,
test 183 / 86469 / 3.71% (`split_summary.json`).

**FINDING P2 (provenance precision).** The shipped artefacts were produced by the
**reuse / external** path, not the harvest path. In `preprocess()`
(`preprocess.py` L411-423) the external branch reads pre-split train/val/test
CSVs from the upstream corpus and only *adapts* them (alias/feature mapping); it
does **not** call `temporal_group_split`. So for the numbers actually shipped, the
"chronological group-aware 70/15/15 split" is *inherited from the ConfTest
corpus*, and this repo trusts that the upstream split was done correctly. This is
disclosed in the `preprocess.py` module docstring, but `DOCUMENTATION.md`
§3/§6 read as though this repo performs the split for the shipped run. Tighten the
wording to "performs (harvest) or inherits (reuse) a chronological group-aware
split; the shipped artefacts use the reuse path." No leakage is introduced by the
reuse path itself; the caveat is that upstream mining was **not** re-audited here.

**Preprocessor fit is leakage-free.** `build_preprocessor` is fit on the train
split only (`preprocess.py` L451) and then transforms val/test. Median imputation
and `StandardScaler` statistics therefore never see held-out data. ✅
`OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)` means an unseen
`repo` at serve time maps to all-zeros rather than crashing. ✅

**Base-rate drift corroborates the caveat (not a bug).** The test split's failure
rate (3.71%) is materially below train/val (5.02% / 5.49%). This is the expected
signature of a temporal split and is precisely why the exchangeability caveat
matters: the conformal threshold was fitted on a 5.49%-positive split and deployed
on a 3.71%-positive one. Coverage still held (0.9523), which is reassuring, but —
consistent with the docs' own warning — is not guaranteed to persist under
further drift. Honest and correctly framed.

**Not independently verified.** The per-test history features
(`failure_rate_lifetime`, `hist_prior_failures`, `co_change_frequency`, etc.)
claim "strictly prior observations only." That leakage-freeness lives in the
upstream corpus mining (`data/features.py` harvest path / the ConfTest corpus),
which this audit did **not** read line-by-line. The *architecture* is consistent
(these columns arrive from the corpus and are left absent + imputed in live
extraction — `data/live_extract.py`, `HISTORY_KEYS` of 8), but the temporal
correctness of the mining itself is asserted, not verified here.

## 6. Results Verification

Every headline number re-derives from the committed reports:

- **Coverage / recall = 0.9523067** (`conformal_report.json`
  `test_metrics.empirical_coverage`, echoed in `evaluation.json`). Clears the
  ≥0.92 and ≥0.95 gates — but the 0.95 gate by only **0.23 points**. That thin
  margin is expected: the rule is calibrated to *exactly* 95%, so barely clearing
  it on test is a sign the guarantee is well-tuned, not a red flag.
- **Selection rate 0.2524604, cost reduction 0.7475396** — consistent
  (`1 − 0.2525 = 0.7475`). ✅
- **ECE 0.005795** — `cmd_evaluate` pulls it from `calibration_report.json`; DOC's
  "0.58%" matches. ✅
- **cost-matched top-k reproduces conformal exactly** (`evaluation.json`
  baselines; `cli.py:_baselines` sets `k = round(conformal_rate × n)` and takes
  the top-`k` by probability). Recall and selection rate match the conformal row
  to the digit. ✅ **This validates the project's most important honest claim:
  conformal is not a better *ranker* — at the same budget a plain top-k matches
  it. Conformal's contribution is principled selection of *how many* tests to run
  (the budget itself), with a coverage guarantee attached.** The code and the
  narrative agree on this.
- **val-tuned-threshold baseline** (`_baselines`,
  `cut = np.quantile(val_positive, 1 − target_recall)`, no finite-sample
  correction): recall `0.9498130`, shortfall `0.000187` vs the 0.95 promise
  (`evaluation.json`). **FINDING P1 (magnitude oversold).** The DOC/README frame
  this baseline as one that visibly "under-delivers." Directionally true — it does
  miss 0.95 — but the miss is **0.0187 percentage points** (≈8 failures out of
  3208), at a selection rate essentially equal to conformal's. On this corpus the
  practical gap between "tune a threshold on val" and "run the conformal
  machinery" is marginal. The correct framing is: the val-tuned threshold *usually
  lands close but carries no guarantee and here falls fractionally short*, whereas
  the conformal rule *provides the finite-sample correction and the certificate*.
  The value-add is methodological rigour, not a large empirical delta — and the
  write-up should say so rather than implying a wide gap.
- **Per-commit catch rates 65.19% / 95.56%** (`commit_full_catch_rate`,
  `commit_any_catch_rate`). DOC's 65.2%/95.6% match. The per-failing-test
  guarantee (95%) does **not** translate into every failing test of every commit
  being caught (65%), and the docs say so. ✅
- **Feature and corpus counts** all reconcile (§1, §2). ✅

## 7. Documentation Corrections

1. **`calibrate.py` docstring (L24-25) — P2.** Remove or correct the
   `ece_after < ece_before` example; on the shipped test split ECE rises
   0.004980 → 0.005795. Replace with "ECE stays well within the 0.08 gate;
   calibration mainly reshapes the probability *scale*, and on an already
   well-calibrated dominant bin the aggregate ECE may move either way."
2. **`DOCUMENTATION.md` §3/§6 — P2.** Clarify split provenance: this repo
   *performs* a chronological group-aware split on the harvest path but *inherits*
   the corpus's split on the reuse path, and the shipped artefacts use the reuse
   path.
3. **`DOCUMENTATION.md` guarantee section (§10/§13) — P1.** Add the caveat that
   the conformal quantile is taken on the same val split the Platt calibrator was
   fitted on, so the *certified* 0.95014 carries mild training-conditional
   optimism; the *empirical* test-split coverage (0.9523) is the real evidence.
4. **`DOCUMENTATION.md` baseline discussion — P1.** Replace the "under-delivers"
   framing of the val-tuned threshold with the actual near-parity numbers
   (recall 0.94981 vs 0.95231; selection ~equal) and reframe conformal's edge as
   the finite-sample correction + certificate, not a large recall gain.
5. **`DOCUMENTATION.md` calibration section — P2.** State that calibration is
   strong at the operating range and weak in the sparse high-confidence tail
   (MCE ≈ 0.457), which is why MCE is reported alongside ECE.

## 8. Required Fixes

**P0 (correctness — blocks the claims): none.** The conformal math, the
calibration measurement, the class-conditional construction, the selection rule,
and the leakage-free preprocessor fit are all correct, and the shipped numbers
reproduce.

**P1 (validity caveats that should be surfaced before publication):**
- Fit the conformal quantile on a split **disjoint** from the Platt calibrator's
  fit (or explicitly rest the claim on the test-split empirical coverage and
  document the shared-split optimism). Ref: `conformal.py:fit_conformal`
  (`calibration_split="val"`) + `calibrate.py` L298.
- Correct the val-tuned-threshold narrative to the true near-parity magnitude.
  Ref: `cli.py:_baselines`, `evaluation.json`.
- Add the shared-split / exchangeability caveat to `DOCUMENTATION.md`'s guarantee
  section (currently only in the `fit_conformal` docstring).

**P2 (precision / hygiene):**
- Fix the false `ece_after < ece_before` example in `calibrate.py`'s docstring.
- Clarify harvest-vs-reuse split provenance in `DOCUMENTATION.md`.
- Note the tail MCE explicitly in the calibration write-up.
- (Optional) Independently re-audit the upstream corpus's history-feature mining
  for temporal correctness; it is trusted, not verified, in the reuse path.

## 9. Corrected Contribution Statement

> This project applies **split conformal prediction** to regression-test
> selection. On a chronological, commit-grouped train/val/test partition of a
> 561,711-row corpus, an XGBoost failure-probability model is Platt-calibrated on
> a held-out split and wrapped in a **class-conditional (Mondrian) conformal
> selector** that guarantees, in the **training-conditional (PAC) sense** — with
> probability ≥ 0.90 over the calibration draw — that **≥ 95% of failing tests
> are selected** (certified lower bound 0.95014 at calibration rank 4369 over
> 4578 failing calibration rows). On the untouched test split the rule achieves
> **95.23% coverage of failing tests at a 25.25% selection rate (74.75% fewer
> tests run)**.
>
> The guarantee is **marginal over the failing class (per failing test), not per
> commit** — the per-commit "every failing test caught" rate is 65.2% and the
> "at least one caught" rate is 95.6%. A budget-matched top-k selector reproduces
> the conformal recall and selection rate exactly: **conformal does not rank
> better than the model already does; its contribution is a principled,
> guaranteed choice of how large the test budget must be, with the guarantee
> made explicit and its assumptions stated.**
>
> **Assumptions and limits (do not omit):** validity rests on exchangeability of
> the calibration and deployment rows, which the *deliberately temporal* split
> breaks by design — so the coverage number should be **monitored in deployment,
> not treated as unconditional.** The conformal quantile is currently taken on the
> same split the calibrator was fitted on, so the *certified* figure carries mild
> optimism; the *empirical test-split* coverage is the trustworthy evidence.
> Calibration is strong at the operating range (test ECE ≈ 0.58%) but weak in the
> sparse high-confidence tail (MCE ≈ 0.46). The shipped artefacts use the reuse
> path, which inherits the upstream corpus's temporal split rather than performing
> it here.

## 10. Suggested Next Command

Regenerate the conformal artefact with the quantile split held disjoint from the
Platt fit, then re-evaluate on the untouched test split to confirm coverage is
unchanged once the shared-split optimism is removed:

```bash
python cli.py conformal   # re-fit selector (ideally on a calib split ≠ the Platt-fit split)
python cli.py evaluate    # re-verify coverage/selection gates on the test split
```

Then apply the P1/P2 documentation edits from §7. No code correctness fix is
required to defend the coverage claim — the empirical test-split coverage already
substantiates it.

