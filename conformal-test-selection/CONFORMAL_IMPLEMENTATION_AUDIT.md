# Conformal Implementation Audit

**Scope:** `conformal-test-selection/` (treated as its own repository root).
**Question:** Does this repository actually implement valid conformal prediction, or does it merely use probability/confidence thresholds and call them conformal?
**Method:** Read-only source and artifact inspection. No code was modified, moved, or deleted. `../ConfTest/` was not touched.
**Reviewer stance:** Skeptical. Every claim below is tied to a source line or a report artifact; nothing is inferred from the README's prose.

---

## 1. Executive Verdict

**Classification: `VALID SPLIT CONFORMAL`** — specifically *class-conditional (Mondrian) split conformal prediction on the failing class*, with a correctly implemented **PAC / training-conditional (Vovk 2012) refinement**. Because the controlled quantity is the miss-rate on failing tests, it also has the character of **conformal risk control** (borders on `CASE E`), but the core mechanism is textbook split conformal (`CASE D`).

**Confidence: HIGH.** The nonconformity score, the held-out calibration set with true labels, the finite-sample quantile, and the Beta-based PAC bound are all present and mathematically correct, and the empirically measured coverage on an untouched test split (95.23%) sits just above the certified PAC bound (95.01%) — exactly the relationship a valid implementation predicts.

This is **not** `CASE A` (bare thresholding): there is a real nonconformity score `s = 1 − p` quantiled on a labelled calibration set, not a hand-picked `p > 0.5`. It is **not** `CASE B` (calibration only): calibration (Platt) is a *separate, earlier* layer; the conformal layer is genuinely distinct. It is **not** `CASE C` (incorrect conformal): the quantile rank formula is correct and the PAC correction is valid.

**What it can honestly claim:**

| Claim | Honest? | Required qualifier |
|---|---|---|
| "Uses conformal prediction" | **YES** | — |
| "Has a coverage guarantee" | **YES** | Only under *exchangeability* of calibration↔test; the split is deliberately *temporal*, so this is an assumption that drift can violate. State it. |
| "Catches ≥95% of failures" | **YES, with care** | It is a *per-test* recall guarantee on the *failing class*, in *PAC* form ("with 90% confidence over the calibration draw"). It is **not** a per-commit "95% of broken builds fully caught" claim — that number is only **65.2%**. |
| "Safely abstains under uncertainty" | **NO** | There is no abstain→run-full-suite state. Only a `min_tests` floor and an advisory `degraded` flag exist. |

---

## 2. Evidence Table

| File | Function / lines | Claimed | Actual behaviour in code | Valid? | Evidence | Required fix |
|---|---|---|---|---|---|---|
| `models/conformal.py` | `nonconformity_scores` (242–252); `ConformalSelector.select` (225–235) | Nonconformity score `s = 1 − p` | Returns `1.0 - p`; selects when `1 - p <= threshold` (i.e. `p ≥ 1 − threshold`) | ✅ Genuine nonconformity score, not a fixed prob cut-off | Score defined and applied identically at fit and serve time | None |
| `models/conformal.py` | `marginal_rank` (82–112) | Split-conformal quantile | `k = ceil((n+1)(1−α))`, k-th smallest score; clips to n with a warning when infeasible | ✅ Correct finite-sample split-conformal rank | Line 101 | None |
| `models/conformal.py` | `pac_rank` (115–167), `_achieved_pac_coverage` (170–187) | PAC / training-conditional coverage (Vovk 2012) | Smallest `k` with `BetaInv(δ; k, n+1−k) ≥ 1−α` via binary search; coverage of a fixed order-statistic threshold is `Beta(k, n+1−k)` | ✅ Mathematically correct tolerance-region bound | Lines 146–159, 154 | None |
| `models/conformal.py` | `build_selector` (255–377) | Class-conditional (Mondrian) conformal | Filters `labels == 1` before scoring (295–301) → guarantee is on the *failing* class = failure recall; raises if no failing rows | ✅ Correct label-conditional construction | Lines 295–305 | None |
| `models/conformal.py` | `build_selector` (338–347) | Fails safe if PAC unattainable | Falls back to marginal and logs "The %d%%-confidence claim does NOT hold" at `LOGGER.error` | ✅ Honest degradation, not silent | Lines 341–345 | None |
| `models/conformal.py` | `fit_conformal` (531–638) | Coverage measured on held-out data | Model+calibrator+threshold fit on train/val; `selection_metrics(y_test, p_test, …)` on **test** (585) | ✅ Real held-out measurement, not circular | Lines 566–585 | None |
| `models/conformal.py` | `fit_conformal(calibration_split="val")` (535, 566–569) | — | Nonconformity quantile taken on `val`, the **same split** Platt was fit on (`calibrate.py:298`) | ⚠️ Minor optimism: violates strict independence of scoring-fn vs calibration set | Docstring 539–543 defends it (2-param map); test coverage corroborates | P2: carve a 4th split, or use `--calibration-split test` for an independent check |
| `models/calibrate.py` | `calibrate` (252–365) | Platt calibration + ECE | Sigmoid on logit of raw score (100–101); ECE headline taken on **test** (324–325), val flagged optimistic | ✅ Correct; honest about which split is reported | Lines 322–326 | None |
| `data/preprocess.py` | `temporal_group_split` (128–183) | Temporal, leak-free split | Commits ordered by earliest timestamp, cut 70/15/15; no commit straddles boundary; preprocessor fit on **train only** (451) | ✅ Prevents change-signal leakage; realistic deployment split | Lines 157–177, 451 | None (but see §4 — temporal ⇒ exchangeability caveat) |
| `models/train.py` | `build_model` (176–215) | XGBoost, imbalance-aware | `XGBClassifier` with `scale_pos_weight=auto` (n_neg/n_pos), early stopping on val | ✅ Sound; raw output explicitly *not* thresholded downstream | Docstring 18–22 | None |
| `models/pipeline.py` | `predict` (181–262), `guarantee_statement` (264–298) | Serving applies the rule + honest guarantee | `min_tests` floor only adds tests (217–220); `max_tests` cap can drop a selected test → sets `holds=False` (293–295) | ✅ Cap honestly voids the guarantee; ⚠️ no full-suite abstention | Lines 222–224, 293–295 | P2: see §5 |
| `reports/evaluation.json` | — | recall 0.9523, sel 0.2525, ECE 0.0058, cov 0.9523, cost 0.7475, all pass | Values present and internally consistent with `conformal_report.json` | ✅ Artifacts match each other | evaluation.json 13–69 | None |

---

## 3. Calibration Audit

**Layer:** `models/calibrate.py`. **Method:** Platt scaling (`ScoreCalibrator`, method `"platt"`), a `LogisticRegression(C=1e10)` fit on the *logit* of the raw XGBoost positive-class score (`calibrate.py:97–101`). Isotonic is available as an alternative (103–104). `config.yaml:29` selects `platt`.

- **Fitted on held-out data:** the calibrator is fit on the **validation** split (`calibrate.py:287, 298`) — data the base model never trained on (early stopping merely *watched* val). Correct.
- **ECE is real and honestly reported:** `expected_calibration_error` (144–211) bins into 15 equal-width confidence bins and returns the sample-weighted mean |mean_predicted − observed| gap, plus MCE (the worst bin). Crucially, the **headline `ece_after` is the *test*-split figure** (`calibrate.py:324–325`), with an explicit comment that the val ECE is optimistic because the map was fit on it. Measured test ECE = **0.005795** (`evaluation.json:30`), well under the 0.08 gate.
- **Why it matters here (not cosmetic):** the module docstring (1–10) correctly notes conformal stays *valid* under miscalibration but its prediction sets bloat; calibration is what keeps the selected subset small. This is the right reason to calibrate, stated correctly.

**Verdict:** Calibration is genuine, held-out, and honestly measured. It is a *separate* layer from conformal — the repo does **not** confuse the two.

---

## 4. Conformal Audit

This is the crux. The implementation is genuine split conformal, with three points a skeptic must weigh.

**4.1 The mechanism is correct.**
- Nonconformity score `s = 1 − p_calibrated` (`conformal.py:242–252`). Low score ⇔ model confidently expects failure ⇔ a truly-failing row *conforms*.
- Calibration scores are taken over **failing rows only** (`class_conditional`, `build_selector:295–301`). This makes the guarantee **label-conditional**: `P(selected | test truly fails) ≥ 1 − α`. That is precisely *failure recall*, so the "coverage = recall over failures" framing (`selection_metrics:426–428`) is not a rhetorical trick — it is the correct reading of a Mondrian conformal guarantee on the positive class.
- Threshold = k-th smallest sorted score (`build_selector:305, 316/332`). Marginal `k = ceil((n+1)(1−α))` is the standard split-conformal quantile. **Correct.**

**4.2 The PAC refinement is correct and non-trivial.**
`pac_rank` (115–167) finds the smallest `k` with `BetaInv(δ; k, n+1−k) ≥ 1−α`. The coverage of a fixed threshold placed at the k-th order statistic is a `Beta(k, n+1−k)` random variable over calibration draws; its δ-quantile is therefore a valid `(1−δ)`-confidence *lower bound* on future coverage (Vovk 2012, training-conditional validity). With n = 4578 failing calibration rows, α = 0.05, δ = 0.10, the code selects rank 4369, certifying **0.95014** coverage (`conformal_report.json:9–12`). This is a genuine tolerance region, not a marginal average dressed up as a guarantee.

**4.3 Empirical corroboration.**
Coverage is measured on the **untouched test split** (`fit_conformal:566–585`): 3055/3208 failing tests selected = **0.95231** (`conformal_report.json:38`, `evaluation.json:19`). That the empirical test coverage (0.9523) lands just above the certified PAC bound (0.9501) is the signature of a correctly-behaving conformal predictor. A `val_tuned_threshold` baseline (`evaluation.json:92–101`) that simply tunes a cut-off to hit 95% recall *on val* undershoots on test (0.9498, a 0.000187 shortfall) — a concise demonstration that the conformal construction is doing something a naive threshold does not.

**4.4 Where a skeptic must push back (real caveats, none fatal):**

- **(P1) Exchangeability vs. a temporal split.** Split-conformal validity requires calibration and test to be *exchangeable*. The split is deliberately **temporal** (`temporal_group_split`), so val and test are *not* exchangeable under any real drift. The finite-sample guarantee is therefore an **assumption that drift can break**, not an unconditional theorem. The docstring (`conformal.py:1–51`) names exchangeability as the assumption but does not foreground that its own temporal design puts that assumption under stress. The strong empirical match (§4.3) is *evidence drift is mild on this corpus/time-window*, not a proof it will hold on future data or another project. **This must be stated wherever the guarantee is claimed.**
- **(P2) Calibration split reused for the conformal quantile.** `fit_conformal` defaults to `calibration_split="val"` (535), the same split Platt was fit on. Strict split conformal wants the nonconformity scores computed under a predictor whose fitting never saw those rows; here the 2-parameter sigmoid did. The induced optimism is small (defended at 539–543) and the *test* measurement is untainted, but it is a real deviation from the textbook recipe. The in-sample `calibration_metrics` coverage (0.9543, `conformal_report.json:80`) is optimistic by construction and should never be quoted as evidence — only `test_metrics` should.
- **(Nuance) Per-test vs. per-commit.** The guarantee is marginal over failing *tests*. A CI owner cares whether *every* failing test in a push is caught. The code computes this honestly: **commit_full_catch_rate = 0.652**, any-catch = 0.956 (`conformal_report.json:45–46`). So "95% of failures caught" is true per-test but becomes "≈65% of broken builds fully caught" per-commit. Do not conflate them.

---

## 5. Test-Selection Safety Audit

The intended safety story includes *"abstain and run the complete test suite when uncertainty is high."* That state **does not exist** as implemented. What exists instead:

- **`min_tests` floor (safe).** `pipeline.predict:217–220` and `_apply_budget:474–479` top a commit up to `min_tests` (config: 10) with its highest-risk *unselected* tests. This can only **add** tests, so it never reduces coverage. It is a budget floor, not an abstention: it does not run the *full* suite, and it is unrelated to model uncertainty.
- **`max_tests` cap (can break the guarantee).** `pipeline.predict:222–224` / `_apply_budget:481–484` drop the lowest-risk *selected* tests when the count exceeds `max_tests` (config: 1000). Dropping a selected test can remove a failing one, voiding coverage. The code is **honest** about this: `guarantee_statement:293–295` sets `holds=False` with the note *"max_tests removed selected tests; the coverage guarantee no longer applies."* Good — but it is still a live footgun if operators raise selection above the cap.
- **`degraded` flag (advisory only).** When a caller supplies `< min_feature_completeness` (config: 0.30) of the modelled features, the rest are median-imputed and the response is flagged `degraded=True` with a warning (`pipeline.py:228–256`). **It does not force the full suite** — it still returns the conformal selection computed on a mostly-imputed vector. So under high input uncertainty the system answers anyway and merely labels the answer suspect.

**Consequence:** there is no state in which the system says "I am too uncertain; run everything." The nearest safe behaviour is the `min_tests` floor, which is a fixed budget, not uncertainty-triggered. Any claim of "safe abstention to the full suite" is **unsupported by the code**.

Two genuine safety strengths worth crediting: (a) the serving path re-`reindex`es transformed columns to the model's training order (`pipeline.py:173`) rather than trusting column order, preventing a silent feature scramble; (b) imputation uses NaN→median, never literal 0 (`pipeline.py:112–135`), so absent features are not read as real measurements.

---

## 6. Claims Audit

| Public/README-style claim | Supported by code+artifacts? | Precise honest form |
|---|---|---|
| "Conformal prediction" | ✅ Yes | Class-conditional split conformal with a Vovk PAC bound. |
| "Distribution-free finite-sample coverage guarantee" | ⚠️ Partly | True *given exchangeability*; the temporal split makes exchangeability an assumption drift can violate. Say "under exchangeability of calibration and future rows." |
| "95% coverage / 95% of failures caught" | ✅ Yes (qualified) | PAC: with ≥90% confidence over the calibration draw, ≥95% of *failing tests* are selected; empirically 95.23% on the held-out test split. |
| "90% confidence" | ✅ Yes | Certified 0.95014 ≥ 0.95 at δ=0.10 (`conformal_report.json:12`). |
| "≈75% cost reduction" | ✅ Yes | 0.7475 on test (`evaluation.json:19`); only 25.25% of tests selected. |
| "Well-calibrated (ECE < 8%)" | ✅ Yes | Test ECE 0.0058 (`evaluation.json:30`). |
| "Catches broken builds / commits" | ⚠️ Overstated if unqualified | Per-commit *full* catch = 65.2%; any-catch = 95.6% (`conformal_report.json:45–46`). |
| "Safely abstains under uncertainty" | ❌ No | No full-suite fallback exists; only a `min_tests` floor + advisory `degraded` flag. |
| "Guarantee always holds in production" | ❌ No | Voided by `max_tests` capping (code admits this) and unproven under distribution shift / cross-project transfer. |

---

## 7. Exact Fix Plan

Ordered by how much each affects the *honesty* of the guarantee. None are required to call the work "conformal" — they harden claims and close the abstention gap.

- **P0 — none.** No correctness defect makes the conformal claim false. The math is right and the held-out numbers corroborate it.
- **P1 — Foreground the exchangeability caveat.** In the README and in `guarantee_statement` (`pipeline.py:264–298`), add one line: the coverage guarantee holds under exchangeability of the calibration and deployment rows, which a temporal or cross-project shift can violate; empirical test coverage (95.23%) is the evidence it held on this corpus. Do not present the guarantee as unconditional.
- **P1 — Separate per-test from per-commit claims.** Wherever "95% of failures caught" appears, pair it with the per-commit full-catch rate (65.2%) so a reader cannot mistake it for "95% of broken builds fully caught."
- **P2 — Remove the calibration/quantile overlap.** Either carve a dedicated conformal-calibration split distinct from the Platt split, or document that `--calibration-split test` gives an independence-clean check (at the cost of an in-sample coverage read). Continue to quote only `test_metrics`, never `calibration_metrics`, as evidence.
- **P2 — Add a real abstention state.** Introduce an "abstain → return full candidate set" branch triggered when `feature_completeness < min_feature_completeness` (and/or when `max_tests` would drop a selected test), instead of merely flagging `degraded`. This makes the "safe under uncertainty" story true rather than aspirational.
- **P3 — Guard `max_tests`.** When the cap would void coverage, emit the honest `holds=False` (already done) *and* consider refusing to cap below the conformal selection unless the caller opts in, so the guarantee is not silently lost by a config default.
- **P4 — Monitor drift in deployment.** Log realized coverage on labelled post-hoc CI outcomes over time; a sustained dip below the certified bound is the observable signal that exchangeability has broken.

---

## 8. Corrected Project Positioning

> This project implements **class-conditional (Mondrian) split conformal prediction** for regression-test selection. A gradient-boosted model estimates per-test failure risk; Platt scaling calibrates those scores; and a conformal layer converts `1 − P(fail)` into a nonconformity score whose quantile over the *failing* rows of a held-out calibration split yields a selection threshold. The threshold is chosen by a **PAC / training-conditional (Vovk 2012)** rank, so that — *with ≥90% confidence over the calibration draw, and under the assumption that calibration and future rows are exchangeable* — at least **95% of failing tests are selected**. On an untouched, temporally-later test split the rule empirically selects **95.23%** of failing tests while running only **25.25%** of the suite (**74.75% cost reduction**), with test-set ECE of **0.58%**. The guarantee is a statement about *individual failing tests*; at the *commit* level it fully catches **65.2%** of failing pushes and catches at least one failing test in **95.6%**. The guarantee is voided if a `max_tests` cap drops a selected test (the system reports this), and its validity is not proven under distribution shift or transfer to a new project — the temporal split makes exchangeability an assumption to monitor, not a certainty. There is currently **no uncertainty-triggered full-suite fallback**; a `min_tests` floor and an advisory low-feature-completeness flag are the only guards.

That paragraph is defensible line-by-line against the source and artifacts. Anything stronger — "guaranteed to catch 95% of all real-world failures," "safely abstains," "distribution-free with no assumptions" — is not.

---

## 9. Suggested Next Command

Verify the independence-clean coverage read and the drift caveat empirically, without modifying any code:

```bash
# From conformal-test-selection/ — recompute the threshold on an independent split
# and compare the held-out coverage against the default (val-calibrated) run.
python cli.py conformal --calibration-split test   # independence check (in-sample cov)
python cli.py evaluate                             # re-confirm held-out test coverage
```

Then, if adopting the fixes, start with **P1** (wording) in `README.md` and the `guarantee_statement` note — the only changes needed before the project can present its numbers honestly.

---

*Audit complete. Inspection-only; no source was altered. All figures cite `reports/evaluation.json`, `reports/conformal_report.json`, and the named source lines.*
