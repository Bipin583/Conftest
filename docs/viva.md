# ConfTest Major Project Viva Defense & Technical Q&A

## Core Concept in 30 Seconds
"ConfTest is a confidence-calibrated regression test selection system for CI/CD pipelines. Instead
of running a reduced test subset with unquantified risk, it estimates epistemic uncertainty with a
5-seed LightGBM ensemble and calibrates the probabilities with temperature scaling. If the model is
confident it runs only the high-risk tests; if uncertain it falls back to the full suite. On 183
held-out commits that policy caught 100% of failures with zero escapes, and abstained on 97.8% of
commits, so it saved essentially no wall-clock time. The safety mechanism works; the ranker
underneath it is not yet accurate enough to make the saving worth having, and we report that rather
than tuning on the test split."

> Two corrections a viva examiner may catch in older drafts of this deck: the shipped calibrator is
> **temperature scaling, not isotonic regression** (isotonic was fitted and rejected -- lower mean
> ECE, worst-bin error nearly doubled), and there is **no measured 50--80% time saving**. Quote
> 3.11% test reduction / 0.0% wall-clock at 100% recall, or 32.85% reduction at 87.69% recall, and
> say which operating point you mean.

## Top Viva Questions & Crisp Answers

### 1. What is the fundamental research gap ConfTest addresses?
Standard RTS models predict test failure without uncertainty awareness. When an unseen or out-of-distribution code change occurs, standard models silently drop failing tests, causing critical regressions to escape. ConfTest introduces selective prediction with confidence calibration and full-suite fallback.

### 2. Why is raw classifier score not equal to confidence?
Modern non-linear classifiers (like deep trees or neural nets) output uncalibrated scores that do not reflect true posterior probabilities. Calibration adjusts scores such that a prediction with confidence $0.90$ fails $90\%$ of the time
empirically. Measured here: the uncalibrated ensemble has validation ECE $0.0390$ and worst-bin error
$0.1235$; temperature scaling at $T^* = 0.7941$ brings ECE to $0.0242$ (paired $\Delta = 0.0148$, 95%
CI $[0.0030, 0.0393]$).

### 3. Why use temporal splitting instead of K-Fold Cross-Validation?
Software repositories evolve over time. Random K-fold sampling creates future-to-past data leakage. ConfTest uses strict chronological splitting: earlier commits train the model, intermediate commits fit calibration, and future commits evaluate performance.
