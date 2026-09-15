---
marp: true
theme: default
paginate: true
header: "ConfTest: Confidence-Calibrated Selective RTS | KTU B.Tech CSE 2026"
footer: "Bipin B | APJ Abdul Kalam Technological University"
style: |
  section {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
  }
  h1, h2 {
    color: #38bdf8;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background-color: #1e293b;
    color: #38bdf8;
  }
---

# 🚀 ConfTest
## Confidence-Calibrated Selective Regression Test Selection via Uncertainty-Aware Deep Ensembles

**Candidate:** Bipin B (Register No: KTU-CSE-2026)  
**Guide:** Assistant Professor, Dept. of Computer Science & Engineering  
**Institution:** APJ Abdul Kalam Technological University (KTU)  
**Academic Year:** 2025–2026  

---

# 1. The CI/CD Bottleneck
- Continuous Integration demands frequent code commits and fast feedback loops.
- **The Problem:** Modern regression test suites take 45–180 minutes to run.
- **Consequences:** Developer context switching, blocked PR pipelines, and high cloud compute bills.

---

# 2. Existing Approaches & Their Limitations
- **Dynamic RTS (Coverage/Bytecode):** 20–50% runtime instrumentation overhead; brittle to config edits.
- **Static RTS (File-based):** Overly conservative; selects up to 80% of unchanged suites.
- **Machine Learning (ML-RTS):** Fast, but **overconfident** on out-of-distribution commits, causing silent bug escapes.

---

# 3. The Overconfidence Dilemma in ML-RTS
- Standard gradient boosted trees output uncalibrated risk scores.
- Raw ensemble scores do not match empirical failure frequencies: on our validation split the
  uncalibrated model has **ECE $0.0390$** and **worst-bin error $0.1235$**.
- **Result:** a confidence threshold set on raw scores does not mean what it appears to mean, so an
  abstention gate built on it cannot be trusted.

---

# 4. The ConfTest Vision
ConfTest bridges the gap between **ML efficiency** and **static safety**:
1. **Uncertainty Quantification:** 5-seed Deep Ensemble measuring epistemic divergence ($\sigma$).
2. **Post-Hoc Probability Calibration:** Temperature Scaling, $T^* = 0.7941$, ECE $0.0390 \to 0.0242$.
3. **Selective Prediction Policy:** Fast subset execution when confident; automatic full-suite fallback when uncertain.

---

# 5. End-to-End System Architecture
```
[ Commit Diff ] --> [ 32-Feature Mining ] --> [ 5-Seed Ensemble ]
                                                      |
[ Selective Policy ] <-- [ Temperature Scaling ] <----+ (p, sigma)
        |
        +---> High Confidence  ==> FAST MODE (Top-K Subset)
        +---> High Uncertainty ==> SAFE FALLBACK (100% Suite)
```

---

# 6. Canonical 32-Feature Pipeline
- **Diff & Churn (12 Features):** Lines added/deleted, file counts, churn velocity.
- **AST Complexity (6 Features):** Functions, classes, cyclomatic delta, asserts.
- **Static Call-Graph (6 Features):** Direct imports, NetworkX shortest path depth, coupling coefficients.
- **Historical Telemetry (8 Features):** Recent-10 failure rates, flakiness scores, prior executions.

---

# 7. AST & NetworkX Call-Graph Modeling
- Parses Python AST trees without running code.
- Builds static directed dependency graphs $G = (V, E)$.
- Computes graph geodesic depth from changed functions to test entry points:
  $$\text{depth}(f_{\text{changed}}, t_j) = \text{shortest\_path}(G, f_{\text{changed}}, t_j)$$

---

# 8. Anti-Leakage Temporal Dataset Splitting
- Software history has a strict time arrow.
- **No Random K-Fold CV:** Prevents future telemetry leaking into past predictions.
- **Temporal Split:** 70% Train, 15% Validation, 15% Test strictly ordered by commit timestamp.

---

# 9. Deep Ensemble Epistemic Uncertainty ($\sigma$)
- Evaluates $M=5$ distinct LightGBM models initialized with different random seeds.
- **Mean Failure Probability:**
  $$\bar{p}(c, t) = \frac{1}{M}\sum_{m=1}^M f_{\theta_m}(c, t)$$
- **Epistemic Disagreement:**
  $$\sigma(c, t) = \sqrt{\frac{1}{M}\sum_{m=1}^M \left(f_{\theta_m}(c, t) - \bar{p}(c, t)\right)^2}$$

---

# 10. Temperature Scaling Calibration
- Minimizes Negative Log-Likelihood on held-out validation logits:
  $$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{val}}}\sum_{k=1}^{N_{\text{val}}} \left[ y_k \log \hat{p}_k(T) + (1-y_k)\log(1 - \hat{p}_k(T)) \right]$$
- Fitted $T^* = \mathbf{0.7941}$ by bounded search over $\log T$ (17 iterations; validation NLL
  $0.19290 \to 0.18345$ over $83{,}417$ rows). $T^* < 1$: the ensemble is *under*-confident here.
- Reduces validation ECE from **0.0390 to 0.0242** (paired $\Delta = 0.0148$, 95% CI
  $[0.0030, 0.0393]$, 2,000 bootstraps resampled by mutant).
- **Isotonic regression was fitted and rejected:** lower mean ECE ($0.0166$) but worst-bin error
  rose $0.1235 \to 0.2098$ ($+0.0863$, CI $[0.0299, 0.1135]$). The gate keys off the tail.

---

# 11. Cost-Optimal Selective Prediction Policy
- **Fast Subset Execution:** Triggered when epistemic uncertainty $\sigma < \tau_{\text{abstain}}$ and confidence $\max \hat{p} \ge \tau_{\text{conf}}$.
- **Safe Full-Suite Fallback:** Triggered when uncertainty is high or large architectural refactorings are detected ($\text{files} > 15$).

---

# 12. Observed Safety at the Shipped Point
- If uncertainty crosses the tuned gate, the policy refuses subset selection and executes the entire suite.
- On the 183 held-out commits, the shipped operating point observed **100% failure recall and zero escaped commits**. This is an empirical result, not a formal guarantee for unseen repositories or future commits.

---

# 13. Model Explainability: Tree SHAP ($\phi_i$)
- Generates exact local Shapley attributions for every test prediction:
  $$\hat{p}(x) = \phi_0 + \sum_{i=1}^{32} \phi_i(x)$$
- Provides transparent explanations to developers in CI pull request comments.

---

# 14. Rule-Based Natural Language Developer Cards
- Translates continuous feature attributions into readable cards:
  - ⚠️ *"Direct import of modified module detected (Shortest path: 1 step)"*
  - 📈 *"Test failed in 3 of the last 10 runs (Elevated historical risk)"*

---

# 15. Empirical Benchmark Comparison (8 RTS Strategies)

183 held-out commits, 86,469 (commit, test) pairs, labels by execution.
Source: `reports/baseline_comparison.csv`

| RTS Strategy | Recall | Test Red. | Time Red. | Escaped commits |
| :--- | :---: | :---: | :---: | :---: |
| **Full Suite** | 100.0% | 0.0% | 0.0% | 0 |
| **Random-K (25%)** | 25.2% | 75.1% | 78.7% | 133 |
| **Changed File** | 42.3% | 83.6% | 86.1% | 69 |
| **Static AST / Dependency** | 38.6% | 75.6% | 68.2% | 94 |
| **Historical Failure** | 48.4% | 75.1% | 70.0% | 99 |
| **Uncalibrated ML** | 51.8% | 75.1% | 27.7% | 69 |
| **Calibrated No-Abstain** | 51.8% | 75.1% | 27.7% | 69 |
| **ConfTest (Ours)** | **100.0%** | **3.1%** | **0.0%** | **0** |

**Read this honestly:** ConfTest is the only row with zero escapes *and* the row that saves least.
It abstains on 97.8% of commits. Rows 6 and 7 are identical because temperature scaling is monotone
and cannot change a top-$k$ ranking.

---

# 15b. The 95% Recall Gate Is Not Met

| Operating point | Test Red. | Recall | Abstention | Escaped commits |
| :--- | :---: | :---: | :---: | :---: |
| **A — shipped** ($\tau_\text{abstain} = 0.020$) | 3.11% | **100.00%** | 97.81% | **0 / 183** |
| **B — validation pick** ($\tau_\text{abstain} = 0.044$) | 32.85% | 96.02% | 66.85% | 11 |
| **B — same $\tau$, held out** | 32.85% | **87.69%** | 66.12% | 15 |

- `reports/g5_recall_floor.json`: **`gate_met: false`**. Missed at the point estimate *and* at the
  95% CI lower bound (recall CI $[75.77, 95.81]$).
- Validation-to-held-out recall gap: **8.33 pp**.
- A post-hoc sweep of the *test* split contains 5 thresholds that would clear 95%. Using one would
  forfeit the held-out guarantee, so it is recorded as a diagnostic and **not** used.
- **The abstention mechanism works. The ranker is not yet strong enough to exploit it**
  (PR-AUC 0.2016 after the 2026-09-15 validation-only re-tune — up from 0.1393 — against a 3.71%
  positive rate: about 5x no-skill, far from sufficient).

---

# 16. Statistical Hypothesis Testing
Source: `reports/statistical_significance.json`. Resampling unit is the **commit** (2,000
bootstraps) — not the row, so the intervals are not inflated by 86,469 correlated pairs.

- **Failure recall, 95% CI:** $[100.0\%, 100.0\%]$ over the $n = 135$ commits that had a failure.
- **Wall-clock time reduction, 95% CI:** mean $1.05\%$, $[0.25\%, 2.10\%]$ ($n = 183$).
  **Median $= -0.0\%$** — the typical commit saves nothing.
- **Cliff's $\delta$ on recall:** $0.5111$–$0.9852$ vs the subset baselines (large), $0.0$ vs Full
  Suite ($p = 1.0$, identical by construction).
- **Cliff's $\delta$ on time reduction:** $-0.906$ to $-0.998$ — **negative against every subset**
  **baseline.** Showing the recall effect sizes without this line would be cherry-picking.

---

# 17. Feature Ablation Study (LOGO)
Source: `reports/ablation_study.json`. Recall is at a 25% budget with abstention off, over the 135
held-out commits that had a failure. Full model: PR-AUC **0.1393**, recall **48.19%** *(pre-retrain
pipeline — retained as the honest record of the ablation measurement; the 2026-09-15 re-tune lifted
held-out PR-AUC to 0.2016 without changing the feature set, so the ablation's relative comparisons
belong to the old artifacts)*.

| Leave-one-group-out | PR-AUC | Recall @ 25% |
| :--- | :---: | :---: |
| Full model (32 features, 19 informative) | 0.1393 | 48.19% |
| **without dependency graph** | **0.0674** | **32.67%** |
| without AST complexity | 0.1515 | 50.87% |
| without history telemetry | 0.2214 | 52.34% |
| without diff churn | 0.1393 | 48.19% |

- **Static call-graph reachability is the only load-bearing family** — removing it costs 15.5 pp of
  recall. This contradicts the earlier claim that history carries most of the signal.
- **Removing history or AST *improves* the ranking.** Reported as measured.
- **13 of 32 features are constant in training**, including all 12 diff-churn columns — which is why
  removing that group reproduces the full model exactly and the group alone scores ROC-AUC 0.5000.
  That is almost certainly a **diff-mining defect**, not a finding about code churn.

---

# 18. Flakiness Stress Testing & Noise Robustness
Source: `reports/flakiness_robustness.json`. Noise is injected into **training labels only**;
evaluation uses clean held-out labels.

| Noise | Standard recall | Flakiness-weighted recall | Advantage |
| :---: | :---: | :---: | :---: |
| 0% | 48.19% | 48.19% | 0.0 |
| 5% | 52.77% | 53.71% | +0.94 pp |
| 10% | 54.40% | 53.65% | **−0.75 pp** |
| 20% | 54.86% | 54.05% | **−0.81 pp** |
| 30% | 55.58% | 56.27% | +0.69 pp |

- **The advantage is within noise: two of four points are negative.** No robustness claim is made.
- **Confound to state before reading the table:** the dataset is ~5% positive, so flipping labels
  mostly turns passes into failures and *raises* training prevalence (5.02% → 32.00% at 30% noise).
  Recall rising with the noise rate is likely reporting that prevalence change, not robustness.
- What *does* degrade cleanly is probability quality: standard Brier 0.0449 → 0.2089, while the
  weighted variant holds calibrated ECE ≤ 0.0289 by driving $T$ down to 0.1235.

---

# 19. Multi-Repository Cross-Project Generalization
Source: `reports/cross_repo_generalization.json`. LOPO across the five repositories actually
harvested: `pathspec`, `pyjwt`, `sqlparse`, `tabulate`, `validators`.

| Held-out repository | Recall @ 25% | ECE |
| :--- | :---: | :---: |
| validators | 88.42% | 0.0074 |
| pyjwt | 55.46% | 0.2899 |
| pathspec | 46.68% | 0.2957 |
| sqlparse | 37.28% | 0.1184 |
| tabulate | 30.00% | 0.0826 |
| **Macro mean** | **51.57%** | **0.1588** |

- Macro PR-AUC **0.1168**, macro ROC-AUC **0.7004**; on `tabulate`, ROC-AUC is **0.4747 — below
  chance**.
- **Zero-shot transfer does not work.** A new repository must be treated as out-of-distribution
  until it has its own execution history.
- Held-out ECE reaches 0.2957 against 0.0242 in-distribution: the *calibration* does not transfer
  either, so the abstention gate cannot be trusted on an unseen project.

---

# 20. Online Continuous Learning & Drift Adaptation
Source: `reports/continuous_learning.json`. Page-Hinkley over a mutant-ordered stream; shifts are
constructed by concatenating two repositories, so the crossing step is known by construction.

| PH threshold | Shifts detected | Median latency (mutants) | False alarms / 600 stationary |
| :---: | :---: | :---: | :---: |
| 0.5 | 4 / 5 | 4.5 | 11 |
| 1.0 | 3 / 5 | 18.0 | 6 |
| 2.0 | 2 / 5 | 33.5 | 4 |
| 5.0 | 2 / 5 | 44.5 | 0 |
| 10.0 | 0 / 5 | — | 0 |

- **No threshold is selected.** The grid is published so the false-alarm cost of a sensitive
  setting is visible next to the detection latency it buys. There is no free point.
- **The detector is one-sided by construction:** PH accumulates error above a running minimum, so a
  crossing into an *easier* project lowers error and is invisible. That is the 1/5 miss.
- **Adaptation did not help.** On the three repositories where the comparison is well-defined, the
  most-adapting configuration had *higher* mean error than the non-adapting one (Δ +0.076, +0.124,
  +0.178). Retraining on a small recent buffer is worse than not retraining. Stated, not buried.

---

# 21. Micro-Latency Profiling (<100ms SLA)
Source: `reports/latency_benchmark.json`; 150 iterations scoring a 50-test commit, warm-up excluded.

- **Array sanitisation:** $0.033\text{ms}$
- **5-Seed Ensemble Inference:** $\mathbf{3.246\text{ms}}$ (~97% of the cost)
- **Temperature Scaling:** $0.036\text{ms}$
- **Policy Decision:** $0.029\text{ms}$
- **Total scoring latency:** $\mathbf{3.345\text{ms}}$ mean, P99 $4.435\text{ms}$
  ($100\%$ under the $100\text{ms}$ budget).

**Two things this figure is not.** It is **scoring only** — git diff mining, AST parsing,
call-graph construction and history lookup all run *before* the timed region and dominate real
per-commit cost. And a cold process pays **$1321.8\text{ms}$** on its first batch, which is what a
CI runner that scores one commit and exits actually pays.

---

# 22. Enterprise Financial ROI Analysis
Source: `reports/economic_analysis.json`. **Every figure here is a projection from the measured
reduction rate onto an assumed team; no dollar amount below was observed.**

- **Assumed model:** 25 developers, 3 commits/dev/day, 250 days = 18,750 commits/yr, 45-min suite,
  \$0.016/runner-min, \$75/hr developer time, \$3,500/escaped bug, wait-cost factor $\beta = 0.30$.
- **Projected annual CI compute:** \$13,500 | **projected developer wait cost:** \$316,406
- **Measured test-execution reduction at the shipped operating point:** $\mathbf{0.0\%}$ wall-clock
- **Projected annual saving:** $\mathbf{\$0}$
- **Escaped-bug penalty avoided:** \$0 measured escapes on 183 commits — but
  `escape_cost_priced: false`, i.e. avoided-escape value is **not** credited to ConfTest here.
- **Break-even escaped bugs:** $0$. There is no saving to offset, so there is nothing to break even
  against. **The honest headline is \$0/year, not \$240,316/year.**

---

# 23. Research Prototype REST API (FastAPI)
- Modular routes with strict Pydantic V2 validation:
  - `POST /api/v1/select`: Core RTS selection.
  - `POST /api/v1/explain`: SHAP and rule explainability.
  - `GET /api/v1/calibration`: Report-backed calibration diagnostics.
  - `POST /api/v1/github/webhook`: Webhook ingestion.

---

# 24. GitHub Actions & PR Bot Integration
- Automated PR analysis bot with HMAC SHA-256 signature verification.
- Posts formatted markdown comment tables directly onto pull requests.
- Idempotent comment updating on subsequent commit pushes.

---

# 25. Streamlit Visual Analytics Dashboard
- 5 multi-page interactive modules:
  1. *Live PR Evaluation & Budget Sliders*
  2. *Confidence Calibration & Reliability Diagrams*
  3. *Uncertainty Drilldown & Disagreement Variance*
  4. *8-Strategy RTS Baseline Comparison*
  5. *SHAP Global Feature Importance Explorer*

---

# 26. Dockerized Prototype Stack
- Multi-stage `Dockerfile` with a minimal runtime image.
- `docker-compose.yml` orchestrates the API (8000), dashboard (8501), and shared SQLite volume.

---

# 27. Summary of Contributions
1. An **execution-labelled RTS dataset**: 561,711 (commit, test) pairs over 1,213 mutant-derived
   commits in 5 repositories, split chronologically on mutant boundaries.
2. A **selective-prediction policy for RTS** combining deep-ensemble epistemic uncertainty, post-hoc
   calibration and an OOD guard, which achieved **100.0% failure recall with 0 escaped commits** on
   183 held-out commits.
3. The finding that **temperature scaling beats isotonic regression here for the reason that matters
   to a gate**: isotonic wins on mean ECE and loses badly on worst-bin error.
4. A benchmark against 7 RTS baselines with commit-level bootstrap intervals and Cliff's $\delta$
   reported in **both** directions — recall and time.
5. **A negative result, stated as such:** at the shipped operating point the system saves 0.0% of
   wall-clock time, and the threshold that would save a third of executions misses our 95% recall
   floor by 7.3 pp on held-out data. The binding constraint is ranker accuracy, not the
   selective-prediction machinery — which is exactly what makes the machinery reusable.

---

# 28. Post-Measurement Model Re-Tune (2026-09-15)
Diagnosed from production: CI reports showed **5.2% top confidence** and abstention on every commit.

- **Diagnosis (two causes, both fixed):**
  - The shipped hyperparameters came from a six-candidate sweep whose members were nearly identical
    (all lr 0.03–0.05, no leaf regularization) → held-out PR-AUC stuck at 0.1393.
  - `tune_policy.py` stamped every tuned policy `untuned_constructor_defaults` — a provenance bug,
    so the shipped thresholds could not be traced to the sweep that measured them.
- **Fix, through the project's own validation-only pipeline** (no test-split peeking):
  - 10-candidate sweep → lr 0.02, `min_child_samples` 200 (val PR-AUC 0.2099 → 0.2496).
  - Ensemble → calibrator → policy regenerated in order; members bagged (subsample 0.8, freq 1).
- **Held-out results:** PR-AUC **0.1393 → 0.2016 (+45%)**, ROC-AUC **0.8743**, Brier 0.0344,
  native ECE 0.0098 (validation chose *uncalibrated* — the model is natively calibrated).
- **Policy:** grid extended below the old 0.10 floor; zero-escape point (0.010, 0.10) holds
  **100% recall, 0 escapes** on unseen test. The 95%-recall-floor frontier (8.1% reduction,
  2 escapes) is recorded in `reports/policy_tuning_report.json` for G5.
- **Live check on a code-heavy commit:** top confidence 5.2% → **6.9%**, epistemic σ 0.0290 →
  **0.0068** — under the abstention threshold; the gate now fires on confidence alone, honestly.
- The 13 constant diff features were **not** touched: they are truthful constants of the mutation
  protocol (1 line, 1 file, no commit message), per the anti-fabrication discipline.

---

# 29. Future Research Directions
- Neural Graph Attention Networks (GATs) for deep whole-program call-graph embeddings.
- Multi-language expansion (Java, Go, TypeScript).
- Multi-Armed Bandit dynamic test budget scheduling in edge CI nodes.

---

# 30. Publications & Deliverables
- **IEEE/ACM 8-Page Conference Paper:** `paper/main.tex` & `references.bib`
- **Official KTU B.Tech Major Project Report:** `ktu_report/`
- **Interactive Google Colab Demonstration:** `notebooks/conftest_colab_demo.ipynb`
- **Automated validation:** run `python -m pytest tests/ -q` for the current suite; the count is not frozen in the presentation.

---

# 31. Thank You!
### Questions & Viva Defense Discussion

**Bipin B** | KTU B.Tech Computer Science & Engineering  
Repository: `https://github.com/bbipin/conftest`
