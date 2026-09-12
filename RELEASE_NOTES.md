# ConfTest v1.0.0 Research Prototype

**Release date:** April 2026
**Author:** Bipin B (KTU B.Tech CSE Final Year Major Project)
**Status:** Research prototype; not production-certified

---

## ⚠️ Read This Before Quoting a Number

Earlier revisions of this release's documentation reported **68.6% CI-time reduction at 100% recall**.
No artifact in this repository supports that figure, and it has been removed rather than re-derived.
The measured frontier has two points and neither is that one:

| Operating point | Test-execution reduction | Wall-clock | Failure recall | Abstention | Escaped commits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A -- shipped** | 3.11% | 0.0% | **100.00%** | 97.81% | **0 / 183** |
| **B -- higher reduction** | 32.85% | — | **87.69%** | 66.12% | 15 / 183 |

Point **B** fails this project's 95% recall gate (`reports/g5_recall_floor.json`, `gate_met: false`),
with an 8.33 pp recall gap between the validation and held-out splits. **The abstention mechanism is
sound; the ranking model is not yet strong enough to buy meaningful savings with it.** Projected
annual saving at the shipped operating point is **$0**.

Three further caveats carry to every number in this release:

1. **All committed artifacts predate the row-bagging fix.** `LGBMClassifier` ignores `subsample`
   unless `subsample_freq > 0`, which was unset, so the "5-seed ensemble with stochastic row bagging"
   trained without row bagging. The code is fixed; **nothing under `reports/` or `models/` has been
   retrained**. Epistemic $\sigma$ is the quantity most likely to move, and $\sigma$ is what the
   abstention gate keys off -- so the operating points above may shift on a retrain.
2. **13 of 32 features are constant in training**, including all 12 diff-churn columns. Effectively 19
   features carry signal.
3. **Zero-shot cross-repository transfer does not work**: macro recall 51.57% across five held-out
   repositories, below chance on one, held-out ECE up to 0.2957.

Full detail: [README -- Measured Results](README.md#-measured-results-and-what-they-do-not-show).

---

## 🌟 Highlights
- **32-Feature Mining Pipeline:** Full extraction across AST syntactic complexity, NetworkX static call-graphs, code churn metrics, and SQLite failure telemetry.
- **5-Seed Deep Ensemble Epistemic Uncertainty:** Measures ensemble disagreement variance $\sigma(c, t)$ to detect out-of-distribution refactorings.
- **Post-Hoc Temperature Scaling Calibration:** Reduces validation ECE from **$0.0390$ to $0.0242$** at
  $T^* = 0.7941$ (paired improvement $0.0148$, 95% CI $[0.0030, 0.0393]$). Isotonic regression was
  evaluated and **rejected**: lower mean ECE, but worst-bin error rose from $0.1235$ to $0.2098$.
- **Selective Prediction Policy:** On 183 held-out commits, **100.0% failure recall with 0 escaped
  commits** at a **3.11% test-execution reduction** and **0.0% wall-clock reduction**, abstaining on
  **97.81%** of commits.
- **Scoring Latency:** **3.345 ms** mean to score a 50-test commit (P99 $4.435$ ms), 100% under the
  $<100\text{ms}$ budget. Excludes feature mining; cold start is $1.32$ s.
- **Production Stack:** FastAPI Backend + GitHub PR Comment Bot + Streamlit Visual Analytics Dashboard + Multi-stage Docker Compose Stack.
- **Academic Package:** Camera-ready 8-page IEEE/ACM research paper LaTeX package + Complete KTU Project Report + 30-slide Reveal.js viva presentation deck.
- **Automated validation:** run `python -m pytest tests/ -q` for the current suite; counts are intentionally not frozen in release prose.
