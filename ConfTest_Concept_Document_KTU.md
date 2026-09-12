# ConfTest: Confidence-Calibrated Regression Test Selection with Selective Prediction for CI/CD Optimization

**APJ Abdul Kalam Technological University (KTU) | Final-Year B.Tech Major Project Specification**  
*Department of Computer Science and Engineering*

> **Status note (added after implementation).** This is the original project *specification*, written before any measurement. The reduction figures below are the targets it set, not results. Measured on 183 held-out commits, the shipped operating point reached **100.0% failure recall with zero escaped commits** but only **3.11% test-execution reduction (0.0% wall-clock)**, because it abstains on 97.81% of commits; the 32.85%-reduction operating point falls to 87.69% recall. See [README -- Measured Results](README.md#-measured-results-and-what-they-do-not-show).

---

## 1. Executive Summary & Problem Disproof

### Decision on Baseline Concept
> **REJECT THIS PROJECT AS CURRENTLY DEFINED:** *“AI-Based Repository-Aware Regression Testing and Code Change Risk Detection”*

### Justification
1. **Prior Work Saturated:** Meta deployed *Predictive Test Selection* (Machalica et al., ICSE-SEIP 2019) using Gradient Boosted Decision Trees on commit diffs and test history across millions of builds. Google uses static dependency graphs and change-impact analysis (Memon et al., 2017). Deterministic tools like *Ekstazi* (ASE 2015) and *pytest-ranking* (FSE 2025) have already established static change-aware testing.
2. **The Open Research Problem:** Standard ML test selectors suffer from **uncontrolled false negatives** because their probability estimates are miscalibrated on out-of-distribution (OOD) code changes (e.g. major refactoring, third-party library upgrades).
3. **The ConfTest Contribution:** Reframes regression testing around **Risk-Calibrated Selective Prediction (Abstention)**:
   - Evaluates test failure probability using fast LightGBM models.
   - Measures prediction uncertainty via Venn-Abers calibration and temperature scaling.
   - If uncertainty on a pull request exceeds the safety threshold $\tau$, ConfTest **ABSTAINS** and triggers a safe fallback (100% full test suite).
   - **Target at proposal time:** 40–50% test time reduction while bounding missed regressions.
     *Measured outcome:* the bound held (zero escapes on held-out data) and the reduction target was
     not met (3.11% of executions, 0.0% wall-clock, at 97.81% abstention).

---

## 2. Literature Map (Summary of Key Venues)

| System / Paper | Venue | Core Technique | Key Result | ConfTest Differentiator |
|---|---|---|---|---|
| **Predictive Test Selection** (Meta) | ICSE-SEIP 2019 | GBDT on diff churn & test history | 50% test reduction, >95% recall | Added Venn-Abers calibration & automated abstention fallback |
| **pytest-ranking** | FSE 2025 | Change distance + failure history | 20–50% faster fault revelation | Binary selection with statistical safety bounds (not just ordering) |
| **Names Are All You Need** | ISSTA 2026 | Static AST symbol dependency | Safe selection, <100ms analysis | Overcomes static over-selection on core utility modifications |
| **Conformal Defect Prediction** | EMSE 2024 | Inductive Conformal Prediction | Guaranteed $1-\alpha$ error bound | Applied to test execution optimization rather than commit warnings |
| **Ekstazi** | ASE / TSE 2015-18 | Checksum bytecode dependency | 32% time reduction in Java | Sub-selects tests when high-level utility interfaces are touched |
| **Can LLMs Replace RTS?** | MSR / arXiv 2024 | Prompting Llama-3 & GPT-4o | High recall, but 10x slower & $$$ | Proves pure LLM inference is too slow (15s/PR) for real-time CI |

---

## 3. Four-Member Work Breakdown Structure (WBS)

```
+----------------------------------------------------------------------------------------------------+
|                                    4-MEMBER TASK DIVISION MATRIX                                   |
+--------------------+---------------------------------------------------+---------------------------+
| TEAM MEMBER        | PRIMARY MODULE & TECHNICAL OWNERSHIP              | PAPER & VIVA SECTIONS     |
+--------------------+---------------------------------------------------+---------------------------+
| **MEMBER 1**       | **Data Engineering, CI/CD Engine & Storage**      | Architecture, CI Pipeline,|
|                    | - GitHub App / Action Webhook Integration         | Data Engineering, Schema, |
|                    | - Git log & CI test log (JUnit XML) ingestion     | Deployment & Benchmarks   |
|                    | - PostgreSQL/SQLite schema & ORM repository layer |                           |
|                    | - Temporal data splitting & flakiness filtering   |                           |
+--------------------+---------------------------------------------------+---------------------------+
| **MEMBER 2**       | **AST Parsing, Dependency & Feature Engineering** | Feature Taxonomy, Code    |
|                    | - Tree-sitter AST syntax parser (Java/Python)     | Representation, AST Graph,|
|                    | - Static call-graph & import dependency analysis  | Extraction Latency Study  |
|                    | - Code churn, cyclomatic complexity delta engine  |                           |
|                    | - [Optional] CodeBERT GPU embedding extractor     |                           |
+--------------------+---------------------------------------------------+---------------------------+
| **MEMBER 3**       | **AI/ML Models, Calibration & Selective Engine**  | ML Formulation, Post-Hoc  |
|                    | - LightGBM / XGBoost failure scoring models       | Calibration, Venn-Abers,  |
|                    | - Venn-Abers & Temperature Calibration algorithms | Uncertainty Theory & RQs  |
|                    | - Selective Prediction & Abstention Decision Rule |                           |
|                    | - Ensemble epistemic uncertainty quantification   |                           |
+--------------------+---------------------------------------------------+---------------------------+
| **MEMBER 4**       | **Dashboard, Explainability, Stats & Experiments**| Statistical Validation,   |
|                    | - Web Analytics Dashboard (FastAPI + React)       | SHAP XAI Evaluation,      |
|                    | - PR Comment Bot with TreeSHAP visual breakdown   | Ablation Studies, ROI &   |
|                    | - Wilcoxon / Cliff's Delta statistical test suite | Threats to Validity       |
|                    | - Automated experiment harness (Defects4J eval)   |                           |
+--------------------+---------------------------------------------------+---------------------------+
```

---

## 4. Hardware, GPU & Zero-Cost Budget Evaluation

### Team Laptop: ASUS ROG Strix G16
- **Processor:** Intel Core i7-13650HX (14 cores, 20 threads) $\implies$ Parallel AST parsing and LightGBM training.
- **GPU:** NVIDIA RTX 4050 6GB VRAM $\implies$ Optional CodeBERT semantic embedding extraction.
- **RAM:** 16 GB DDR5 $\implies$ In-memory graph processing and tabular datasets.
- **Storage:** 1 TB NVMe SSD $\implies$ Defects4J repository storage and log databases.
- **Budget:** **₹0.00** (Zero Cloud / Zero API cost tier).

---

## 5. Quick Viva Voce Defense

- **What problem does ConfTest solve?**  
  *Large CI test suites take 30–60 minutes per pull request. ConfTest aims to cut test execution time
  while preventing missed bugs through uncertainty-calibrated abstention. In our evaluation the safety
  half worked (100% failure recall, zero escapes on 183 held-out commits) and the saving half did not
  (3.11% of executions, 0.0% wall-clock, because the policy abstained on 97.81% of commits).*
- **Why not run all tests?**  
  *Running all tests costs thousands of dollars in CI runner fees and creates severe feedback delays for developers.*
- **What is Abstention?**  
  *When prediction uncertainty exceeds a safety budget $\tau$ (e.g. on out-of-distribution refactorings), ConfTest abstains from pruning and safely executes the full test suite.*
- **How was data leakage avoided?**  
  *By enforcing strict chronological temporal splitting (70% train / 15% calibration / 15% test) instead of random $k$-fold cross validation.*
