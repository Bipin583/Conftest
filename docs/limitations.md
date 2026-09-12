# Limitations and Project Status

ConfTest is a research-grade Python regression-test-selection prototype. Its outputs are empirical recommendations from specific artifacts and data—not a formal proof that omitted tests pass.

## Observed safety is not guaranteed safety

The shipped operating point observed 100% failure recall and zero escaped commits on 183 held-out commits (86,469 commit-test rows). It also abstained on 97.81% of commits, reduced executed test count by only 3.11%, and measured 0.0% wall-clock reduction. This is an observed outcome in `reports/baseline_comparison.csv` and `reports/economic_analysis.json`, not a universal guarantee.

A higher-reduction point achieved 32.85% count reduction but only 87.69% held-out recall and 15 escaped commits. It failed the project's 95% recall gate. Deployment must therefore measure its own distribution and retain full-suite fallback/audits.

## Artifact provenance and trainer change

The committed evaluation reports predate the row-bagging fix. Their ensemble metadata lacks `subsample_freq`; LightGBM therefore trained every member on all rows despite `subsample=0.8`. Current code enables and records row bagging, but published numbers must not be attributed to the corrected trainer until all dependent artifacts and reports are regenerated.

A model, calibrator, and policy form one operating unit. Replacing one invalidates calibration and threshold evidence for the others.

## Feature and dataset constraints

Thirteen of the 32 feature columns are constant in the committed training harvest, including all 12 diff/churn columns. Those columns cannot contribute learned variation in those artifacts. This indicates a data-generation limitation and weakens claims about churn-aware prediction.

Historical features depend on prior executions. Cold-start extraction mostly zero-fills history (`hist_avg_duration` uses `0.05` seconds); it does not assign a separately learned conservative uncertainty to every novel test. Static feature extraction is Python-specific and can miss dynamic imports, generated code, `getattr`, monkey patching, runtime plugin registration, mocks, native extensions, and environment-dependent behavior.

Mutation-derived labels and harvested real bugs answer different validity questions. Mutants are controlled and scalable but may not represent natural faults; external bugs increase realism but introduce repository/tooling and survivorship constraints. Temporal splits reduce future leakage within the committed corpus but do not establish transfer to a new repository or future development regime.

## Weak cross-repository transfer

The zero-shot cross-repository evaluation is weak. Structural feature names do not make distributions invariant across projects. Repositories differ in test conventions, architecture, failure prevalence, dependency topology, and history availability. A new repository should default to full-suite behavior until local evidence supports subset selection.

## Uncertainty limits

Ensemble disagreement is a proxy for epistemic uncertainty. Members share the same model family, feature representation, and mostly the same data; they can agree and still be wrong. The pre-row-bagging artifacts further narrow the intended diversity. OOD file/churn limits are hand-designed/tuned triggers, not comprehensive distribution-shift detection.

## Calibration limits

Calibration is conditional on the score and data distributions used to fit it. ECE is bin-dependent, positive outcomes are sparse, and point improvements can be within resampling uncertainty. Calibration cannot restore predictive information absent from features, and a calibrated score is not the probability that the selected subset is safe.

## Flakiness study confounding

The flakiness stress evaluation changes labels and therefore failure prevalence as noise increases. Metric movement cannot be attributed solely to “robustness to flakiness”; prevalence and task difficulty move too. Synthetic flips also do not reproduce all mechanisms of real flaky tests. Treat the report as a stress test, not an estimate of production flaky-test performance.

## Performance and economics

The latency benchmark measures model scoring/decision work on prepared features. It excludes repository checkout, Git diff mining, pytest collection, AST parsing, dependency-graph construction, database/history queries, model cold start, API/network overhead, and test execution. It cannot support a sub-100 ms end-to-end SLA claim.

Economic outputs are scenario projections based on configurable runner cost, developer wait assumptions, and escape penalties. They are not realized savings. At the shipped operating point, measured wall-clock reduction is zero. Count reduction must not be relabeled as duration or dollar saving.

## Continuous learning maturity

Page-Hinkley detection and replay retraining are implemented and unit-tested, and an evaluation stage exists. They are not wired into the default serving/selection path as an autonomous production loop. Drift thresholds, labels, retraining approvals, rollback, and post-adaptation calibration/policy tuning remain operational responsibilities. An adaptation report does not justify silent artifact replacement.

## Persistence and deployment

SQLite is the configured and exercised database. SQLAlchemy accepting another URL does not establish tested PostgreSQL support. FastAPI and Streamlit are prototype interfaces; production use requires secret management, explicit CORS origins, authentication/authorization at the deployment boundary, backups, monitoring, resource limits, and concurrency testing.

A selective run cannot observe failures in tests it did not execute. Database outcome fields requiring a complete comparison remain null unless the full suite also ran. Any aggregate claiming zero misses must restrict itself to verified outcomes.

## Appropriate use

ConfTest is appropriate for research, offline evaluation, shadow-mode ranking, and guarded CI experiments with complete-suite audits. It should not be the sole control for safety-critical release decisions or be presented as a certified, enterprise-ready, or formally verified system.

For evidence and reproduction paths, see [Reproducibility](reproducibility.md), [Selective prediction](selective_prediction.md), [Statistical methodology](statistical_methodology.md), and `python scripts/evaluate.py`.
