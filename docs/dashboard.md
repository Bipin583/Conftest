# Dashboard

The Streamlit dashboard is a five-page interface over the ConfTest selector and committed evaluation artifacts. It is an analysis UI, not an independent source of measurements: loaders in [`dashboard/utils.py`](../dashboard/utils.py) either read an artifact or stop the page with the exact producer command. They do not substitute placeholder metrics.

## Start the dashboard

From the repository root, after installing the project and dependencies:

```bash
python -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`. The equivalent POSIX convenience target is `make run-dashboard`; direct `python -m` commands are preferred on Windows.

The Docker stack publishes the same port:

```bash
# Requires production secret setup; see deployment_docker.md
docker compose up -d --build
```

The dashboard reads repository-local artifacts and the SQLite database directly. It does not currently call the FastAPI service for its page data.

## Pages

| Page | Implementation | Data and behavior |
|---|---|---|
| Home | `dashboard/app.py` | Headline and calibration cards from measured report artifacts; baseline frontier when available. |
| Live PR Evaluation | `dashboard/pages/1_🚀_Live_PR_Evaluation.py` | Interactive sample-suite selection using `ConfTestEngine`; reports `FAST_SELECTED` or `SAFE_FULL_SUITE`. This is a local evaluation form, not a live GitHub event stream. |
| Confidence Calibration | `dashboard/pages/2_📉_Confidence_Calibration.py` | Reliability bins and calibration metrics from `reports/calibration_report.json`. |
| Uncertainty Drilldown | `dashboard/pages/3_🔮_Uncertainty_Drilldown.py` | Risk-coverage analysis plus ensemble and policy metadata. |
| Baseline Comparison | `dashboard/pages/4_📊_Baseline_Comparison.py` | Strategy comparisons from `reports/baseline_comparison.csv`. |
| SHAP Explainability | `dashboard/pages/5_🔍_SHAP_Explainability.py` | Global and per-test attributions from `reports/explanations.json`. |

## Artifact contract

| Artifact | Used for | Producer |
|---|---|---|
| `reports/baseline_comparison.csv` | Headline/frontier and baseline page | `python scripts/train_baseline.py` |
| `reports/calibration_report.json` | Calibration cards and reliability page | `python scripts/calibrate_model.py` |
| `reports/uncertainty_analysis.json` | Uncertainty page | `python scripts/uncertainty_eval.py` |
| `reports/explanations.json` | SHAP page | `python scripts/generate_explanations.py` |
| `models/ensembles/5_seed_lgbm/ensemble_metadata.json` | Ensemble member metadata | `python scripts/train_ensemble.py` |
| `models/policy_config.json` | Active selective-policy thresholds | `python scripts/tune_policy.py` |

Run `python scripts/evaluate.py` to inspect the broader report-to-producer mapping. Running `python scripts/evaluate.py --all` launches expensive evaluations and is not required merely to start the UI.

Missing evidence is expected to be visible. A page displays the absent path and producer and then stops instead of printing an unverified zero, cached example, or invented fallback.

## Interpreting displayed results

- Reduction values distinguish test-count reduction from measured execution-time reduction.
- `SAFE_FULL_SUITE` is an abstention decision, not a selector failure.
- Dashboard values describe the split and artifact named by their provenance; they are observations, not formal guarantees for future repositories.
- The canonical feature list is maintained in [Feature schema](feature_schema.md), not duplicated by the dashboard.
- Known measurement and deployment constraints are catalogued in [Limitations](limitations.md).

## Database behavior

Dashboard utilities open the same SQLAlchemy session used by the application. With the default host setup this is `data/conftest.db`; in Compose it is `/app/data/conftest.db` through `CONFTEST_DATABASE_URL`. Selection decisions appear only when the invoking path persists them. A dry selection without persistence does not become telemetry automatically.

SQLite is configured for this project-scale workload. Multiple writers can still encounter lock contention; close duplicate development processes and inspect API/dashboard logs before deleting any database file.

## Troubleshooting

**The page says an artifact has not been produced.** Run the command shown by the page, or verify that the expected `reports/` and `models/` mounts are present in Docker.

**The Live PR page cannot construct the engine.** Verify these artifacts exist:

```text
models/ensembles/5_seed_lgbm/
models/calibrator.joblib
models/policy_config.json
```

**Displayed results seem stale.** Streamlit may cache resources. Restart the dashboard after replacing model or report artifacts, and confirm the artifact timestamps and producer metadata.

**The browser cannot open the page.** Confirm the process is listening on port 8501. For remote hosts, bind Streamlit to an appropriate interface and place it behind authenticated TLS rather than exposing the development server directly.

**A chart is empty.** Check the underlying artifact for empty bins or missing rows. In particular, empirical reliability diagrams intentionally omit bins with no samples rather than plotting them as zero-valued measurements.

For API operation, use [API reference](api.md). For container startup and persistence, use [Docker deployment](deployment_docker.md).
