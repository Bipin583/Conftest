# Statistical Methodology

ConfTest treats the change observation—not an individual test row—as the dependence unit. Tests evaluated for the same mutant share the same code change and budget, so row-level resampling would overstate the effective sample size.

## Point estimates and denominators

The held-out split has 183 mutant observations and 86,469 test rows. Metrics are aggregated according to their producer:

- TRR and ETR can be computed for all 183 observations when durations are present.
- Per-observation failure recall is defined only for observations with at least one failing test; `reports/statistical_significance.json` uses 135 such paired observations and reports 48 dropped undefined values.
- `reports/baseline_comparison_intervals.json` also reports pooled failure recall over all failing rows while resampling all 183 observations as intact clusters.

These summaries answer different questions. A mean of per-observation recalls must not be substituted for pooled failing-row recall.

## Commit-unit bootstrap

`scripts/train_baseline.py --bootstraps 2000` produces 95% percentile bootstrap intervals by resampling whole observations with replacement. Test rows within a sampled observation move together. The committed artifact records:

```text
resampling_unit = commit
num_bootstraps = 2000
confidence_level = 0.95
```

It reports intervals both for strategy metrics and paired differences from the selective ConfTest reference. “Excludes zero” describes the committed bootstrap interval; it is not a universal performance claim.

For the shipped point, the aggregate report gives:

- TRR 3.1086%, 95% interval 0.7577% to 6.1630%;
- ETR 0.0292%, 95% interval 0.0067% to 0.0606%;
- pooled failure recall 100%, 95% interval 100% to 100% on this sample.

The CSV rounds ETR to `0.0%`. A degenerate 100% interval records no observed misses in these resamples; it does not bound an unseen population escape rate.

## Pairwise non-parametric analysis

`reports/statistical_significance.json` compares the reference with each baseline using paired Wilcoxon signed-rank results and Cliff's delta labels. It records the exact pair counts for each metric. At the full-suite comparison, failure recall is identical and the report records `p=1.0` and negligible effect. Against more aggressive selectors, higher recall is paired with materially lower reduction, so significance must not be summarized as unqualified superiority.

The report serializes some very small p-values as `0.0`; this means below its output precision, not mathematical probability zero. The committed artifact does not establish a multiple-comparison correction, so individual `p < 0.05` flags should be read as pairwise evidence rather than family-wise confirmation.

## Calibration uncertainty

Calibration selection occurs on validation data; the selected method is then measured on test data. Paired differences and bootstrap intervals in `reports/calibration_report.json` are the evidence for improvement. Validation and test estimates cannot be pooled into one before/after result. ECE and MCE remain bin- and sample-dependent. See [Confidence calibration](confidence_calibration.md).

## Interpretation rules

- Report observation counts and metric denominators with estimates.
- Keep validation selection separate from held-out evaluation.
- Distinguish count reduction from duration reduction.
- Treat missing or undefined metrics as null, not zero.
- Do not infer production guarantees from zero observed held-out escapes.
- Do not claim independent natural commits: the committed observations are mutation records ordered by a surrogate timeline.

Reproduce the baseline statistics with:

```bash
python scripts/evaluate.py --run baselines -- --bootstraps 2000
```

See [Experiments](experiments.md), [Dataset](dataset.md), and [Limitations](limitations.md).
