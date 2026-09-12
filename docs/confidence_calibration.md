# Confidence Calibration

Calibration maps a model score to an empirically interpretable failure probability. It does not improve ranking automatically and does not make an uncertain model safe by itself.

## Methods implemented

ConfTest evaluates:

- **Temperature scaling**: fits one positive temperature to transform model log-odds while preserving score order.
- **Isotonic regression**: fits a non-decreasing piecewise mapping with greater flexibility and greater overfitting risk.
- **Uncalibrated**: a valid selection outcome when candidate improvements are not supported.

The maintained calibration code is under `src/conftest/models/calibration.py`; `scripts/calibrate_model.py` fits candidates and writes `models/calibrator.joblib` and `reports/calibration_report.json`.

## Split discipline

The base model is trained on the training split. Candidate calibrators are fitted and selected on validation data. The chosen mapping is then measured on the held-out test split. Report consumers must distinguish:

- `selection_split`: where the method was chosen;
- `metrics_split`: where its reported performance was measured; and
- `resampling_unit`: the unit used for confidence intervals.

The API exposes these labels because a validation metric used for method selection is not interchangeable with a held-out metric.

## Metrics

For binary outcomes `y_i` and probabilities `p_i`:

- **Brier score** is the mean squared probability error.
- **ECE** is a bin-weighted mean absolute gap between observed frequency and mean confidence.
- **MCE** is the largest observed bin gap.
- **Reliability bins** expose count, confidence, and observed frequency so aggregate scores can be inspected.

ECE and MCE depend on binning and sample composition. A lower point estimate can arise from sampling variation. The committed report therefore records paired differences and bootstrap intervals where commit labels permit commit-level resampling.

## Committed result

The committed run selected temperature scaling. On validation, ECE changed from `0.0390` to `0.0242`; on the held-out test split the report records `0.0415` to `0.0161`. The validation pair explains selection, while the held-out pair describes evaluation. They must not be combined into one before/after claim.

Point estimates alone do not establish a statistically reliable improvement: paired intervals in `reports/calibration_report.json` are part of the result and should be quoted alongside claims of superiority. The API deliberately serves no hard-coded defaults and returns 503 when the report is absent.

## Reproduce or inspect

```bash
python scripts/calibrate_model.py
python scripts/evaluate.py --run calibration
```

`python scripts/evaluate.py` lists the expected producer and artifacts without rerunning the stage. Any new ensemble requires a newly fitted calibrator and a retuned policy; a calibrator is not portable across arbitrary score distributions.

## Limits

Calibration quality is conditional on the validation/test distributions. Sparse positive outcomes, temporal drift, repository transfer, and flakiness can invalidate probability semantics. Temperature scaling changes score sharpness but cannot add missing predictive signal; isotonic calibration can overfit small or unrepresentative validation sets. See [Dataset](dataset.md), [Statistical methodology](statistical_methodology.md), and [Limitations](limitations.md).
