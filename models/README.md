# ConfTest Models Directory

This directory stores serialized models and calibration artifacts:
- `baselines/`: Serialized models for the 8 budget-matched baselines.
- `ensembles/`: 5-seed diverse LightGBM/XGBoost ensemble model checkpoints.
- `calibrated/`: Post-hoc isotonic regression and temperature scaling mapping artifacts.
- `calibrator.joblib`: The fitted calibrator every consumer loads by default, written by
  `scripts/calibrate_model.py`. Not tracked -- `reports/calibration_report.json` records
  which method won, on which split, and by how much.
- `policy_config.json`: The shipped abstention thresholds, written by
  `scripts/tune_policy.py`. Tracked, because it is a decision rather than a build product,
  and `reports/policy_tuning_report.json` carries the sweep it was chosen from.
