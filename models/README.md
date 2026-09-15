# ConfTest Models Directory

This directory stores serialized models and calibration artifacts:
- `baselines/`: Serialized models for the 8 budget-matched baselines.
- `ensembles/`: 5-seed LightGBM (`LGBMClassifier`) ensemble member checkpoints.
  The 2026-09-15 re-tune (validation-only, via `scripts/tune_model.py` with
  `reports/tuning_candidates_20260915.json`) selected lr=0.02 with
  `min_child_samples=200` over a 10-candidate sweep; members early-stop at
  **18, 21, 14, 17 and 15 trees**. Early stopping this early is a real signal
  ceiling for this feature set, not a misconfiguration: the low-LR +
  leaf-regularized candidates improved held-out PR-AUC substantially, and the
  remaining gap is the features, not the booster capacity.
- Held-out test metrics for the shipped ensemble (`reports/ensemble_training_report.json`):
  **PR-AUC 0.2016** (was 0.1393 pre-re-tune, +45%), ROC-AUC 0.8743,
  Brier 0.0344, mean epistemic std 0.0021.
- `calibrated/`: Post-hoc calibrator mappings. On the retrained ensemble,
  validation selected **uncalibrated** (native ECE 0.0098) over isotonic and
  temperature scaling; an explicit identity artifact is shipped for
  reproducibility. `reports/calibration_report.json` records the comparison.
- `calibrator.joblib`: The fitted calibrator every consumer loads by default, written by
  `scripts/calibrate_model.py`. Tracked since the reusable CI workflow needs it on a
  fresh checkout -- without it the engine runs uncalibrated and abstains on every
  commit, silently degrading to a full-suite run.
- `ensembles/5_seed_lgbm/`: Tracked for the same reason as the calibrator: CI
  (`.github/workflows/conftest-rts-reusable.yml`) checks out this repo to run test
  selection against other repositories, and a checkout without the ensemble falls
  back to the heuristic, which abstains on every commit. Other ensembles stay
  ignored as local build products.
- `policy_config.json`: The shipped abstention thresholds, written by
  `scripts/tune_policy.py` and stamped with the `source` path of the sweep that
  chose them (zero-escape objective: `tau_abstain=0.010`, `tau_conf=0.10`,
  100% failure recall, 0 escaped commits on the unseen test split).
  `reports/policy_tuning_report.json` carries the full sweep, including the
  recall-floor frontier (8.1% reduction at 98.4% recall, 2 escaped commits)
  that gate G5 is stated against.
