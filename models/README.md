# ConfTest Models Directory

This directory stores serialized models and calibration artifacts:
- `baselines/`: Serialized models for the 8 budget-matched baselines.
- `ensembles/`: 5-seed LightGBM (`LGBMClassifier`) ensemble member checkpoints.
  `ensemble_metadata.json` records `n_estimators: 150`, which is the configured
  ceiling, not what was kept: early stopping on the validation split fires within
  the first handful of rounds, and the shipped boosters hold **7, 5, 5, 5 and 6
  trees** respectively (verify with `booster_.num_trees()`). Read "150 trees" as
  a budget the fit never spent. It is the most likely reason the ranker is weak
  (held-out PR-AUC 0.1393 against a 3.71% positive rate) and it is the first thing
  to change before concluding that these features cannot rank tests.
- The shipped `ensembles/5_seed_lgbm/ensemble_metadata.json` also predates two
  fixes: it stores absolute member paths (the loader ignores them and resolves
  beside the metadata) and it has no `subsample_freq` key, so its members were
  trained without row bagging. `load_ensemble` warns about that on every load.
  Every published epistemic-uncertainty number therefore comes from members that
  differ only by seed-dependent tie-breaking, which understates the spread.
- `calibrated/`: Post-hoc calibrator mappings. Temperature scaling is the shipped method;
  isotonic regression was fitted and rejected (lower mean ECE, worst-bin error nearly doubled).
- `calibrator.joblib`: The fitted calibrator every consumer loads by default, written by
  `scripts/calibrate_model.py`. Tracked since the reusable CI workflow needs it on a
  fresh checkout -- without it the engine runs uncalibrated and abstains on every
  commit, silently degrading to a full-suite run. `reports/calibration_report.json`
  still records which method won, on which split, and by how much.
- `ensembles/5_seed_lgbm/`: Tracked for the same reason as the calibrator: CI
  (`.github/workflows/conftest-rts-reusable.yml`) checks out this repo to run test
  selection against other repositories, and a checkout without the ensemble falls
  back to the heuristic, which abstains on every commit. Other ensembles stay
  ignored as local build products.
- `policy_config.json`: The shipped abstention thresholds, written by
  `scripts/tune_policy.py`. Tracked, because it is a decision rather than a build product,
  and `reports/policy_tuning_report.json` carries the sweep it was chosen from.
