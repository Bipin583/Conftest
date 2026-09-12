# Dataset and Data Provenance

ConfTest's committed training corpus is mutation-derived. Each row represents one `(mutant, test)` pair whose label came from an executed pytest run. It is not a collection of ordinary production commits, and the synthetic ordering field must not be presented as wall-clock history.

## Subject repositories

`data/processed/real_features_manifest.json` records five Python repositories:

| Repository | Usable mutants | Tests | Rows | Positive rows |
|---|---:|---:|---:|---:|
| `pathspec` | 246 | 191 | 46,986 | 3,738 |
| `pyjwt` | 250 | 365 | 91,250 | 4,751 |
| `sqlparse` | 236 | 507 | 119,652 | 9,628 |
| `tabulate` | 232 | 349 | 80,968 | 7,651 |
| `validators` | 249 | 895 | 222,855 | 1,676 |

The combined dataset has 1,213 mutant observations, 561,711 commit-test rows, and 27,444 positive rows. Repository revisions and import-provenance checks are recorded in the manifest. These figures describe the committed artifacts only.

## Label construction

`scripts/build_real_dataset.py` accepts only harvested mutation records with trustworthy outcomes. For a mutant `c` and candidate test `t`:

```text
y(c, t) = 1  when the executed test killed the mutant
y(c, t) = 0  otherwise
```

Mutants classified as suite-breaking, timed out, or contaminated are excluded according to the producer's recorded rules rather than assigned inferred labels. The full harvest retains their statuses for audit.

A mutant that no test kills contributes all-zero rows. The manifest therefore publishes two prevalence definitions:

- `failure_rate_all_mutants = 4.8858%` over all committed rows;
- `failure_rate_detected_mutants = 6.8174%` after restricting the denominator to mutants killed by at least one test.

The project's G1 gate uses the second definition because an undetected mutant has no positive selection target. Its rows remain in the dataset with `mutant_detected=0`; they are not silently deleted.

## Causal history and ordering

Mutation records have no real commit timestamp. `commit_timestamp` is a monotonic surrogate beginning at 2000-01-01 with one second per mutant, and `mutant_index` exposes the actual ordering. Historical features for mutant `i` are computed before recording outcomes from `i`, so they use only mutants `0..i-1` within the harvest sequence.

The committed split is chronological with respect to that surrogate order:

| Split | Mutants | Rows | Positives | Positive rate |
|---|---:|---:|---:|---:|
| Train | 849 | 391,825 | 19,658 | 5.017% |
| Validation | 181 | 83,417 | 4,578 | 5.488% |
| Test | 183 | 86,469 | 3,208 | 3.710% |

This reduces future-to-past leakage within the generated sequence. It does not establish natural chronological evolution across repositories.

## Inert columns in the committed corpus

Thirteen canonical features are constant: all 12 diff/churn fields and `hist_flaky_score`. A mutation changes one source line but carries neither a real commit diff nor a commit message, while the baseline test universe admits only tests stable during screening. The fields remain in the canonical schema for serving compatibility, but they cannot contribute variation to models trained on this artifact. See [Feature schema](feature_schema.md) and [Feature ablation](feature_ablation.md).

## External real-bug status

The checked-in BugsInPy harvest summary records one attempted `tqdm` case ending in `env_error`, with zero labelled records and `labels_measured: false`. It is an attempted external-validation artifact, not evidence of real-bug performance. No headline metric should attribute BugsInPy validation to the committed results.

## Reproduction

```bash
python scripts/build_real_dataset.py --all
python scripts/build_splits.py --input data/processed/real_features.csv --output-dir data/splits
```

The authoritative audit artifacts are `data/processed/real_features_manifest.json` and `data/splits/split_metadata.json`. See [Task formulation](task_formulation.md), [Experiments](experiments.md), and [Limitations](limitations.md).
