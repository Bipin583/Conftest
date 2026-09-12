# Canonical Feature Schema

ConfTest represents each candidate `(commit, test)` pair with 32 numeric features. The executable authority for names and order is `src/conftest/features/pipeline.py::FEATURE_NAMES`; this page describes that list but does not supersede it. Serialized arrays, trained models, calibrators, and explainers depend on the exact order below.

## Canonical order

### Diff and churn (1–12)

| # | Identifier | Meaning |
|---:|---|---|
| 1 | `diff_lines_added` | Lines added across changed files. |
| 2 | `diff_lines_deleted` | Lines deleted across changed files. |
| 3 | `diff_total_churn` | Added plus deleted lines. |
| 4 | `diff_num_files_changed` | Number of changed files. |
| 5 | `diff_num_src_files` | Changed non-test source files. |
| 6 | `diff_num_test_files` | Changed test files. |
| 7 | `diff_has_python` | `1.0` when the diff contains Python files. |
| 8 | `diff_has_config` | `1.0` when the diff contains recognized configuration files. |
| 9 | `diff_msg_length` | Commit-message character count. |
| 10 | `diff_msg_word_count` | Commit-message word count. |
| 11 | `diff_is_fix_commit` | Keyword indicator for a fix/bug/patch message. |
| 12 | `diff_is_refactor_commit` | Keyword indicator for a refactor/clean message. |

### Test-file AST (13–18)

| # | Identifier | Meaning |
|---:|---|---|
| 13 | `ast_test_file_functions_count` | Functions declared in the candidate test file. |
| 14 | `ast_test_file_classes_count` | Classes declared in the candidate test file. |
| 15 | `ast_test_file_imports_count` | Imports in the candidate test file. |
| 16 | `ast_test_file_complexity` | Test-file decision complexity. |
| 17 | `ast_test_is_parameterized` | `1.0` when the collected test node ID carries parameter brackets. |
| 18 | `ast_test_func_name_length` | Candidate test-function/node-name length. |

### Static dependency graph (19–24)

| # | Identifier | Meaning |
|---:|---|---|
| 19 | `dep_is_direct_import` | Candidate test directly imports a changed module. |
| 20 | `dep_name_heuristic_coupled` | Test/source naming heuristic indicates coupling. |
| 21 | `dep_shortest_path_depth` | Shortest static dependency path; the extractor uses a sentinel for disconnected nodes. |
| 22 | `dep_is_reachable` | A directed dependency path reaches a changed module. |
| 23 | `dep_max_reverse_dependencies` | Maximum reverse-dependency count among changed modules. |
| 24 | `dep_test_total_out_degree` | Candidate test module's dependency out-degree. |

### Prior execution history (25–32)

| # | Identifier | Meaning |
|---:|---|---|
| 25 | `hist_total_prior_runs` | Executions strictly before the target commit. |
| 26 | `hist_prior_failures` | Prior failed executions. |
| 27 | `hist_lifetime_failure_rate` | Prior failures divided by prior runs. |
| 28 | `hist_recent_10_failure_rate` | Failure rate over at most ten prior runs. |
| 29 | `hist_avg_duration` | Mean prior duration in seconds. |
| 30 | `hist_flaky_score` | Historical retry/status-instability score. |
| 31 | `hist_has_ever_failed` | `1.0` after at least one prior failure. |
| 32 | `hist_changed_files_prior_mod_count` | Prior modifications involving the current changed paths. |

History queries are timestamp-filtered to prevent future outcomes from entering a feature row. See `src/conftest/features/history_features.py` for the executable definitions.

## Missing history and numeric normalization

`FeatureExtractionPipeline.extract_features_for_pair()` uses a cold-start fallback when no database session, test-case ID, or commit timestamp is available. History values are `0.0`, except `hist_avg_duration`, which is initialized to `0.05` seconds. After extraction, any absent or NaN canonical value becomes `0.0`.

`FeatureExtractionPipeline.to_feature_vector()` creates a `float32` NumPy array by iterating `FEATURE_NAMES`. Dictionaries are therefore convenient inputs, but dictionary insertion order is not the model contract; `FEATURE_NAMES` is.

## Artifact compatibility

Changing a name, definition, or position changes the model input contract. Such a change requires coordinated regeneration of datasets, ensemble members, calibrator, tuned policy, evaluation reports, and any stored feature records. Adding a 33rd feature is also a schema migration, not a documentation-only change.

The compatibility module `conftest.features.extractor` re-exports the canonical pipeline. New code should import from `conftest.features.pipeline`; both paths intentionally resolve to the same implementation.

## Measured data caveat

The committed training harvest contains 13 constant columns, including all 12 diff/churn columns. A model cannot learn variation from a constant column, regardless of the feature's intended semantics. This is a limitation of the committed data/artifacts, not a reason to silently remove or reorder columns. See [Limitations](limitations.md) and the producer-backed [Feature ablation study](feature_ablation.md).
