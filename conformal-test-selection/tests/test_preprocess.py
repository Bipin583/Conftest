"""Splitting and preprocessing: where a leak or a schema drift would hide.

The temporal split is the reason the reported numbers mean anything. Test
selection is deployed on commits the model has never seen, so a random split
would let a commit contribute rows to both training and test, and the model
would be scored partly on rows it memorised. Everything in the first section
exists to make that impossible.

The second section covers the fitted transformer, whose failure modes are
quieter: an unseen repository at serving time, or a one-hot column name that
XGBoost refuses to load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.features import ALL_FEATURES, CROSS_FEATURES, LABEL_COLUMN
from data.preprocess import (
    CATEGORICAL_CANDIDATES,
    EXTERNAL_ALIASES,
    EXTERNAL_UNAVAILABLE,
    PreprocessError,
    _sanitize_feature_names,
    adapt_external_frame,
    build_preprocessor,
    temporal_group_split,
)


def _corpus(n_commits: int = 20, tests_per_commit: int = 3) -> pd.DataFrame:
    """A frame shaped like the corpus: several tests observed per commit."""
    rows = []
    for index in range(n_commits):
        for test in range(tests_per_commit):
            rows.append(
                {
                    "commit_sha": f"c{index:03d}",
                    "timestamp": f"2024-01-{index + 1:02d}T12:00:00Z",
                    "test_id": f"tests/test_{test}.py::test_case",
                    "test_path": f"tests/test_{test}.py",
                    "changed_file_path": "src/module.py",
                    "repo": "acme/service",
                    "lines_added": float(index),
                    "duration_mean": 1.0 + test,
                    LABEL_COLUMN: int(index % 5 == 0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The temporal split
# --------------------------------------------------------------------------


def test_no_commit_appears_on_both_sides_of_a_split():
    """Grouping by commit is what makes the evaluation honest.

    Rows of one commit share every change-level feature and are highly
    correlated. Splitting them across train and test would let the model score
    on rows it effectively memorised, inflating recall for free.
    """
    train, val, test = temporal_group_split(_corpus())
    train_commits = set(train["commit_sha"])
    val_commits = set(val["commit_sha"])
    test_commits = set(test["commit_sha"])

    assert not train_commits & val_commits
    assert not val_commits & test_commits
    assert not train_commits & test_commits


def test_every_evaluation_commit_is_in_the_future_of_training():
    """Chronological order, not random assignment: the deployment condition.

    A model selecting tests for tomorrow has only yesterday to learn from.
    Any overlap here would be training on the future.
    """
    train, val, test = temporal_group_split(_corpus())
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_the_split_cuts_commits_at_the_configured_ratios():
    """70/15/15 of *commits*, which is what the ratios are defined over.

    Row counts drift from the ratios because commits carry different numbers
    of tests; the commit counts are the quantity actually controlled.
    """
    frame = _corpus(n_commits=100)
    train, val, test = temporal_group_split(frame, ratios=(0.70, 0.15, 0.15))
    assert train["commit_sha"].nunique() == 70
    assert val["commit_sha"].nunique() == 15
    assert test["commit_sha"].nunique() == 15


def test_the_split_keeps_every_row():
    """A dropped row is a silently smaller dataset, never reported anywhere."""
    frame = _corpus()
    parts = temporal_group_split(frame)
    assert sum(len(part) for part in parts) == len(frame)


def test_splitting_an_empty_frame_is_refused():
    """An empty frame means an upstream stage failed, which must not pass through."""
    with pytest.raises(PreprocessError, match="empty"):
        temporal_group_split(pd.DataFrame())


def test_splitting_without_a_grouping_column_is_refused():
    """Without ``commit_sha`` the split cannot be group-aware, so it must stop."""
    with pytest.raises(PreprocessError, match="commit_sha"):
        temporal_group_split(pd.DataFrame({"timestamp": ["2024-01-01"], "x": [1]}))


def test_a_corpus_too_small_to_split_is_refused():
    """Fewer than three commits cannot produce three non-degenerate splits."""
    with pytest.raises(PreprocessError, match="at least 3"):
        temporal_group_split(_corpus(n_commits=2))


# --------------------------------------------------------------------------
# The fitted transformer
# --------------------------------------------------------------------------


def test_numeric_gaps_are_filled_with_the_median_not_the_mean():
    """Churn and duration are heavy-tailed, so the mean chases outliers.

    One 10,000-line refactor would drag a mean imputation far from anything
    typical; the median stays where the data is.
    """
    frame = pd.DataFrame({"churn": [1.0, 2.0, 3.0, 10_000.0, np.nan]})
    transformer = build_preprocessor(["churn"], [])
    transformed = transformer.fit_transform(frame)

    median = frame["churn"].median()
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    restored = transformed[-1, 0] * np.sqrt(scaler.var_[0]) + scaler.mean_[0]
    assert restored == pytest.approx(median)


def test_numeric_columns_are_standardised():
    """Scaling keeps the artefact reusable if the model family ever changes."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    transformed = build_preprocessor(["x"], []).fit_transform(frame)
    assert transformed.mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed.std() == pytest.approx(1.0, abs=1e-9)


def test_an_unseen_category_at_serving_time_does_not_crash_inference():
    """A new repository must degrade to all-zero indicators, not raise.

    This is a serving-time failure the training run would never surface: the
    API is asked about a repo that did not exist when the model was fitted.
    """
    train = pd.DataFrame({"repo": ["acme/a", "acme/b", "acme/a"]})
    transformer = build_preprocessor([], ["repo"]).fit(train)
    encoded = transformer.transform(pd.DataFrame({"repo": ["brand/new"]}))

    assert encoded.shape[0] == 1
    assert not np.isnan(encoded).any()


def test_the_transformer_keeps_a_stable_column_count_across_calls():
    """Training and serving must produce the same width, or the model rejects it."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "repo": ["a", "b", "a"]})
    transformer = build_preprocessor(["x"], ["repo"]).fit(train)
    width = transformer.transform(train).shape[1]
    assert transformer.transform(train.head(1)).shape[1] == width


def test_a_preprocessor_with_no_columns_is_refused():
    """An empty feature list is a configuration bug, not an identity transform."""
    with pytest.raises(PreprocessError, match="No columns"):
        build_preprocessor([], [])


def test_categorical_candidates_are_the_low_cardinality_columns():
    """Only these are one-hot encoded; anything else would explode the width."""
    assert CATEGORICAL_CANDIDATES == ["repo", "mutant_operator"]


# --------------------------------------------------------------------------
# Feature names
# --------------------------------------------------------------------------


def test_comparison_operators_are_stripped_from_generated_names():
    """XGBoost rejects ``<`` and ``>`` outright; one-hot encoding produces them.

    The corpus has a ``mutant_operator`` category literally named ``cmp_<=_to_>=``,
    so this is the exact string that took a training run down before the
    sanitiser existed.
    """
    cleaned = _sanitize_feature_names(["cmp_<=_to_>="])[0]
    assert "<" not in cleaned and ">" not in cleaned
    assert "lt" in cleaned and "gt" in cleaned


def test_sanitising_never_merges_two_distinct_columns():
    """Two names that collapse to the same string must stay distinguishable.

    A collision here would map two different one-hot columns onto one feature,
    which the model would read as a single, meaningless indicator.
    """
    cleaned = _sanitize_feature_names(["a b", "a_b", "a<b", "a_lt_b"])
    assert len(set(cleaned)) == len(cleaned)


def test_sanitising_preserves_order():
    """The name list is positional: it labels the transformed matrix columns."""
    names = ["zulu", "alpha", "mike"]
    assert _sanitize_feature_names(names) == names


# --------------------------------------------------------------------------
# Adapting the mined corpus
# --------------------------------------------------------------------------


@pytest.fixture
def external() -> pd.DataFrame:
    """A frame using the mined corpus column names."""
    return pd.DataFrame(
        {
            "diff_lines_added": [10, 4],
            "diff_lines_deleted": [2, 1],
            "diff_total_churn": [12, 5],
            "hist_avg_duration": [1.5, 0.4],
            "hist_lifetime_failure_rate": [0.1, 0.0],
            "changed_file_path": ["src/payment/gateway.py", "src/user/profile.py"],
            "test_path": ["tests/test_payment_gateway.py", "tests/test_user_profile.py"],
            "commit_timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "repo": ["acme/service", "acme/service"],
            LABEL_COLUMN: [1, 0],
        }
    )


def test_corpus_columns_are_renamed_onto_the_project_schema(external):
    """The corpus is reused, not re-mined, so the names must be translated."""
    adapted, columns = adapt_external_frame(external)
    assert adapted["lines_added"].tolist() == [10, 4]
    assert adapted["duration_mean"].tolist() == [1.5, 0.4]
    assert {"lines_added", "lines_removed", "duration_mean"} <= set(columns)


def test_the_cross_features_are_recomputed_rather_than_imported(external):
    """They do not exist in the corpus, which is precisely why they add value."""
    adapted, columns = adapt_external_frame(external)
    assert set(CROSS_FEATURES) <= set(columns)
    assert adapted["file_test_similarity"].iloc[0] > 0.0


def test_the_corpus_timestamp_becomes_the_split_key(external):
    """``temporal_group_split`` orders on ``timestamp``; the corpus names it differently."""
    adapted, _ = adapt_external_frame(external)
    assert "timestamp" in adapted.columns


def test_an_unlabelled_corpus_is_refused(external):
    """Without the outcome column there is nothing to calibrate against."""
    with pytest.raises(PreprocessError, match=LABEL_COLUMN):
        adapt_external_frame(external.drop(columns=[LABEL_COLUMN]))


def test_every_unsupported_schema_feature_has_a_written_reason():
    """Dropping a feature silently is how a model quietly gets worse.

    ``EXTERNAL_UNAVAILABLE`` is the record of what the corpus cannot supply and
    why -- for example, its timestamps are synthetic mutation-ordering stamps,
    so calendar windows are meaningless on it.
    """
    assert set(EXTERNAL_UNAVAILABLE) <= set(ALL_FEATURES)
    assert all(reason.strip() for reason in EXTERNAL_UNAVAILABLE.values())
    # The aliased and unavailable sets must not overlap: a feature is either
    # mapped or explained, never both.
    assert not set(EXTERNAL_ALIASES) & set(EXTERNAL_UNAVAILABLE)
