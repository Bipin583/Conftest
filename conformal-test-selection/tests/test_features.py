"""Feature extraction: the 30 columns the ranker sees, and their causality.

Two properties matter more than any individual formula here. The first is
**schema stability**: the model artefact was fitted against an ordered feature
list, so a column that silently changes name or position corrupts inference
without raising anything. The second is **causality**: every history feature is
computed from strictly prior observations, and a single off-by-one in a shift
would leak the label being predicted, producing an impressive evaluation and a
useless system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.features import (
    ALL_FEATURES,
    CHANGE_FEATURES,
    CROSS_FEATURES,
    LABEL_COLUMN,
    TEST_FEATURES,
    FeatureError,
    build_features,
    dice,
    extract_change_features,
    extract_cross_features,
    extract_test_features,
    jaccard,
    path_overlap_ratio,
    tokenize_identifier,
)


def _history(labels, *, test_id: str = "t1", start: str = "2024-01-01") -> pd.DataFrame:
    """One test observed over consecutive days with the given outcomes."""
    stamps = pd.date_range(start, periods=len(labels), freq="D")
    return pd.DataFrame(
        {
            "commit_sha": [f"c{i}" for i in range(len(labels))],
            "timestamp": stamps.astype(str),
            "test_id": test_id,
            "test_path": "tests/test_alpha.py",
            "changed_file_path": "src/alpha.py",
            "duration": 1.0,
            LABEL_COLUMN: list(labels),
        }
    )


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_the_feature_schema_is_thirty_uniquely_named_columns():
    """The artefact was fitted on this exact ordered list.

    A duplicate or reordered name would not raise at load time; it would
    silently feed the model the wrong column, so the schema is pinned here.
    """
    assert ALL_FEATURES == CHANGE_FEATURES + TEST_FEATURES + CROSS_FEATURES
    assert len(ALL_FEATURES) == 30
    assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)


def test_the_cross_family_is_the_projects_own_contribution():
    """The six interaction features are what this work adds to the corpus.

    The mined corpus supplies change-only and test-only columns; nothing in it
    relates a specific change to a specific test. If this family shrank, the
    novelty claim in the README would no longer be supported by the code.
    """
    assert set(CROSS_FEATURES) == {
        "file_test_similarity",
        "co_change_frequency",
        "dependency_distance",
        "lexical_similarity",
        "path_overlap",
        "same_module",
    }


# --------------------------------------------------------------------------
# Tokenisation and similarity
# --------------------------------------------------------------------------


def test_tokenizer_splits_paths_snake_case_and_camel_case():
    """One tokeniser must handle every naming convention in a Python repo."""
    assert tokenize_identifier("tests/test_json_parser.py") == {"json", "parser"}
    assert tokenize_identifier("src/api/HttpClientView.py") == {"api", "http", "client", "view"}
    assert tokenize_identifier("src/order-service/checkout.py") == {"order", "service", "checkout"}


def test_tokenizer_drops_the_words_every_test_shares():
    """``test``, ``py`` and single characters carry no discriminative signal.

    Left in, they would make every (change, test) pair look similar to every
    other, flattening the feature that does most of the ranking work.
    """
    assert tokenize_identifier("tests/test_a.py") == set()
    assert "test" not in tokenize_identifier("tests/test_payment_gateway.py")


def test_similarities_are_zero_against_an_empty_token_set():
    """An unparseable path scores zero similarity, never a division error."""
    assert jaccard({"payment"}, set()) == 0.0
    assert dice({"payment"}, set()) == 0.0


def test_jaccard_and_dice_match_their_definitions():
    """Dice is the more forgiving of the two on size-mismatched sets."""
    left, right = {"a", "b"}, {"b", "c"}
    assert jaccard(left, right) == pytest.approx(1 / 3)
    assert dice(left, right) == pytest.approx(0.5)
    assert dice(left, right) > jaccard(left, right)


def test_path_overlap_scores_directory_proximity():
    """Co-located files score 1.0 and unrelated trees score 0.0."""
    assert path_overlap_ratio("src/api/views.py", "src/api/test_views.py") == 1.0
    assert path_overlap_ratio("src/api/views.py", "docs/guide.md") == 0.0
    # Shares only ``src``, against a three-deep path: 1 / 3.
    assert path_overlap_ratio("src/api/views.py", "src/core/deep/engine.py") == pytest.approx(1 / 3)


def test_path_overlap_accepts_windows_separators():
    """The corpus mixes separators; the feature must not depend on the OS."""
    assert path_overlap_ratio("src\\api\\views.py", "src/api/test_views.py") == 1.0


# --------------------------------------------------------------------------
# Change features
# --------------------------------------------------------------------------


def test_churn_is_the_sum_of_both_edit_directions():
    """A pure deletion is as risky as a pure addition, so both count."""
    frame = _history([0, 0])
    frame["lines_added"] = [10, 0]
    frame["lines_removed"] = [3, 7]
    change = extract_change_features(frame)
    assert change["total_churn"].tolist() == [13.0, 7.0]


def test_source_and_test_edits_are_distinguished():
    """A commit that only touches tests is a different risk from one that does not."""
    frame = _history([0, 0])
    frame["changed_file_path"] = ["src/alpha.py", "tests/test_alpha.py"]
    change = extract_change_features(frame)
    assert change["has_source_change"].tolist() == [1.0, 0.0]
    assert change["has_test_change"].tolist() == [0.0, 1.0]


def test_clock_features_come_from_the_commit_timestamp():
    """Weekend and late-night commits are a documented risk signal."""
    frame = _history([0, 0])
    frame["timestamp"] = ["2024-01-06T23:00:00Z", "2024-01-08T09:00:00Z"]  # Saturday, Monday
    change = extract_change_features(frame)
    assert change["hour_of_day"].tolist() == [23.0, 9.0]
    assert change["is_weekend"].tolist() == [1.0, 0.0]
    assert change["day_of_week"].tolist() == [5.0, 0.0]


def test_change_features_survive_a_lean_corpus():
    """Missing optional columns fill with neutral defaults rather than raising.

    The mined corpus has no ``author`` column. Extraction must degrade to a
    documented default instead of refusing the whole dataset.
    """
    lean = pd.DataFrame({"commit_sha": ["a", "b"], LABEL_COLUMN: [0, 1]})
    change = extract_change_features(lean)
    assert set(CHANGE_FEATURES) <= set(change.columns)
    assert change["authors_count"].tolist() == [1.0, 1.0]


# --------------------------------------------------------------------------
# History features: causality
# --------------------------------------------------------------------------


def test_a_tests_first_run_knows_nothing_about_itself():
    """The first observation has no history, and must not borrow its own label.

    This is the leakage test that matters most. Were the shift off by one, a
    failing first run would report a lifetime failure rate of 1.0 -- a feature
    that is a perfect copy of the target.
    """
    history = extract_test_features(_history([1, 1, 1]))
    assert history["execution_count"].iloc[0] == 0.0
    assert history["failure_rate_lifetime"].iloc[0] == 0.0
    assert history["recent_failures"].iloc[0] == 0.0


def test_lifetime_failure_rate_excludes_the_row_being_predicted():
    """Row *i* sees rows ``[0, i)`` exactly -- no more, no less.

    The final row here is the only failure. If its own label leaked in, the
    rate would be 1/4; the causal answer is 0.0.
    """
    history = extract_test_features(_history([0, 0, 0, 1]))
    assert history["failure_rate_lifetime"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert history["execution_count"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_history_of_failures_is_visible_to_the_next_run():
    """The converse: real prior failures must reach the model."""
    history = extract_test_features(_history([1, 1, 0]))
    assert history["failure_rate_lifetime"].iloc[2] == pytest.approx(1.0)
    assert history["recent_failures"].iloc[2] == pytest.approx(2.0)


def test_runs_since_last_failure_resets_on_a_failure():
    """A long green streak is the signal that a test is safe to skip."""
    history = extract_test_features(_history([0, 0, 1, 0, 0]))
    assert history["runs_since_last_failure"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_histories_of_different_tests_do_not_bleed_into_each_other():
    """Grouping is by ``test_id``; a noisy test must not incriminate a quiet one."""
    noisy = _history([1, 1, 1], test_id="flaky")
    quiet = _history([0, 0, 0], test_id="stable")
    combined = pd.concat([noisy, quiet], ignore_index=True)

    history = extract_test_features(combined)
    stable_rows = history.loc[combined["test_id"] == "stable"]
    assert stable_rows["failure_rate_lifetime"].max() == 0.0


def test_unknown_mean_duration_stays_missing_rather_than_becoming_zero():
    """A first run has no known duration, and zero is a different claim.

    Imputation is the preprocessor's job, and it uses the median. Filling the
    gap with 0.0 here would tell the model this test is instantaneous.
    """
    history = extract_test_features(_history([0, 0]))
    assert np.isnan(history["duration_mean"].iloc[0])
    assert history["duration_mean"].iloc[1] == pytest.approx(1.0)


def test_history_extraction_requires_the_columns_it_groups_on():
    """A missing ``test_id`` is a data bug that must surface immediately."""
    with pytest.raises(FeatureError, match="test_id"):
        extract_test_features(pd.DataFrame({LABEL_COLUMN: [0, 1]}))


def test_history_features_are_returned_in_the_callers_row_order():
    """Internal sorting must not permute the caller frame.

    The extractors are concatenated column-wise, so a reordered index here
    would pair each row history with another row change features.
    """
    frame = _history([0, 1, 0, 1])
    shuffled = frame.iloc[[3, 1, 0, 2]]
    history = extract_test_features(shuffled)
    assert history.index.tolist() == shuffled.index.tolist()


# --------------------------------------------------------------------------
# Cross features
# --------------------------------------------------------------------------


def test_similarity_ranks_the_test_that_names_the_changed_module_highest():
    """The interaction feature that does the ranking work must actually rank."""
    frame = pd.DataFrame(
        {
            "commit_sha": ["c1", "c1"],
            "changed_file_path": ["src/payment/gateway.py", "src/payment/gateway.py"],
            "test_path": ["tests/test_payment_gateway.py", "tests/test_user_profile.py"],
            LABEL_COLUMN: [1, 0],
        }
    )
    cross = extract_cross_features(frame)
    assert cross["file_test_similarity"].iloc[0] > cross["file_test_similarity"].iloc[1]


def test_same_module_is_an_exact_directory_match():
    """A blunt but strong signal: the test living beside the changed file."""
    frame = pd.DataFrame(
        {
            "commit_sha": ["c1", "c1"],
            "changed_file_path": ["src/api/views.py", "src/api/views.py"],
            "test_path": ["src/api/test_views.py", "tests/unit/test_views.py"],
            LABEL_COLUMN: [0, 0],
        }
    )
    assert extract_cross_features(frame)["same_module"].tolist() == [1.0, 0.0]


def test_dependency_distance_falls_back_to_path_distance():
    """Without a call graph, distance is the inverse of path overlap.

    The fallback must keep the *direction* of the real feature: near is small.
    """
    frame = pd.DataFrame(
        {
            "commit_sha": ["c1", "c1"],
            "changed_file_path": ["src/api/views.py", "src/api/views.py"],
            "test_path": ["src/api/test_views.py", "docs/test_guide.py"],
            LABEL_COLUMN: [0, 0],
        }
    )
    distance = extract_cross_features(frame)["dependency_distance"]
    assert distance.iloc[0] == pytest.approx(0.0)
    assert distance.iloc[1] == pytest.approx(1.0)


def test_an_unreachable_dependency_is_far_away_not_adjacent():
    """A graph pass reports unreachable pairs as depth 0, which reads as *closest*.

    Mapping that to a large distance is the difference between the feature
    helping and actively misleading the model.
    """
    frame = pd.DataFrame(
        {
            "commit_sha": ["c1", "c1"],
            "changed_file_path": ["src/api/views.py", "src/api/views.py"],
            "test_path": ["tests/test_views.py", "tests/test_other.py"],
            "dep_shortest_path_depth": [2, 0],
            "dep_is_reachable": [True, False],
            LABEL_COLUMN: [0, 0],
        }
    )
    distance = extract_cross_features(frame)["dependency_distance"]
    assert distance.iloc[0] == pytest.approx(2.0)
    assert distance.iloc[1] == 99.0


def test_co_change_frequency_is_computed_from_prior_rows_only():
    """The same causality rule as the history features, keyed on (file, test)."""
    frame = _history([1, 1, 1])
    co_change = extract_cross_features(frame)["co_change_frequency"]
    assert co_change.iloc[0] == 0.0, "no prior observation of this pair"
    assert co_change.iloc[1] == pytest.approx(1.0)
    assert co_change.iloc[2] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_build_features_emits_the_declared_schema_in_order(raw_frame):
    """Column order is part of the contract with the fitted artefact."""
    built = build_features(raw_frame, show_progress=False)
    assert [c for c in built.columns if c in ALL_FEATURES] == ALL_FEATURES
    assert LABEL_COLUMN in built.columns
    assert len(built) == len(raw_frame)


def test_build_features_carries_identifiers_through_unmodelled(raw_frame):
    """Evaluation groups by commit, so the identifier must survive the build."""
    built = build_features(raw_frame, show_progress=False)
    assert built["commit_sha"].tolist() == raw_frame["commit_sha"].tolist()
    assert "commit_sha" not in ALL_FEATURES, "identifiers must never be modelled"


def test_every_built_feature_is_numeric_and_finite_or_missing(raw_frame):
    """XGBoost accepts NaN; it does not accept strings or infinities."""
    built = build_features(raw_frame, show_progress=False)[ALL_FEATURES]
    assert all(pd.api.types.is_numeric_dtype(built[c]) for c in built.columns)
    assert not np.isinf(built.to_numpy(dtype=float)).any()


def test_build_features_refuses_an_unlabelled_frame(raw_frame):
    """Without the label there is nothing to learn, and no history to derive."""
    with pytest.raises(FeatureError, match=LABEL_COLUMN):
        build_features(raw_frame.drop(columns=[LABEL_COLUMN]), show_progress=False)


def test_the_test_family_is_history_only():
    """Every test-level feature must be derivable from prior runs alone."""
    assert set(TEST_FEATURES) <= set(extract_test_features(_history([0, 1, 0])).columns)
