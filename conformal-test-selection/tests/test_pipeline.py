"""The inference path, end to end, against the artefacts that ship.

Unit tests on the individual layers cannot catch the failure this module is
about: three artefacts that are each correct but wired together in the wrong
order, or applied to columns in the wrong order. The model does not raise when
handed a permuted feature matrix -- it returns confident nonsense. So these
tests exercise the real fitted pipeline and check the properties that a mis-wire
would break.

Everything here needs the committed ``.pkl`` files and is marked ``artifacts``;
on a fresh clone that has not trained yet, these skip rather than fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pipeline import (
    PipelineError,
    SelectionPipeline,
    iter_batches,
    records_from_frame,
)

pytestmark = pytest.mark.artifacts


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_the_pipeline_loads_all_four_fitted_artefacts(pipeline):
    """Every stage must be present; a missing one is a silent quality loss.

    Serving without the calibrator, for instance, still returns numbers -- the
    raw ``scale_pos_weight``-inflated scores -- and the conformal threshold
    would then be applied to a different quantity than it was fitted on.
    """
    assert pipeline.preprocessor is not None
    assert pipeline.model is not None
    assert pipeline.calibrator is not None
    assert pipeline.selector is not None
    assert pipeline.feature_columns, "the model records the columns it expects"
    assert pipeline.modelled_columns, "the preprocessor records the columns it consumes"


def test_a_missing_artefact_names_the_command_that_produces_it(tmp_path, config):
    """The error is the user's next instruction, so it must be actionable."""
    broken = {**config, "artifacts": {**config["artifacts"]}}
    broken["artifacts"]["model_path"] = str(tmp_path / "absent.pkl")

    with pytest.raises(PipelineError, match="cli.py train"):
        SelectionPipeline.load(broken)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_scores_are_probabilities_one_per_candidate(pipeline, candidate_tests):
    """The API publishes these as probabilities, so they must be in range."""
    probabilities = pipeline.probabilities(candidate_tests)
    assert probabilities.shape == (len(candidate_tests),)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0


def test_scoring_is_deterministic(pipeline, candidate_tests):
    """Two identical requests must select the same tests.

    CI reruns the same commit after an infrastructure blip often enough that a
    non-deterministic selection would be noticed as flakiness in the selector
    itself -- the one component that must not be flaky.
    """
    first = pipeline.probabilities(candidate_tests)
    second = pipeline.probabilities(candidate_tests)
    assert np.array_equal(first, second)


def test_identical_records_receive_identical_scores(pipeline, candidate_tests):
    """Row order must not influence any row score.

    A history feature computed across the request, rather than per row, would
    show up exactly here.
    """
    record = candidate_tests[0]
    probabilities = pipeline.probabilities([record, record, record])
    assert len(set(probabilities.tolist())) == 1


def test_scoring_an_empty_request_is_refused(pipeline):
    """Zero candidates is a caller bug, not an empty successful answer."""
    with pytest.raises(PipelineError, match="No records"):
        pipeline.probabilities([])


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_prediction_reports_predictions_summary_and_guarantee(pipeline, candidate_tests):
    """The response shape the API and the CLI both serialise."""
    result = pipeline.predict(candidate_tests)
    assert set(result) == {"predictions", "summary", "guarantee"}
    assert len(result["predictions"]) == len(candidate_tests)
    assert {"test_id", "failure_probability", "selected", "margin"} == set(result["predictions"][0])


def test_every_test_above_the_threshold_is_selected(pipeline, candidate_tests):
    """``margin`` is ``p - probability_floor``; a non-negative one must be kept.

    Only the implication holds, not the equivalence: ``selection.min_tests``
    tops the set up with negative-margin tests so CI never receives an
    implausibly small suite. That direction is safe -- adding tests cannot
    lower coverage -- whereas dropping a positive-margin test would break the
    guarantee outright.
    """
    result = pipeline.predict(candidate_tests, min_tests=0, max_tests=None)
    for prediction in result["predictions"]:
        assert prediction["selected"] == (prediction["margin"] >= -1e-6)


def test_the_floor_only_ever_adds_negative_margin_tests(pipeline, candidate_tests):
    """Topping up must not disturb what the conformal rule already decided."""
    strict = pipeline.predict(candidate_tests, min_tests=0, max_tests=None)
    topped = pipeline.predict(candidate_tests, min_tests=len(candidate_tests))

    strict_ids = {p["test_id"] for p in strict["predictions"] if p["selected"]}
    topped_ids = {p["test_id"] for p in topped["predictions"] if p["selected"]}
    assert strict_ids <= topped_ids


def test_cost_reduction_is_the_complement_of_the_selection_rate(pipeline, candidate_tests):
    """The headline business number is derived, never independently computed."""
    summary = pipeline.predict(candidate_tests)["summary"]
    assert summary["selection_rate"] + summary["cost_reduction"] == pytest.approx(1.0)
    assert summary["n_candidates"] == len(candidate_tests)


def test_a_floor_tops_the_selection_up_to_the_requested_size(pipeline, candidate_tests):
    """Teams often want a minimum smoke set regardless of what the rule says.

    Topping up only ever adds tests, so coverage cannot fall and the guarantee
    still holds.
    """
    floor = len(candidate_tests)
    result = pipeline.predict(candidate_tests, min_tests=floor)
    assert result["summary"]["n_selected"] == floor
    assert result["guarantee"]["holds"] is True


def test_a_cap_withdraws_the_guarantee_it_breaks(pipeline, candidate_tests):
    """This is the honesty the whole project rests on.

    A hard budget can drop a test the conformal rule selected. The response
    must then say the coverage guarantee no longer applies, rather than
    quoting 95% over a set that was trimmed afterwards.
    """
    unconstrained = pipeline.predict(candidate_tests)
    if unconstrained["summary"]["n_selected"] < 2:
        pytest.skip("rule selected too few tests for a cap to bind")

    capped = pipeline.predict(candidate_tests, max_tests=1)
    assert capped["summary"]["n_selected"] == 1
    assert capped["summary"]["budget_capped"] is True
    assert capped["guarantee"]["holds"] is False
    assert "max_tests" in capped["guarantee"]["note"]


def test_a_cap_that_does_not_bind_leaves_the_guarantee_intact(pipeline, candidate_tests):
    """A budget larger than the selection changes nothing, including the claim."""
    result = pipeline.predict(candidate_tests, max_tests=len(candidate_tests))
    assert result["summary"]["budget_capped"] is False
    assert result["guarantee"]["holds"] is True


def test_a_cap_keeps_the_riskiest_tests(pipeline, candidate_tests):
    """If tests must be dropped, drop the ones least likely to fail."""
    result = pipeline.predict(candidate_tests, max_tests=2)
    kept = [p for p in result["predictions"] if p["selected"]]
    dropped = [p for p in result["predictions"] if not p["selected"]]
    if not kept or not dropped:
        pytest.skip("nothing was dropped, so there is no ordering to check")
    assert min(p["failure_probability"] for p in kept) >= max(
        p["failure_probability"] for p in dropped
    )


# --------------------------------------------------------------------------
# Degraded input
# --------------------------------------------------------------------------


def test_a_sparse_request_is_answered_but_flagged(pipeline):
    """Imputed features predict the average test, not the one asked about.

    Refusing the request would make the API unusable from a CI job that only
    knows a file path. Answering silently would be worse. The response
    therefore carries the completeness and a degraded flag.
    """
    result = pipeline.predict([{"test_id": "tests/test_thin.py::test_case"}])
    summary = result["summary"]
    assert 0.0 <= summary["feature_completeness"] < 1.0
    assert summary["degraded"] is True


def test_a_rich_request_is_not_flagged(pipeline, config):
    """Completeness above the configured floor must not raise the flag.

    Built from the columns the preprocessor actually consumes, so this stays
    correct if the feature set changes.
    """
    record = {column: 1.0 for column in pipeline.modelled_columns}
    record["test_id"] = "tests/test_full.py::test_case"
    summary = pipeline.predict([record])["summary"]

    assert summary["feature_completeness"] == pytest.approx(1.0)
    assert summary["degraded"] is False


# --------------------------------------------------------------------------
# The guarantee statement
# --------------------------------------------------------------------------


def test_the_served_guarantee_is_the_configured_one(pipeline, config):
    """What the API claims must be what the artefact was fitted to deliver."""
    statement = pipeline.guarantee_statement()
    assert statement["type"] == config["conformal"]["guarantee"]
    assert statement["target_coverage"] == config["conformal"]["coverage"]
    assert statement["confidence"] == config["conformal"]["confidence"]
    assert statement["calibration_size"] > 0


def test_the_pac_claim_quotes_both_numbers(pipeline):
    """A training-conditional claim is meaningless with only one number.

    "95% coverage" alone is the marginal statement. The confidence level is
    what makes it a statement about the calibration set actually drawn.
    """
    claim = pipeline.guarantee_statement()["claim"]
    assert "95%" in claim
    assert "90%" in claim


def test_the_certified_coverage_meets_the_target(pipeline, config):
    """The shipped threshold certifies at or above the configured coverage."""
    certified = pipeline.guarantee_statement()["certified_coverage"]
    if certified is None:
        pytest.skip("marginal rule reports no certified coverage")
    assert certified >= config["conformal"]["coverage"]


# --------------------------------------------------------------------------
# Record helpers
# --------------------------------------------------------------------------


def test_missing_values_become_none_not_nan():
    """NaN is not valid JSON, and the API serialises these records directly."""
    frame = pd.DataFrame({"test_id": ["t"], "duration_mean": [np.nan]})
    record = records_from_frame(frame)[0]
    assert record["duration_mean"] is None


def test_batching_covers_every_item_exactly_once():
    """The batch endpoint relies on this to chunk large requests."""
    items = list(range(25))
    batches = list(iter_batches(items, 10))
    assert [len(b) for b in batches] == [10, 10, 5]
    assert [i for batch in batches for i in batch] == items
