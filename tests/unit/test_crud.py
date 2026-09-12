"""
Unit tests for ConfTest database CRUD operations and query helpers.
"""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from conftest.db import crud


def test_repository_crud_lifecycle(db_session: Session):
    """Test repository creation, lookup, listing, and deletion."""
    repo = crud.create_repository(
        db=db_session,
        full_name="scikit-learn/scikit-learn",
        url="https://github.com/scikit-learn/scikit-learn",
        local_path="/tmp/repos/sklearn",
    )
    assert repo.id is not None

    fetched = crud.get_repository(db_session, repo.id)
    assert fetched is not None
    assert fetched.full_name == "scikit-learn/scikit-learn"

    by_name = crud.get_repository_by_name(db_session, "scikit-learn/scikit-learn")
    assert by_name is not None
    assert by_name.id == repo.id

    all_repos = crud.list_repositories(db_session)
    assert len(all_repos) >= 1

    deleted = crud.delete_repository(db_session, repo.id)
    assert deleted is True
    assert crud.get_repository(db_session, repo.id) is None


def test_commit_and_diff_crud(db_session: Session):
    """Test commit creation, status update, and changed file persistence."""
    repo = crud.create_repository(
        db=db_session,
        full_name="encode/starlette",
        url="https://github.com/encode/starlette",
        local_path="/tmp/repos/starlette",
    )

    commit = crud.create_commit(
        db=db_session,
        repository_id=repo.id,
        sha="c" * 40,
        timestamp=datetime.utcnow(),
        message="fix: resolve connection keepalive race condition",
    )
    assert commit.ci_status == "pending"

    updated = crud.update_commit_status(db_session, commit.id, ci_status="passed", total_duration=8.4)
    assert updated.ci_status == "passed"
    assert updated.total_duration == 8.4

    # Add changed files
    diff_data = [
        {"file_path": "starlette/responses.py", "lines_added": 12, "lines_deleted": 4, "cyclomatic_complexity": 2.1},
        {"file_path": "tests/test_responses.py", "lines_added": 25, "lines_deleted": 0, "cyclomatic_complexity": 1.0},
    ]
    files = crud.add_changed_files(db_session, commit.id, diff_data)
    assert len(files) == 2

    retrieved_files = crud.get_changed_files_for_commit(db_session, commit.id)
    assert len(retrieved_files) == 2


def test_predictions_decisions_and_outcomes_crud(db_session: Session):
    """Test full decision and evaluation outcome persistence and aggregation."""
    repo = crud.create_repository(
        db=db_session,
        full_name="pytest-dev/pytest",
        url="https://github.com/pytest-dev/pytest",
        local_path="/tmp/repos/pytest",
    )

    commit = crud.create_commit(
        db=db_session,
        repository_id=repo.id,
        sha="d" * 40,
        timestamp=datetime.utcnow(),
    )

    test_case = crud.get_or_create_test_case(
        db=db_session,
        repository_id=repo.id,
        test_id="testing/test_assert.py::test_repr_compare",
        test_path="testing/test_assert.py",
        test_function="test_repr_compare",
    )

    # Save Prediction
    preds = crud.save_predictions(
        db=db_session,
        commit_id=commit.id,
        predictions_data=[
            {
                "test_case_id": test_case.id,
                "raw_score": 0.92,
                "uncertainty": 0.05,
                "calibrated_confidence": 0.95,
                "model_version": "v1.0.0",
            }
        ],
    )
    assert len(preds) == 1

    # Save Decision
    decision = crud.save_selection_decision(
        db=db_session,
        commit_id=commit.id,
        mode="FAST_SELECTED",
        abstained=False,
        uncertainty_score=0.05,
        threshold_used=0.15,
        selected_count=1,
        total_count=10,
        estimated_saving=0.85,
    )
    assert decision.abstained is False

    # Save Outcome. The full suite ran alongside the selection here, so the
    # full-suite fields are measurements and the outcome is verifiable.
    outcome = crud.save_outcome(
        db=db_session,
        commit_id=commit.id,
        detected_failures=1,
        selected_duration=15.0,
        ground_truth_complete=True,
        actual_failures=1,
        missed_failures=0,
        full_duration=100.0,
        time_reduction_ratio=0.85,
    )
    assert outcome.detected_failures == 1
    assert outcome.ground_truth_complete is True

    # Verify Summary Aggregates
    summary = crud.get_aggregate_metrics_summary(db_session)
    assert summary["total_commits"] == 1
    assert summary["verified_commits"] == 1
    assert summary["unverified_commits"] == 0
    assert summary["failure_recall"] == 1.0
    assert summary["missed_failure_rate"] == 0.0
    assert summary["average_time_reduction"] == 0.85


def test_save_outcome_refuses_unmeasured_safety_fields(db_session: Session):
    """A caller without a full-suite run must not be able to record 0 escapes."""
    repo = crud.create_repository(
        db=db_session,
        full_name="sample/unverified-outcomes",
        url="https://github.com/sample/unverified-outcomes",
        local_path="./tests/sample_suite",
    )
    commit = crud.create_commit(
        db=db_session, repository_id=repo.id, sha="c" * 40, timestamp=datetime.utcnow()
    )

    # Claiming zero escapes without the full suite is rejected, not stored.
    with pytest.raises(ValueError, match="cannot record"):
        crud.save_outcome(
            db=db_session,
            commit_id=commit.id,
            detected_failures=0,
            selected_duration=15.0,
            ground_truth_complete=False,
            missed_failures=0,
        )

    # And claiming verification without supplying the measurements is too.
    with pytest.raises(ValueError, match="requires measured values"):
        crud.save_outcome(
            db=db_session,
            commit_id=commit.id,
            detected_failures=0,
            selected_duration=15.0,
            ground_truth_complete=True,
        )

    outcome = crud.save_outcome(
        db=db_session,
        commit_id=commit.id,
        detected_failures=2,
        selected_duration=15.0,
        ground_truth_complete=False,
    )
    assert outcome.ground_truth_complete is False
    assert outcome.missed_failures is None
    assert outcome.actual_failures is None
    assert outcome.time_reduction_ratio is None

    # The unverified row must not read out as a perfect safety record.
    summary = crud.get_aggregate_metrics_summary(db_session)
    assert summary["verified_commits"] == 0
    assert summary["unverified_commits"] == 1
    assert summary["failure_recall"] is None
    assert summary["missed_failure_rate"] is None
