"""
ConfTest Database CRUD Operations and Data Access Layer.

Provides high-level, type-annotated query helpers and persistence functions
for all ConfTest ORM entities with transaction safety.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import Session

from conftest.db.models import (
    Repository,
    Commit,
    ChangedFile,
    TestCase,
    TestRun,
    FeatureRecord,
    Prediction,
    SelectionDecision,
    Outcome,
)
from conftest.logging_config import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Repository CRUD Operations
# ==============================================================================

def create_repository(
    db: Session,
    full_name: str,
    url: str,
    local_path: str,
    language: str = "python",
    default_branch: str = "main",
) -> Repository:
    """Create and persist a new repository record."""
    repo = Repository(
        full_name=full_name,
        url=url,
        local_path=local_path,
        language=language,
        default_branch=default_branch,
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    logger.info(f"Registered repository: {repo.full_name} (ID: {repo.id})")
    return repo


def get_repository(db: Session, repository_id: int) -> Optional[Repository]:
    """Retrieve repository by primary key ID."""
    return db.get(Repository, repository_id)


def get_repository_by_name(db: Session, full_name: str) -> Optional[Repository]:
    """Retrieve repository by unique full name (e.g. 'pallets/flask')."""
    stmt = select(Repository).where(Repository.full_name == full_name)
    return db.execute(stmt).scalar_one_or_none()


def get_or_create_repository(
    db: Session,
    full_name: str,
    url: str,
    local_path: str,
    language: str = "python",
    default_branch: str = "main",
) -> Repository:
    """Retrieve repository if exists, or create a new record."""
    existing = get_repository_by_name(db, full_name)
    if existing:
        return existing
    return create_repository(
        db=db,
        full_name=full_name,
        url=url,
        local_path=local_path,
        language=language,
        default_branch=default_branch,
    )


def list_repositories(db: Session, skip: int = 0, limit: int = 100) -> List[Repository]:
    """List all registered repositories with pagination."""
    stmt = select(Repository).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def delete_repository(db: Session, repository_id: int) -> bool:
    """Delete a repository and all cascaded child records."""
    repo = db.get(Repository, repository_id)
    if not repo:
        return False
    db.delete(repo)
    db.commit()
    logger.info(f"Deleted repository ID: {repository_id}")
    return True


# ==============================================================================
# Commit CRUD Operations
# ==============================================================================

def create_commit(
    db: Session,
    repository_id: int,
    sha: str,
    timestamp: datetime,
    parent_sha: Optional[str] = None,
    author_hash: Optional[str] = None,
    message: Optional[str] = None,
    ci_status: str = "pending",
    total_duration: float = 0.0,
) -> Commit:
    """Create and persist a new commit record."""
    commit = Commit(
        repository_id=repository_id,
        sha=sha,
        parent_sha=parent_sha,
        timestamp=timestamp,
        author_hash=author_hash,
        message=message,
        ci_status=ci_status,
        total_duration=total_duration,
    )
    db.add(commit)
    db.commit()
    db.refresh(commit)
    return commit


def get_commit_by_sha(db: Session, sha: str) -> Optional[Commit]:
    """Retrieve a commit by its full SHA-1 hash."""
    stmt = select(Commit).where(Commit.sha == sha)
    return db.execute(stmt).scalar_one_or_none()


def list_commits_for_repo(
    db: Session, repository_id: int, skip: int = 0, limit: int = 100
) -> List[Commit]:
    """List commits chronologically for temporal analysis."""
    stmt = (
        select(Commit)
        .where(Commit.repository_id == repository_id)
        .order_by(Commit.timestamp.asc())
        .offset(skip)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def update_commit_status(
    db: Session, commit_id: int, ci_status: str, total_duration: Optional[float] = None
) -> Optional[Commit]:
    """Update CI status and total runtime of a commit."""
    commit = db.get(Commit, commit_id)
    if not commit:
        return None
    commit.ci_status = ci_status
    if total_duration is not None:
        commit.total_duration = total_duration
    db.commit()
    db.refresh(commit)
    return commit


# ==============================================================================
# Changed Files CRUD Operations
# ==============================================================================

def add_changed_files(
    db: Session, commit_id: int, files_data: List[Dict[str, Any]]
) -> List[ChangedFile]:
    """Batch insert changed file records for a commit diff."""
    records = []
    for item in files_data:
        record = ChangedFile(
            commit_id=commit_id,
            file_path=item["file_path"],
            change_type=item.get("change_type", "MODIFIED"),
            lines_added=item.get("lines_added", 0),
            lines_deleted=item.get("lines_deleted", 0),
            cyclomatic_complexity=item.get("cyclomatic_complexity", 0.0),
        )
        records.append(record)
    db.add_all(records)
    db.commit()
    return records


def get_changed_files_for_commit(db: Session, commit_id: int) -> List[ChangedFile]:
    """Retrieve all modified files associated with a commit."""
    stmt = select(ChangedFile).where(ChangedFile.commit_id == commit_id)
    return list(db.execute(stmt).scalars().all())


# ==============================================================================
# Test Case CRUD Operations
# ==============================================================================

def get_or_create_test_case(
    db: Session,
    repository_id: int,
    test_id: str,
    test_path: str,
    test_function: str,
    framework: str = "pytest",
) -> TestCase:
    """Retrieve existing test case by unique test_id or create a new entry."""
    stmt = select(TestCase).where(
        TestCase.repository_id == repository_id, TestCase.test_id == test_id
    )
    test_case = db.execute(stmt).scalar_one_or_none()
    if not test_case:
        test_case = TestCase(
            repository_id=repository_id,
            test_id=test_id,
            test_path=test_path,
            test_function=test_function,
            framework=framework,
        )
        db.add(test_case)
        db.commit()
        db.refresh(test_case)
    return test_case


def list_test_cases_for_repo(
    db: Session, repository_id: int, skip: int = 0, limit: int = 500
) -> List[TestCase]:
    """List all known test cases for a repository."""
    stmt = (
        select(TestCase)
        .where(TestCase.repository_id == repository_id)
        .offset(skip)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


# ==============================================================================
# Test Run CRUD Operations
# ==============================================================================

def record_test_runs(
    db: Session, commit_id: int, runs_data: List[Dict[str, Any]]
) -> List[TestRun]:
    """Batch insert test execution outcomes for a commit."""
    records = []
    for item in runs_data:
        record = TestRun(
            commit_id=commit_id,
            test_case_id=item["test_case_id"],
            status=item["status"].upper(),
            duration=item.get("duration", 0.0),
            retry_count=item.get("retry_count", 0),
            source=item.get("source", "ci"),
        )
        records.append(record)
    db.add_all(records)
    db.commit()
    return records


def get_test_runs_for_commit(db: Session, commit_id: int) -> List[TestRun]:
    """Retrieve test execution results for a specific commit."""
    stmt = select(TestRun).where(TestRun.commit_id == commit_id)
    return list(db.execute(stmt).scalars().all())


# ==============================================================================
# Feature Records CRUD Operations
# ==============================================================================

def save_feature_records(
    db: Session, commit_id: int, features: List[Dict[str, Any]]
) -> List[FeatureRecord]:
    """Store extracted tabular feature vectors for model training."""
    records = []
    for item in features:
        record = FeatureRecord(
            commit_id=commit_id,
            test_case_id=item["test_case_id"],
            feature_vector=item["feature_vector"],
        )
        records.append(record)
    db.add_all(records)
    db.commit()
    return records


def get_features_for_commit(db: Session, commit_id: int) -> List[FeatureRecord]:
    """Fetch feature vectors for a given commit."""
    stmt = select(FeatureRecord).where(FeatureRecord.commit_id == commit_id)
    return list(db.execute(stmt).scalars().all())


# ==============================================================================
# Prediction CRUD Operations
# ==============================================================================

def save_predictions(
    db: Session, commit_id: int, predictions_data: List[Dict[str, Any]]
) -> List[Prediction]:
    """Store raw scores, epistemic uncertainty, and calibrated probabilities."""
    records = []
    for item in predictions_data:
        record = Prediction(
            commit_id=commit_id,
            test_case_id=item["test_case_id"],
            raw_score=item["raw_score"],
            uncertainty=item["uncertainty"],
            calibrated_confidence=item["calibrated_confidence"],
            model_version=item.get("model_version", "v1.0.0"),
        )
        records.append(record)
    db.add_all(records)
    db.commit()
    return records


def get_predictions_for_commit(db: Session, commit_id: int) -> List[Prediction]:
    """Retrieve all test predictions for a commit."""
    stmt = select(Prediction).where(Prediction.commit_id == commit_id)
    return list(db.execute(stmt).scalars().all())


# ==============================================================================
# Selection Decision CRUD Operations
# ==============================================================================

def save_selection_decision(
    db: Session,
    commit_id: int,
    mode: str,
    abstained: bool,
    uncertainty_score: float,
    threshold_used: float,
    selected_count: int,
    total_count: int,
    estimated_saving: float = 0.0,
    reasons: Optional[dict] = None,
) -> SelectionDecision:
    """Record the final selective prediction policy decision."""
    decision = SelectionDecision(
        commit_id=commit_id,
        mode=mode,
        abstained=abstained,
        uncertainty_score=uncertainty_score,
        threshold_used=threshold_used,
        selected_count=selected_count,
        total_count=total_count,
        estimated_saving=estimated_saving,
        reasons=reasons or {},
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision


def get_decision_for_commit(db: Session, commit_id: int) -> Optional[SelectionDecision]:
    """Fetch the selection decision for a commit."""
    stmt = select(SelectionDecision).where(SelectionDecision.commit_id == commit_id)
    return db.execute(stmt).scalar_one_or_none()


# ==============================================================================
# Outcome Evaluation CRUD Operations
# ==============================================================================

def save_outcome(
    db: Session,
    commit_id: int,
    detected_failures: int,
    selected_duration: float,
    ground_truth_complete: bool,
    actual_failures: Optional[int] = None,
    missed_failures: Optional[int] = None,
    full_duration: Optional[float] = None,
    time_reduction_ratio: Optional[float] = None,
) -> Outcome:
    """
    Store post-execution evaluation and savings audit.

    Callers must state, via `ground_truth_complete`, whether the full suite ran.
    The four full-suite-dependent arguments are then checked against that claim:
    supplying them without the full suite is rejected rather than stored, and
    omitting them with the full suite is rejected too. There is deliberately no
    default for the escape count -- a caller that has not measured it cannot
    accidentally record a zero.

    Args:
        detected_failures: Failures seen in the tests that actually ran.
        selected_duration: Wall time of the tests that actually ran.
        ground_truth_complete: True iff the full suite also ran.
        actual_failures: Full-suite failure count. Required iff complete.
        missed_failures: Escapes (actual - detected). Required iff complete.
        full_duration: Full-suite wall time. Required iff complete.
        time_reduction_ratio: Measured 1 - selected/full time. Required iff
            complete. Must be a *time* ratio; a selected/total count ratio is a
            different quantity and does not belong in this column.

    Raises:
        ValueError: if the supplied fields contradict `ground_truth_complete`.
    """
    verified_fields = {
        "actual_failures": actual_failures,
        "missed_failures": missed_failures,
        "full_duration": full_duration,
        "time_reduction_ratio": time_reduction_ratio,
    }
    if ground_truth_complete:
        absent = sorted(name for name, val in verified_fields.items() if val is None)
        if absent:
            raise ValueError(
                f"ground_truth_complete=True requires measured values for {absent}; "
                "pass ground_truth_complete=False if the full suite did not run."
            )
    else:
        supplied = sorted(name for name, val in verified_fields.items() if val is not None)
        if supplied:
            raise ValueError(
                f"ground_truth_complete=False cannot record {supplied}: without a "
                "full-suite run these are unobserved. A selective run cannot show "
                "that an unselected test would have passed."
            )

    outcome = Outcome(
        commit_id=commit_id,
        ground_truth_complete=ground_truth_complete,
        detected_failures=detected_failures,
        selected_duration=selected_duration,
        actual_failures=actual_failures,
        missed_failures=missed_failures,
        full_duration=full_duration,
        time_reduction_ratio=time_reduction_ratio,
    )
    db.add(outcome)
    db.commit()
    db.refresh(outcome)
    return outcome


def get_outcome_for_commit(db: Session, commit_id: int) -> Optional[Outcome]:
    """Retrieve outcome evaluation for a commit."""
    stmt = select(Outcome).where(Outcome.commit_id == commit_id)
    return db.execute(stmt).scalar_one_or_none()


def get_aggregate_metrics_summary(db: Session) -> Dict[str, Any]:
    """
    Compute global aggregate metrics across historical outcomes.

    Safety metrics are computed *only* over outcomes with
    `ground_truth_complete`, because recall and missed-failure rate both divide
    by a full-suite failure count that an unverified run never measured.
    Unverified outcomes are counted and reported separately rather than folded
    in at zero escapes, which would drive the aggregate recall towards 1.0 for
    no reason other than that nobody looked.

    Returns:
        Aggregates over verified outcomes. The safety and saving figures are
        None when no verified outcome exists -- an average over an empty set is
        not zero, and a recall over no observations is not 1.0.
    """
    verified = Outcome.ground_truth_complete.is_(True)

    total_outcomes = db.execute(select(func.count(Outcome.id))).scalar() or 0
    stmt = select(
        func.count(Outcome.id).label("verified_commits"),
        func.sum(Outcome.actual_failures).label("total_actual_failures"),
        func.sum(Outcome.detected_failures).label("total_detected_failures"),
        func.sum(Outcome.missed_failures).label("total_missed_failures"),
        func.sum(Outcome.full_duration).label("total_full_duration"),
        func.sum(Outcome.selected_duration).label("total_selected_duration"),
        func.avg(Outcome.time_reduction_ratio).label("avg_reduction_ratio"),
    ).where(verified)
    row = db.execute(stmt).first()
    verified_commits = (row.verified_commits if row else 0) or 0

    if not verified_commits:
        return {
            "total_commits": total_outcomes,
            "verified_commits": 0,
            "unverified_commits": total_outcomes,
            "total_actual_failures": None,
            "total_detected_failures": None,
            "total_missed_failures": None,
            "failure_recall": None,
            "missed_failure_rate": None,
            "average_time_reduction": None,
            "total_hours_saved": None,
        }

    total_actual = row.total_actual_failures or 0
    total_detected = row.total_detected_failures or 0
    total_missed = row.total_missed_failures or 0
    full_sec = row.total_full_duration or 0.0
    sel_sec = row.total_selected_duration or 0.0

    # With no failures in the verified window there is nothing to have caught or
    # missed; a ratio there would be 0/0, not a perfect score.
    failure_recall = round(total_detected / total_actual, 4) if total_actual > 0 else None
    missed_rate = round(total_missed / total_actual, 4) if total_actual > 0 else None
    saved_hours = max(0.0, (full_sec - sel_sec) / 3600.0)

    return {
        "total_commits": total_outcomes,
        "verified_commits": verified_commits,
        "unverified_commits": total_outcomes - verified_commits,
        "total_actual_failures": total_actual,
        "total_detected_failures": total_detected,
        "total_missed_failures": total_missed,
        "failure_recall": failure_recall,
        "missed_failure_rate": missed_rate,
        "average_time_reduction": round(row.avg_reduction_ratio or 0.0, 4),
        "total_hours_saved": round(saved_hours, 2),
    }
