"""
ConfTest Relational Database Models (SQLAlchemy 2.0).

Defines the complete schema for repositories, commits, file diffs, test cases,
test runs, extracted feature vectors, ML predictions, selective decisions, and outcomes.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conftest.db.base import Base


class Repository(Base):
    """Represents a monitored software repository."""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    language: Mapped[str] = mapped_column(String(64), default="python", nullable=False)
    default_branch: Mapped[str] = mapped_column(String(64), default="main", nullable=False)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commits: Mapped[List["Commit"]] = relationship(
        "Commit", back_populates="repository", cascade="all, delete-orphan"
    )
    test_cases: Mapped[List["TestCase"]] = relationship(
        "TestCase", back_populates="repository", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Repository(id={self.id}, full_name='{self.full_name}')>"


class Commit(Base):
    """Represents an ingested Git commit or Pull Request head commit."""

    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sha: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    parent_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    author_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ci_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    total_duration: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    repository: Mapped["Repository"] = relationship("Repository", back_populates="commits")
    changed_files: Mapped[List["ChangedFile"]] = relationship(
        "ChangedFile", back_populates="commit", cascade="all, delete-orphan"
    )
    test_runs: Mapped[List["TestRun"]] = relationship(
        "TestRun", back_populates="commit", cascade="all, delete-orphan"
    )
    feature_records: Mapped[List["FeatureRecord"]] = relationship(
        "FeatureRecord", back_populates="commit", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="commit", cascade="all, delete-orphan"
    )
    selection_decision: Mapped[Optional["SelectionDecision"]] = relationship(
        "SelectionDecision", back_populates="commit", uselist=False, cascade="all, delete-orphan"
    )
    outcome: Mapped[Optional["Outcome"]] = relationship(
        "Outcome", back_populates="commit", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Commit(id={self.id}, sha='{self.sha[:8]}', status='{self.ci_status}')>"


class ChangedFile(Base):
    """Represents a modified source or test file in a commit diff."""

    __tablename__ = "changed_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    change_type: Mapped[str] = mapped_column(String(16), default="MODIFIED", nullable=False)
    lines_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lines_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cyclomatic_complexity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="changed_files")

    def __repr__(self) -> str:
        return f"<ChangedFile(id={self.id}, path='{self.file_path}', type='{self.change_type}')>"


class TestCase(Base):
    """Represents a unique discovered regression test case."""

    __tablename__ = "test_cases"
    __test__ = False  # Prevent pytest from treating ORM model as a test suite

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_id: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    test_path: Mapped[str] = mapped_column(String(512), nullable=False)
    test_function: Mapped[str] = mapped_column(String(256), nullable=False)
    framework: Mapped[str] = mapped_column(String(32), default="pytest", nullable=False)
    average_duration: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    flaky_indicator: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Constraints
    __table_args__ = (
        UniqueConstraint("repository_id", "test_id", name="uq_repo_test_id"),
    )

    # Relationships
    repository: Mapped["Repository"] = relationship("Repository", back_populates="test_cases")
    test_runs: Mapped[List["TestRun"]] = relationship(
        "TestRun", back_populates="test_case", cascade="all, delete-orphan"
    )
    feature_records: Mapped[List["FeatureRecord"]] = relationship(
        "FeatureRecord", back_populates="test_case", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="test_case", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<TestCase(id={self.id}, test_id='{self.test_id}')>"


class TestRun(Base):
    """Represents an execution instance of a test case on a specific commit."""

    __tablename__ = "test_runs"
    __test__ = False  # Prevent pytest from treating ORM model as a test suite

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # PASSED, FAILED, SKIPPED, ERROR
    duration: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="ci", nullable=False)  # ci, local, replay
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="test_runs")
    test_case: Mapped["TestCase"] = relationship("TestCase", back_populates="test_runs")

    def __repr__(self) -> str:
        return f"<TestRun(id={self.id}, status='{self.status}', duration={self.duration:.3f}s)>"


class FeatureRecord(Base):
    """Stores the extracted feature matrix record for a (commit, test_case) pair."""

    __tablename__ = "feature_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    feature_vector: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="feature_records")
    test_case: Mapped["TestCase"] = relationship("TestCase", back_populates="feature_records")

    __table_args__ = (
        UniqueConstraint("commit_id", "test_case_id", name="uq_commit_test_feature"),
    )

    def __repr__(self) -> str:
        return f"<FeatureRecord(id={self.id}, commit_id={self.commit_id}, test_id={self.test_case_id})>"


class Prediction(Base):
    """Stores raw score, epistemic uncertainty, and calibrated confidence per test per commit."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="predictions")
    test_case: Mapped["TestCase"] = relationship("TestCase", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("commit_id", "test_case_id", name="uq_commit_test_prediction"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(commit_id={self.commit_id}, test_id={self.test_case_id}, "
            f"conf={self.calibrated_confidence:.3f}, unc={self.uncertainty:.3f})>"
        )


class SelectionDecision(Base):
    """Records the final RTS decision, policy mode, uncertainty score, and selection rationale."""

    __tablename__ = "selection_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    mode: Mapped[str] = mapped_column(String(32), nullable=False)  # FAST_SELECTED or SAFE_FULL_SUITE
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    uncertainty_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_saving: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reasons: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="selection_decision")

    def __repr__(self) -> str:
        return (
            f"<SelectionDecision(commit_id={self.commit_id}, mode='{self.mode}', "
            f"abstained={self.abstained}, selected={self.selected_count}/{self.total_count})>"
        )


class Outcome(Base):
    """
    Records the post-execution ground truth evaluation and savings audit.

    A selective run observes only the tests it selected. Whether an unselected
    test would have failed, and how long the whole suite would have taken, are
    simply not measured on that run -- they are knowable only by also running
    the full suite. So the fields that depend on the unselected tests are
    nullable, and `ground_truth_complete` records whether they were observed.

    This is not a formality. `missed_failures` feeds the failure-recall and
    missed-failure-rate figures that this project reports as its safety claim.
    Defaulting it to 0 on an unverified run does not mean "no escapes were
    found"; it asserts that none exist, on evidence that could not have
    detected one. Aggregates must filter on `ground_truth_complete`.
    """

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    # True only when the full suite also ran, making the nullable fields below
    # measurements rather than guesses.
    ground_truth_complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    # Observed on any run: these come from the tests that actually executed.
    detected_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    selected_duration: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Require the full suite. NULL when ground_truth_complete is False.
    actual_failures: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    missed_failures: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    full_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_reduction_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    commit: Mapped["Commit"] = relationship("Commit", back_populates="outcome")

    def __repr__(self) -> str:
        if not self.ground_truth_complete:
            return (
                f"<Outcome(commit_id={self.commit_id}, detected={self.detected_failures}, "
                f"missed=UNVERIFIED, saving=UNVERIFIED)>"
            )
        return (
            f"<Outcome(commit_id={self.commit_id}, detected={self.detected_failures}/{self.actual_failures}, "
            f"missed={self.missed_failures}, saving={self.time_reduction_ratio:.1%})>"
        )
