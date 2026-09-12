"""
ConfTest Pydantic Request & Response Data Schemas.

Defines strict type-safe schemas for API request validation, model outputs,
selective prediction decisions, SHAP explanations, and repository metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ==============================================================================
# Common & Diff Schemas
# ==============================================================================

class ChangedFileSchema(BaseModel):
    file_path: str = Field(..., description="Relative path of the modified file.")
    change_type: str = Field(default="M", description="Git change type: A (Added), M (Modified), D (Deleted).")
    lines_added: int = Field(default=0, ge=0, description="Lines of code added.")
    lines_deleted: int = Field(default=0, ge=0, description="Lines of code deleted.")


# ==============================================================================
# Test Selection Schemas
# ==============================================================================

class SelectRequestSchema(BaseModel):
    repository_name: str = Field(default="local/sample-app", description="Registered repository identifier.")
    commit_sha: str = Field(default="HEAD", description="Target commit SHA or branch ref.")
    changed_files: List[ChangedFileSchema] = Field(default_factory=list, description="List of modified files.")
    commit_message: str = Field(default="Update application logic", description="Git commit message.")
    budget_ratio: float = Field(default=0.25, ge=0.01, le=1.0, description="Max test fraction to run in fast mode.")
    repo_path: Optional[str] = Field(default=None, description="Local filesystem path to code repository.")
    execute: bool = Field(default=False, description="Whether to execute selected tests live in subprocess.")


class RankedTestSchema(BaseModel):
    test_id: str
    raw_score: float
    calibrated_confidence: float
    epistemic_uncertainty: float
    is_selected: bool
    reasons: List[str] = Field(default_factory=list)


class SelectResponseSchema(BaseModel):
    commit_sha: str
    decision_mode: str  # "FAST_SELECTED" or "SAFE_FULL_SUITE"
    abstained: bool
    selected_count: int
    total_count: int
    test_reduction_pct: float
    top_confidence: float
    epistemic_uncertainty: float
    reasons: List[str]
    selected_test_ids: List[str]
    ranked_tests: List[RankedTestSchema] = Field(default_factory=list)
    markdown_summary: Optional[str] = None
    execution_outcome: Optional[Dict[str, Any]] = None


# ==============================================================================
# Explainability Schemas
# ==============================================================================

class ExplainRequestSchema(BaseModel):
    test_id: str
    features: Dict[str, float] = Field(..., description="32-dimensional feature dictionary.")
    top_k: int = Field(default=5, ge=1, le=32)


class FeatureDriverSchema(BaseModel):
    feature: str
    feature_value: float
    shap_attribution: float
    impact: str


class ExplainResponseSchema(BaseModel):
    test_id: str
    predicted_probability: float
    base_expected_value: float
    risk_level: str
    primary_reasons: List[str]
    top_risk_increasing_features: List[FeatureDriverSchema]
    top_risk_decreasing_features: List[FeatureDriverSchema]


# ==============================================================================
# Calibration Schemas
# ==============================================================================

class CalibrationDifferenceItem(BaseModel):
    """A paired difference against the uncalibrated model, with its interval."""
    point: float
    ci_lower: float
    ci_upper: float
    excludes_zero: bool


class CalibrationMetricItem(BaseModel):
    method: Optional[str] = None
    ece: float
    mce: float
    brier_score: float
    ece_reduction_pct: Optional[float] = None
    # A client shown three bare ECE values will assume the smallest one won
    # something. On this project's own splits the best ECE belonged to a method
    # whose paired difference spanned zero, so the interval is what decides.
    ece_vs_uncalibrated: Optional[CalibrationDifferenceItem] = None
    mce_vs_uncalibrated: Optional[CalibrationDifferenceItem] = None
    brier_score_vs_uncalibrated: Optional[CalibrationDifferenceItem] = None


class CalibrationResponseSchema(BaseModel):
    best_method: str
    uncalibrated: CalibrationMetricItem
    # None when no method beat noise. Declining to calibrate is an outcome the
    # response has to be able to state; duplicating the uncalibrated numbers into
    # a 'calibrated' field would report the opposite of what was decided.
    calibrated: Optional[CalibrationMetricItem] = None
    temperature: Optional[float] = None
    # Which split `uncalibrated` and `calibrated` were measured on. Without this
    # the payload is ambiguous in a way that reads as a contradiction: the metric
    # blocks come from the held-out test split (ECE 0.0415 -> 0.0161) while
    # `selection_reason` quotes the validation split the method was *chosen* on
    # (0.0390 -> 0.0242). Two different pairs of numbers, both correct, and
    # nothing previously said which was which.
    metrics_split: Optional[str] = None
    selection_basis: Optional[str] = None
    selection_reason: Optional[str] = None
    selection_split: Optional[str] = None
    resampling_unit: Optional[str] = None
    reliability_diagram_bins: List[Dict[str, Any]] = Field(default_factory=list)


# ==============================================================================
# Repository & Analytics Schemas
# ==============================================================================

class RepositoryCreateSchema(BaseModel):
    full_name: str = Field(..., description="e.g. 'owner/repo-name'")
    url: str = Field(..., description="Git remote URL")
    local_path: Optional[str] = Field(default=None)


class RepositoryResponseSchema(BaseModel):
    id: int
    full_name: str
    url: str
    local_path: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnalyticsSummarySchema(BaseModel):
    total_repositories: int
    total_commits_evaluated: int
    total_decisions: int
    total_selective_fast_mode: int
    total_safe_abstentions: int
    # None when no decision has been recorded: an average over an empty set is
    # not zero, and must not be returned in the shape of a measurement.
    average_test_reduction_pct: Optional[float] = None
    total_failures_detected: int
    # None when no outcome has a verified full-suite comparison. Escapes are
    # unobservable on a selective run, so a 0 here would claim a safety result
    # that no measurement supports. verified/unverified counts say how much of
    # the history the figure actually covers.
    total_missed_failures: Optional[int] = None
    verified_outcomes: int = 0
    unverified_outcomes: int = 0
    average_uncertainty: Optional[float] = None
    recent_decisions: List[Dict[str, Any]] = Field(default_factory=list)
