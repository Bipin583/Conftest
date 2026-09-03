"""
ConfTest Selective Prediction Policy & Safe Fallback Engine.

Implements confidence-calibrated selective regression test selection:
- If model is confident and epistemic uncertainty is low: Fast Selective Mode (subset execution).
- If model is uncertain or diff is out-of-distribution: Safe Full-Suite Fallback (100% execution).

Provides threshold grid search optimization on validation splits and CI utility modeling.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyDecision:
    """Standard outcome container for a selective prediction decision."""

    commit_sha: str
    decision_mode: str  # "FAST_SELECTED" or "SAFE_FULL_SUITE"
    abstained: bool
    selected_test_ids: List[str]
    total_test_count: int
    top_confidence: float
    epistemic_uncertainty: float
    reasons: List[str]
    estimated_time_saved_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "decision_mode": self.decision_mode,
            "abstained": self.abstained,
            "selected_count": len(self.selected_test_ids),
            "total_count": self.total_test_count,
            "test_reduction_ratio": round(1.0 - (len(self.selected_test_ids) / max(1, self.total_test_count)), 4),
            "top_confidence": round(self.top_confidence, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "reasons": self.reasons,
            "estimated_time_saved_pct": round(self.estimated_time_saved_pct, 2),
            "selected_test_ids": self.selected_test_ids,
        }


class SelectivePredictionPolicy:
    """Configurable Selective Prediction Decision Policy for RTS."""

    def __init__(
        self,
        tau_abstain: float = 0.015,
        tau_conf: float = 0.60,
        budget_ratio: float = 0.25,
        ood_file_limit: int = 15,
        ood_churn_limit: int = 500,
    ):
        """
        Initialize policy parameters.

        Args:
            tau_abstain: Maximum allowed epistemic uncertainty std before abstaining.
            tau_conf: Minimum required top-1 test failure confidence to execute fast mode.
            budget_ratio: Maximum fraction of tests to run in fast mode (e.g. 0.25 = top 25%).
            ood_file_limit: Max changed files threshold before flagging as OOD refactoring.
            ood_churn_limit: Max lines churn threshold before flagging as OOD refactoring.
        """
        self.tau_abstain = tau_abstain
        self.tau_conf = tau_conf
        self.budget_ratio = budget_ratio
        self.ood_file_limit = ood_file_limit
        self.ood_churn_limit = ood_churn_limit

    def evaluate_commit(
        self,
        commit_sha: str,
        candidate_test_ids: List[str],
        calibrated_confidences: np.ndarray,
        epistemic_uncertainties: np.ndarray,
        num_changed_files: int = 1,
        total_churn_lines: int = 10,
    ) -> PolicyDecision:
        """
        Make a selective prediction decision for a commit.

        Args:
            commit_sha: SHA of the target commit.
            candidate_test_ids: List of available test node IDs.
            calibrated_confidences: Array of calibrated probabilities p_hat for each candidate test.
            epistemic_uncertainties: Array of epistemic uncertainty stds sigma for each candidate test.
            num_changed_files: Number of modified files in the commit.
            total_churn_lines: Total lines added + deleted in the commit diff.

        Returns:
            PolicyDecision object detailing mode, selected tests, confidence, and reasons.
        """
        total_tests = len(candidate_test_ids)
        if total_tests == 0:
            return PolicyDecision(
                commit_sha=commit_sha,
                decision_mode="SAFE_FULL_SUITE",
                abstained=False,
                selected_test_ids=[],
                total_test_count=0,
                top_confidence=0.0,
                epistemic_uncertainty=0.0,
                reasons=["No candidate tests found in repository."],
                estimated_time_saved_pct=0.0,
            )

        top_conf = float(np.max(calibrated_confidences)) if len(calibrated_confidences) > 0 else 0.0
        max_uncertainty = float(np.max(epistemic_uncertainties)) if len(epistemic_uncertainties) > 0 else 0.0

        # Check Out-of-Distribution (OOD) Refactoring Churn
        is_ood = (num_changed_files > self.ood_file_limit) or (total_churn_lines > self.ood_churn_limit)

        # Evaluate Fallback Criteria
        reasons: List[str] = []
        should_abstain = False

        if max_uncertainty > self.tau_abstain:
            should_abstain = True
            reasons.append(f"High epistemic uncertainty ({max_uncertainty:.4f} > tau_abstain {self.tau_abstain:.4f}).")

        if top_conf < self.tau_conf:
            should_abstain = True
            reasons.append(f"Low risk confidence ({top_conf:.4f} < tau_conf {self.tau_conf:.4f}).")

        if is_ood:
            should_abstain = True
            reasons.append(f"OOD architectural refactoring detected ({num_changed_files} files, {total_churn_lines} lines churn).")

        # Branch 1: Safe Full-Suite Fallback
        if should_abstain:
            reasons.append("Abstaining from subset selection. Executing 100% full test suite for regression safety.")
            return PolicyDecision(
                commit_sha=commit_sha,
                decision_mode="SAFE_FULL_SUITE",
                abstained=True,
                selected_test_ids=list(candidate_test_ids),
                total_test_count=total_tests,
                top_confidence=top_conf,
                epistemic_uncertainty=max_uncertainty,
                reasons=reasons,
                estimated_time_saved_pct=0.0,
            )

        # Branch 2: Fast Selective Execution
        k = max(1, int(total_tests * self.budget_ratio))
        # Rank candidate tests by calibrated failure risk (descending)
        ranked_indices = np.argsort(calibrated_confidences)[::-1]
        selected_indices = ranked_indices[:k]
        selected_tests = [candidate_test_ids[i] for i in selected_indices]

        # 1 - selected/total: a fraction of the test COUNT, not of the wall-clock.
        # The two coincide only if every test costs the same, and durations are
        # heavily skewed in practice. The field keeps its name for the DB column
        # it feeds, but nothing downstream may call it a time saving: the reason
        # string said "time saved" and the dashboard repeated it.
        time_saved_pct = (1.0 - (len(selected_tests) / total_tests)) * 100.0
        reasons.append(f"High confidence ({top_conf*100:.1f}%) and low epistemic uncertainty ({max_uncertainty:.4f}).")
        reasons.append(f"Fast selective mode: Executing {len(selected_tests)}/{total_tests} risk-ranked tests ({time_saved_pct:.1f}% of the suite skipped by count).")

        return PolicyDecision(
            commit_sha=commit_sha,
            decision_mode="FAST_SELECTED",
            abstained=False,
            selected_test_ids=selected_tests,
            total_test_count=total_tests,
            top_confidence=top_conf,
            epistemic_uncertainty=max_uncertainty,
            reasons=reasons,
            estimated_time_saved_pct=time_saved_pct,
        )

    def save(self, filepath: str) -> str:
        """Save policy thresholds to JSON."""
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "tau_abstain": self.tau_abstain,
            "tau_conf": self.tau_conf,
            "budget_ratio": self.budget_ratio,
            "ood_file_limit": self.ood_file_limit,
            "ood_churn_limit": self.ood_churn_limit,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Policy configuration saved to {out_path}")
        return str(out_path)

    @classmethod
    def load(cls, filepath: str) -> "SelectivePredictionPolicy":
        """Load policy configuration from JSON."""
        in_path = Path(filepath).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Policy config not found: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(**cfg)


class CostBenefitModel:
    """Models CI execution cost savings versus escaped bug penalties."""

    def __init__(self, cost_per_test_second: float = 0.01, penalty_per_escaped_failure: float = 50.0):
        self.cost_per_sec = cost_per_test_second
        self.penalty_per_escape = penalty_per_escaped_failure

    def compute_utility(
        self,
        full_suite_duration_sec: float,
        selected_duration_sec: float,
        escaped_failures_count: int,
    ) -> Dict[str, float]:
        """Compute net CI financial utility."""
        time_saved = max(0.0, full_suite_duration_sec - selected_duration_sec)
        gross_savings = time_saved * self.cost_per_sec
        penalty_cost = escaped_failures_count * self.penalty_per_escape
        net_utility = gross_savings - penalty_cost

        return {
            "time_saved_sec": round(time_saved, 2),
            "gross_savings_dollars": round(gross_savings, 2),
            "penalty_cost_dollars": round(penalty_cost, 2),
            "net_ci_utility_dollars": round(net_utility, 2),
        }
