"""
ConfTest Selective Prediction Decision Engine
Member 3 & Member 1 Technical Domain: Selective Prediction, Abstention Policy, Safe Fallbacks.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **conftest.engine.selector_engine** (ConfTestEngine, which records whether an outcome was verified).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "conftest.engine.selector_engine"

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class SelectionDecision:
    action: str                     # 'SELECTIVE_RUN', 'ABSTAIN_SAFE_FALLBACK', 'NO_OP'
    selected_tests: List[str]
    total_tests_count: int
    selected_count: int
    test_reduction_ratio: float
    max_uncertainty: float
    calibrated_confidence: float
    reason: str


class SelectiveDecisionEngine:
    """
    Executes selective prediction with risk-calibrated abstention for CI/CD test execution.
    """
    def __init__(self, tau_abstain: float = 0.18, theta_select: float = 0.08):
        self.tau_abstain = tau_abstain       # Uncertainty threshold triggering abstention
        self.theta_select = theta_select     # Failure probability threshold for test selection

    def decide(
        self,
        test_ids: List[str],
        calibrated_probs: np.ndarray,
        uncertainties: np.ndarray,
        direct_dep_tests: Optional[List[str]] = None,
        is_ood_refactoring: bool = False
    ) -> SelectionDecision:
        """
        Determines whether to execute a selective subset or ABSTAIN to full test suite.
        """
        total_count = len(test_ids)
        if total_count == 0:
            return SelectionDecision(
                action="NO_OP",
                selected_tests=[],
                total_tests_count=0,
                selected_count=0,
                test_reduction_ratio=0.0,
                max_uncertainty=0.0,
                calibrated_confidence=1.0,
                reason="Empty test suite"
            )

        max_uncertainty = float(np.max(uncertainties)) if len(uncertainties) > 0 else 0.0
        confidence = float(1.0 - max_uncertainty)

        # Rule 1: Out-of-Distribution or Massive Refactoring check
        if is_ood_refactoring:
            return SelectionDecision(
                action="ABSTAIN_SAFE_FALLBACK",
                selected_tests=test_ids,
                total_tests_count=total_count,
                selected_count=total_count,
                test_reduction_ratio=0.0,
                max_uncertainty=max_uncertainty,
                calibrated_confidence=confidence,
                reason="Out-of-distribution code change detected. Safe fallback executed."
            )

        # Rule 2: Epistemic Uncertainty Abstention Check
        if max_uncertainty > self.tau_abstain:
            return SelectionDecision(
                action="ABSTAIN_SAFE_FALLBACK",
                selected_tests=test_ids,
                total_tests_count=total_count,
                selected_count=total_count,
                test_reduction_ratio=0.0,
                max_uncertainty=max_uncertainty,
                calibrated_confidence=confidence,
                reason=f"Max prediction uncertainty ({max_uncertainty:.3f}) exceeded safety budget ({self.tau_abstain}). Full suite executed."
            )

        # Rule 3: Selective Run Selection
        selected = [test_ids[i] for i, p in enumerate(calibrated_probs) if p >= self.theta_select]

        # Ensure safety floor: always include direct import dependents
        if direct_dep_tests:
            selected = list(set(selected).union(set(direct_dep_tests)))

        # If nothing selected, pick top-3 fastest sanity tests
        if len(selected) == 0:
            selected = test_ids[:min(3, total_count)]

        reduction = 1.0 - (len(selected) / total_count)

        return SelectionDecision(
            action="SELECTIVE_RUN",
            selected_tests=selected,
            total_tests_count=total_count,
            selected_count=len(selected),
            test_reduction_ratio=float(reduction),
            max_uncertainty=max_uncertainty,
            calibrated_confidence=confidence,
            reason=f"High-confidence prediction ({confidence:.1%}). Selected {len(selected)}/{total_count} tests ({reduction:.1%} of the suite skipped by count)."
        )
