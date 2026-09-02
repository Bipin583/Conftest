"""
ConfTest Abstract Baseline Selector Interface.

Defines the contract for Regression Test Selection (RTS) baseline strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SelectionResult:
    """Standard container for test selection decisions."""

    strategy_name: str
    selected_tests: List[str]
    total_tests: int
    abstained: bool = False
    mode: str = "FAST_SELECTED"  # FAST_SELECTED or SAFE_FULL_SUITE
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return len(self.selected_tests)

    @property
    def test_reduction_ratio(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return max(0.0, 1.0 - (len(self.selected_tests) / self.total_tests))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "selected_count": self.selected_count,
            "total_tests": self.total_tests,
            "test_reduction_ratio": round(self.test_reduction_ratio, 4),
            "abstained": self.abstained,
            "mode": self.mode,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "selected_tests": self.selected_tests,
            "reasons": self.reasons,
        }


class BaseSelector(ABC):
    """Abstract base class for all RTS baseline algorithms."""

    def __init__(self, name: str):
        self.name = name

    def reset(self) -> None:
        """
        Return the selector to the state it was constructed in.

        Called once per evaluation sweep. A selector that carries random state
        would otherwise give different answers on the second sweep over the same
        dataset, which makes the benchmark depend on how many times it has been
        run in this process. Stateless selectors need not override this.
        """
        return None

    @abstractmethod
    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        """
        Select a subset of candidate tests for execution on a commit.

        Args:
            candidate_tests: List of candidate test dictionaries containing 'test_id', 'test_path', etc.
            changed_files: List of changed file dictionaries in the commit diff.
            budget_ratio: Maximum budget fraction of tests to select (e.g. 0.25 = top 25%).

        Returns:
            SelectionResult object.
        """
        pass
