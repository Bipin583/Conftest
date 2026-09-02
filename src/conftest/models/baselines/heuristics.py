"""
ConfTest Rule-Based & Heuristic RTS Baselines.

Implements:
1. FullSuiteSelector (Oracle safety reference)
2. RandomKSelector (Budget-matched uniform random sampling)
3. ChangedFileSelector (Static file path string matching)
4. DependencyGraphSelector (Static call-graph reachability)
5. HistoricalFailureSelector (Historical failure frequency ranking)
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from conftest.models.baselines.base import BaseSelector, SelectionResult


class FullSuiteSelector(BaseSelector):
    """Baseline 1: Executes all candidate tests (0% savings, 100% failure recall)."""

    def __init__(self):
        super().__init__(name="1. Full Test Suite")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 1.0,
        **kwargs: Any,
    ) -> SelectionResult:
        all_ids = [t["test_id"] for t in candidate_tests]
        return SelectionResult(
            strategy_name=self.name,
            selected_tests=all_ids,
            total_tests=len(candidate_tests),
            abstained=False,
            mode="SAFE_FULL_SUITE",
            reasons=["Full suite execution requested."],
        )


class RandomKSelector(BaseSelector):
    """Baseline 2: Randomly selects tests up to budget limit."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name="2. Random-k Selection")
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def reset(self) -> None:
        """
        Rewind the stream so every sweep draws the same sample.

        This is the one baseline whose selections depend on call history: without
        the rewind, evaluating the same dataset twice in one process reports two
        different Random-k rows, and the point estimate stops matching the
        bootstrap that was supposed to describe it.
        """
        self.rng = random.Random(self.random_seed)

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0
        all_ids = [t["test_id"] for t in candidate_tests]

        selected = self.rng.sample(all_ids, k=min(k, total)) if all_ids else []
        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Uniform random sampling of {len(selected)} tests."],
        )


class ChangedFileSelector(BaseSelector):
    """
    Baseline 3: file-level RTS by name correspondence (Ekstazi-style).

    Selects every test whose module name corresponds to a changed source file.
    This is a *natural operating point* technique, not a budgeted one: it
    selects what the change touches and nothing more. Budget capping is
    therefore opt-in, because forcing this baseline to a fixed budget makes it
    look artificially weak when a change is genuinely broad.
    """

    def __init__(self, respect_budget: bool = False):
        super().__init__(name="3. Changed-File Selection")
        self.respect_budget = respect_budget

    @staticmethod
    def _source_stem(file_path: str) -> str:
        """Normalise a changed source file to its bare module name."""
        return Path(file_path).stem.lower()

    @staticmethod
    def _test_stem(test_path: str) -> str:
        """
        Normalise a test module to the source module it is presumed to cover.

        `tests/test_payment.py` -> `payment`. Handles IDs that carry no `.py`
        suffix, which is how pytest reports tests collected via `classname`.
        """
        name = Path(test_path).name.lower()
        for suffix in (".py",):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = name.split("::")[0]
        if name.startswith("test_"):
            name = name[len("test_") :]
        if name.endswith("_test"):
            name = name[: -len("_test")]
        return name

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)

        # A blank stem would substring-match every test, silently selecting the
        # whole suite and disguising missing changed-file data as a 0% result.
        changed_stems = {
            stem
            for stem in (
                self._source_stem(f.get("file_path", "")) for f in changed_files
            )
            if stem
        }

        matched: List[str] = []
        for t in candidate_tests:
            t_path = str(t.get("test_path") or t.get("test_id") or "").lower()
            t_stem = self._test_stem(t_path)
            if t_stem and t_stem in changed_stems:
                matched.append(t["test_id"])
            elif any(stem in t_path for stem in changed_stems):
                matched.append(t["test_id"])

        selected = matched
        reasons = [f"{len(matched)} of {total} tests correspond to {len(changed_stems)} changed file(s)."]

        if self.respect_budget and total > 0:
            k = max(1, int(total * budget_ratio))
            if len(matched) > k:
                selected = matched[:k]
                reasons.append(f"Truncated to budget k={k}.")

        if not changed_stems:
            reasons.append("WARNING: no changed files supplied; selection is vacuous.")
        elif not matched:
            reasons.append("No name correspondence found; file-level RTS selects nothing.")

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=reasons,
        )


class DependencyGraphSelector(BaseSelector):
    """Baseline 4: Selects tests using static call-graph reachability."""

    def __init__(self):
        super().__init__(name="4. Static AST Call-Graph Selection")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Rank by shortest dependency depth (lowest depth first)
        def get_depth(t: Dict[str, Any]) -> float:
            return float(t.get("features", {}).get("dep_shortest_path_depth", 10.0))

        sorted_tests = sorted(candidate_tests, key=get_depth)
        # Select reachable tests (depth < 10.0) up to budget k
        reachable = [t["test_id"] for t in sorted_tests if get_depth(t) < 10.0]
        selected = reachable[:k] if reachable else [t["test_id"] for t in sorted_tests[:k]]

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Selected {len(selected)} tests within call-graph dependency path."],
        )


class HistoricalFailureSelector(BaseSelector):
    """Baseline 5: Selects tests ranked by historical failure frequency."""

    def __init__(self):
        super().__init__(name="5. Historical Failure Frequency")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Rank by historical failure rate or failure count (descending)
        def get_hist_score(t: Dict[str, Any]) -> float:
            feats = t.get("features", {})
            return float(feats.get("hist_lifetime_failure_rate", 0.0) + feats.get("hist_prior_failures", 0.0) * 0.1)

        sorted_tests = sorted(candidate_tests, key=get_hist_score, reverse=True)
        selected = [t["test_id"] for t in sorted_tests[:k]]

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Selected {len(selected)} tests with highest historical failure counts."],
        )
