"""
ConfTest Machine Learning Baseline & Proposed Selective Selectors.

Implements:
6. UncalibratedMLSelector (Raw model score ranking)
7. CalibratedNoAbstentionSelector (Calibrated probabilities without abstention)
8. ConfTestSelectiveSelector (Proposed calibrated selector with epistemic abstention fallback)
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from conftest.models.baselines.base import BaseSelector, SelectionResult

# The operating point this system ships. Baseline 8 is the proposed method, so it has
# to be evaluated at the thresholds scripts/tune_policy.py chose -- not at literals
# that happen to sit in a constructor signature. Until 2026-09-03 this class defaulted
# to 0.15 / 0.70 while the tuned policy was 0.02 / 0.10, so the published headline row
# and the policy report described two different systems.
POLICY_CONFIG_PATH = Path("models/policy_config.json")
POLICY_PRODUCED_BY = "python scripts/tune_policy.py"


class UncalibratedMLSelector(BaseSelector):
    """Baseline 6: Ranks tests by raw uncalibrated model scores."""

    def __init__(self, model_scorer: Optional[Callable[[Dict[str, Any]], float]] = None):
        super().__init__(name="6. Uncalibrated ML (LightGBM)")
        self.model_scorer = model_scorer

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Extract score from features / candidate dictionary
        def score_fn(t: Dict[str, Any]) -> float:
            if "raw_score" in t:
                return float(t["raw_score"])
            if self.model_scorer:
                return float(self.model_scorer(t))
            # Fallback heuristic score based on structural/diff features
            feats = t.get("features", {})
            return float(
                feats.get("dep_is_direct_import", 0.0) * 0.5
                + feats.get("hist_lifetime_failure_rate", 0.0) * 0.3
                + (1.0 / max(1.0, feats.get("dep_shortest_path_depth", 10.0))) * 0.2
            )

        sorted_tests = sorted(candidate_tests, key=score_fn, reverse=True)
        selected = [t["test_id"] for t in sorted_tests[:k]]
        avg_score = float(np.mean([score_fn(t) for t in sorted_tests[:k]])) if selected else 0.5

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            confidence=round(avg_score, 4),
            reasons=[f"Selected top {len(selected)} tests by raw ML failure score."],
        )


class CalibratedNoAbstentionSelector(BaseSelector):
    """Baseline 7: Ranks tests by calibrated probabilities without abstention."""

    def __init__(self):
        super().__init__(name="7. Calibrated ML (No Abstention)")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        def score_fn(t: Dict[str, Any]) -> float:
            if "calibrated_confidence" in t:
                return float(t["calibrated_confidence"])
            if "raw_score" in t:
                # Isotonic/Sigmoid calibration approximation
                return float(1.0 / (1.0 + np.exp(-4.0 * (t["raw_score"] - 0.5))))
            feats = t.get("features", {})
            raw = (
                feats.get("dep_is_direct_import", 0.0) * 0.5
                + feats.get("hist_lifetime_failure_rate", 0.0) * 0.3
            )
            return float(np.clip(raw, 0.01, 0.99))

        sorted_tests = sorted(candidate_tests, key=score_fn, reverse=True)
        selected = [t["test_id"] for t in sorted_tests[:k]]
        avg_conf = float(np.mean([score_fn(t) for t in sorted_tests[:k]])) if selected else 0.5

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            confidence=round(avg_conf, 4),
            reasons=[f"Selected top {len(selected)} tests by calibrated empirical confidence."],
        )


class ConfTestSelectiveSelector(BaseSelector):
    """
    Proposed Method (Baseline 8):
    Confidence-Calibrated Regression Test Selection with Epistemic Abstention Fallback.
    """

    def __init__(
        self,
        abstention_threshold: Optional[float] = None,
        min_confidence_threshold: Optional[float] = None,
        policy_config_path: Optional[Path] = None,
    ):
        """
        Args:
            abstention_threshold: Max epistemic uncertainty before falling back to the
                full suite. None reads tau_abstain from the tuned policy.
            min_confidence_threshold: Min top-1 calibrated confidence required to run in
                fast mode. None reads tau_conf from the tuned policy.
            policy_config_path: Override for the tuned policy location.

        Raises:
            FileNotFoundError: A threshold was left unspecified and there is no tuned
                policy to read it from. An untuned operating point is not a default.
        """
        super().__init__(name="8. ConfTest (Calibrated + Selective Abstention)")
        if abstention_threshold is None or min_confidence_threshold is None:
            path = policy_config_path or POLICY_CONFIG_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"No tuned policy at {path}, and this selector was constructed "
                    "without explicit thresholds. Run "
                    f"`{POLICY_PRODUCED_BY}` first, or pass both thresholds "
                    "explicitly to evaluate a deliberately different operating point."
                )
            tuned = json.loads(path.read_text(encoding="utf-8"))
            for key in ("tau_abstain", "tau_conf"):
                if key not in tuned:
                    raise KeyError(f"{path} has no '{key}'; re-run {POLICY_PRODUCED_BY}.")
            if abstention_threshold is None:
                abstention_threshold = float(tuned["tau_abstain"])
            if min_confidence_threshold is None:
                min_confidence_threshold = float(tuned["tau_conf"])
            self.thresholds_from = str(path)
        else:
            self.thresholds_from = "explicit constructor arguments"
        self.abstention_threshold = float(abstention_threshold)
        self.min_confidence_threshold = float(min_confidence_threshold)

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        all_ids = [t["test_id"] for t in candidate_tests]
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Extract or estimate epistemic uncertainty
        uncertainties = []
        confidences = []

        # No invented uncertainty and no invented confidence. The abstention rule is
        # the method's contribution; thresholding a filled-in 0.08 measures nothing.
        for t in candidate_tests:
            for key in ("uncertainty", "calibrated_confidence"):
                if key not in t:
                    raise KeyError(
                        f"Candidate {t.get('test_id', '<unknown>')} has no '{key}'. "
                        "Baseline 8 abstains on measured ensemble disagreement and "
                        "measured calibrated confidence; score the tests first."
                    )
            uncertainties.append(float(t["uncertainty"]))
            confidences.append(float(t["calibrated_confidence"]))

        max_uncertainty = float(np.max(uncertainties)) if uncertainties else 0.0
        top_confidence = float(np.max(confidences)) if confidences else 0.85
        avg_confidence = float(np.mean(confidences)) if confidences else 0.85

        # Check for out-of-distribution diff indicators (e.g. large cross-module refactoring)
        num_changed = len(changed_files)
        total_churn = sum(f.get("lines_added", 0) + f.get("lines_deleted", 0) for f in changed_files)
        is_ood = num_changed > 15 or total_churn > 500

        # Abstention Decision Rule: Abstain if high uncertainty, low top confidence, or OOD diff
        should_abstain = (
            max_uncertainty > self.abstention_threshold
            or top_confidence < self.min_confidence_threshold
            or is_ood
        )

        if should_abstain:
            reason = (
                f"Abstained due to high uncertainty ({max_uncertainty:.3f} > {self.abstention_threshold:.3f})"
                if max_uncertainty > self.abstention_threshold
                else f"Abstained due to low confidence ({top_confidence:.3f} < {self.min_confidence_threshold:.3f})"
                if top_confidence < self.min_confidence_threshold
                else "Abstained due to OOD architectural churn diff."
            )
            return SelectionResult(
                strategy_name=self.name,
                selected_tests=all_ids,  # Full Suite fallback
                total_tests=total,
                abstained=True,
                mode="SAFE_FULL_SUITE",
                confidence=round(top_confidence, 4),
                uncertainty=round(max_uncertainty, 4),
                reasons=[reason, "Executing full test suite for regression safety."],
            )

        # High Confidence: Rank and select top budget-matched tests
        def rank_key(idx: int) -> float:
            return confidences[idx]

        sorted_indices = sorted(range(total), key=rank_key, reverse=True)
        selected = [candidate_tests[i]["test_id"] for i in sorted_indices[:k]]

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            confidence=round(top_confidence, 4),
            uncertainty=round(max_uncertainty, 4),
            reasons=[
                f"High confidence ({top_confidence*100:.1f}%) with low uncertainty ({max_uncertainty:.3f}).",
                f"Selected {len(selected)}/{total} high-risk tests "
                f"({100*(1-len(selected)/total):.1f}% of the suite skipped by count).",
            ],
        )
