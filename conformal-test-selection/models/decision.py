"""SELECT / ABSTAIN / full-suite decision on top of a conformal selection.

The conformal layer always returns *a* subset. That is not the same as it always
being *safe to trust* that subset. When the request was scored on mostly-imputed
features, when a hard budget dropped a test the rule wanted (voiding the
guarantee), or when the "selection" is so large that running everything is
simpler and barely more expensive, the honest action is to **abstain and run the
full suite** rather than ship a subset whose guarantee no longer holds.

This module turns those conditions into an explicit, reason-coded decision. It is
deliberately conservative: abstaining can only add tests, so it never lowers
coverage -- it trades some CI cost for safety exactly when the guarantee is in
doubt. Running the full suite covers 100% of failing tests by construction, so an
abstention is always a valid fallback.

The reason codes are a small closed set (:data:`REASON_EXPLANATIONS`), not free
text, so a CI dashboard can branch on them and a reader can audit every reason a
selection was or was not trusted.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Decision actions. ``run_selected`` trusts the conformal subset; ``run_full_suite``
#: abstains and recommends running every candidate test.
RUN_SELECTED = "run_selected"
RUN_FULL_SUITE = "run_full_suite"

#: Reason codes. Closed set: a CI consumer may switch on these.
WITHIN_GUARANTEE = "WITHIN_GUARANTEE"
DEGRADED_FEATURE_COMPLETENESS = "DEGRADED_FEATURE_COMPLETENESS"
GUARANTEE_VOIDED_BY_BUDGET = "GUARANTEE_VOIDED_BY_BUDGET"
SELECTION_EXCEEDS_FALLBACK_FRACTION = "SELECTION_EXCEEDS_FALLBACK_FRACTION"
EMPTY_CANDIDATE_SET = "EMPTY_CANDIDATE_SET"

REASON_EXPLANATIONS: Dict[str, str] = {
    WITHIN_GUARANTEE: "The conformal rule's coverage guarantee holds for the returned subset.",
    DEGRADED_FEATURE_COMPLETENESS: (
        "Too few features were supplied; the score rests on imputed values, so the "
        "subset is not trustworthy and the full suite is run instead."
    ),
    GUARANTEE_VOIDED_BY_BUDGET: (
        "A max_tests budget dropped a test the rule selected, voiding the coverage "
        "guarantee; the full suite is run to stay safe."
    ),
    SELECTION_EXCEEDS_FALLBACK_FRACTION: (
        "The selected fraction is at or above the fallback rate, so the saving is "
        "negligible and running the full suite is simpler and strictly safer."
    ),
    EMPTY_CANDIDATE_SET: "No candidate tests were supplied to select from.",
}


def build_decision(
    *,
    n_candidates: int,
    n_selected: int,
    selection_rate: float,
    degraded: bool,
    budget_capped: bool,
    guarantee_holds: bool,
    selection_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether to trust the conformal subset or abstain to the full suite.

    Args:
        n_candidates: Number of candidate tests scored.
        n_selected: Number the conformal rule (plus budget) selected.
        selection_rate: ``n_selected / n_candidates``.
        degraded: Whether the request was flagged degraded (sparse features).
        budget_capped: Whether ``max_tests`` dropped a selected test.
        guarantee_holds: Whether the coverage guarantee still applies.
        selection_cfg: The ``selection`` config block, read for the policy toggles
            ``abstain_on_degraded``, ``abstain_on_voided_guarantee`` and
            ``full_suite_fallback_rate``.

    Returns:
        A decision dictionary with ``action`` (:data:`RUN_SELECTED` or
        :data:`RUN_FULL_SUITE`), ``reason_codes`` (a closed-set list),
        ``tests_to_run``, ``safe_fallback`` and a human-readable ``explanation``.
    """
    abstain_on_degraded = bool(selection_cfg.get("abstain_on_degraded", True))
    abstain_on_voided = bool(selection_cfg.get("abstain_on_voided_guarantee", True))
    fallback_rate = float(selection_cfg.get("full_suite_fallback_rate", 0.90) or 0.90)

    reasons: List[str] = []
    if n_candidates <= 0:
        reasons.append(EMPTY_CANDIDATE_SET)
    if degraded and abstain_on_degraded:
        reasons.append(DEGRADED_FEATURE_COMPLETENESS)
    if budget_capped and not guarantee_holds and abstain_on_voided:
        reasons.append(GUARANTEE_VOIDED_BY_BUDGET)
    if n_candidates > 0 and selection_rate >= fallback_rate:
        reasons.append(SELECTION_EXCEEDS_FALLBACK_FRACTION)

    abstain = bool(reasons)
    if not abstain:
        reasons = [WITHIN_GUARANTEE]

    action = RUN_FULL_SUITE if abstain else RUN_SELECTED
    return {
        "action": action,
        "reason_codes": reasons,
        "safe_fallback": abstain,
        "tests_to_run": int(n_candidates if abstain else n_selected),
        "explanation": " ".join(REASON_EXPLANATIONS[code] for code in reasons),
    }
