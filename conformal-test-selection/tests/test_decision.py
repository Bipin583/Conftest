"""The SELECT / ABSTAIN / full-suite decision rule, in isolation.

These tests need no fitted artefacts: :func:`models.decision.build_decision` is
pure logic over a handful of flags, so it can be pinned exhaustively here while
``test_pipeline.py`` checks only that the pipeline wires the real numbers into
it. They enumerate every reason a selection is or is not trusted, and the
overriding safety property -- abstaining always falls back to the full suite,
which can only add tests.
"""

from __future__ import annotations

from models.decision import (
    DEGRADED_FEATURE_COMPLETENESS,
    EMPTY_CANDIDATE_SET,
    GUARANTEE_VOIDED_BY_BUDGET,
    REASON_EXPLANATIONS,
    RUN_FULL_SUITE,
    RUN_SELECTED,
    SELECTION_EXCEEDS_FALLBACK_FRACTION,
    WITHIN_GUARANTEE,
    build_decision,
)

_CFG = {
    "abstain_on_degraded": True,
    "abstain_on_voided_guarantee": True,
    "full_suite_fallback_rate": 0.90,
}


def _decide(**overrides):
    """A trustworthy baseline selection, with named fields overridden per test."""
    kwargs = dict(
        n_candidates=100,
        n_selected=40,
        selection_rate=0.40,
        degraded=False,
        budget_capped=False,
        guarantee_holds=True,
        selection_cfg=_CFG,
    )
    kwargs.update(overrides)
    return build_decision(**kwargs)


def test_a_healthy_selection_is_trusted():
    """No abstention condition means run the conformal subset, not the full suite."""
    decision = _decide()
    assert decision["action"] == RUN_SELECTED
    assert decision["reason_codes"] == [WITHIN_GUARANTEE]
    assert decision["safe_fallback"] is False
    assert decision["tests_to_run"] == 40


def test_degraded_features_force_the_full_suite():
    """An imputed feature vector describes the average test, so we cannot trust it."""
    decision = _decide(degraded=True)
    assert decision["action"] == RUN_FULL_SUITE
    assert DEGRADED_FEATURE_COMPLETENESS in decision["reason_codes"]
    assert decision["safe_fallback"] is True
    assert decision["tests_to_run"] == 100  # abstain runs every candidate


def test_a_budget_that_voids_the_guarantee_forces_the_full_suite():
    """A cap dropped a selected test, so the certified subset no longer exists."""
    decision = _decide(budget_capped=True, guarantee_holds=False)
    assert decision["action"] == RUN_FULL_SUITE
    assert GUARANTEE_VOIDED_BY_BUDGET in decision["reason_codes"]


def test_a_binding_cap_that_keeps_the_guarantee_does_not_abstain():
    """budget_capped alone is not enough; the guarantee must actually be void."""
    decision = _decide(budget_capped=True, guarantee_holds=True)
    assert decision["action"] == RUN_SELECTED
    assert GUARANTEE_VOIDED_BY_BUDGET not in decision["reason_codes"]


def test_a_selection_at_the_fallback_rate_prefers_the_full_suite():
    """If we are keeping ~everything, running everything is simpler and safer."""
    decision = _decide(n_selected=90, selection_rate=0.90)
    assert decision["action"] == RUN_FULL_SUITE
    assert SELECTION_EXCEEDS_FALLBACK_FRACTION in decision["reason_codes"]


def test_a_selection_just_below_the_fallback_rate_is_trusted():
    """The boundary is inclusive at the top only; just under it stays selected."""
    decision = _decide(n_selected=89, selection_rate=0.89)
    assert decision["action"] == RUN_SELECTED


def test_an_empty_candidate_set_abstains():
    """Nothing to select from is not an empty successful selection."""
    decision = _decide(n_candidates=0, n_selected=0, selection_rate=0.0)
    assert decision["action"] == RUN_FULL_SUITE
    assert EMPTY_CANDIDATE_SET in decision["reason_codes"]


def test_the_fallback_rate_does_not_trigger_on_an_empty_set():
    """selection_rate is 0.0 when there are no candidates, not a division error."""
    decision = _decide(n_candidates=0, n_selected=0, selection_rate=0.0)
    assert SELECTION_EXCEEDS_FALLBACK_FRACTION not in decision["reason_codes"]


def test_multiple_conditions_are_all_reported():
    """Reason codes accumulate; a dashboard sees every reason, not just the first."""
    decision = _decide(degraded=True, budget_capped=True, guarantee_holds=False)
    assert DEGRADED_FEATURE_COMPLETENESS in decision["reason_codes"]
    assert GUARANTEE_VOIDED_BY_BUDGET in decision["reason_codes"]
    assert WITHIN_GUARANTEE not in decision["reason_codes"]


def test_policy_toggles_can_disable_an_abstention():
    """A team may opt out of degraded-abstention; the code must honour that."""
    cfg = {**_CFG, "abstain_on_degraded": False}
    decision = _decide(degraded=True, selection_cfg=cfg)
    assert decision["action"] == RUN_SELECTED
    assert DEGRADED_FEATURE_COMPLETENESS not in decision["reason_codes"]


def test_every_reason_code_has_an_explanation():
    """A code with no human explanation would break the joined ``explanation``."""
    decision = _decide(degraded=True, budget_capped=True, guarantee_holds=False)
    for code in decision["reason_codes"]:
        assert code in REASON_EXPLANATIONS
    assert decision["explanation"]  # non-empty
