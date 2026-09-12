"""
Compatibility import path for the abstention machinery.

The documented module name for abstention is ``conftest.engine.abstention``.
The implementation is split across two modules, because abstention here is two
separable things:

* :class:`conftest.models.policy.SelectivePredictionPolicy` -- the decision
  rule. Given per-test calibrated confidences and the ensemble's epistemic
  standard deviation, it returns either ``ABSTAIN`` (run the full suite) or a
  top-k selection under ``budget_ratio``.
* :class:`conftest.engine.selector_engine.ConfTestEngine` -- the caller that
  mines features, scores the ensemble, calibrates, and then applies the policy.

This module contains no logic; it re-exports both so the documented path
resolves. Nothing is duplicated, so nothing can drift.

Thresholds
----------
There is no single default tau. ``SelectivePredictionPolicy.__init__`` carries
placeholders (``tau_abstain=0.015``, ``tau_conf=0.60``) that are explicitly *not*
an operating point; the shipped operating point is ``models/policy_config.json``
(``tau_abstain=0.02``, ``tau_conf=0.10``, ``budget_ratio=0.25``), produced by
``scripts/tune_policy.py`` under the ``zero_escape`` objective, and loaded via
:meth:`SelectivePredictionPolicy.load`. ``is_tuned`` distinguishes the two, and
:meth:`SelectivePredictionPolicy.untuned_always_abstain` is what you get when no
tuned artifact exists -- the full suite, not an arbitrary constant.

At the shipped point the policy abstains on 97.81% of the 183 held-out commits
(``reports/baseline_comparison.csv``), which is why failure recall is 100.0%
with 0 escapes and the reduction is only 3.11% of executions. Raising reduction
means lowering tau_abstain, and ``reports/g5_recall_floor.json`` records what
that costs: 32.85% reduction at 87.69% recall, with 15 escaped commits.
"""

from __future__ import annotations

from conftest.engine.selector_engine import ConfTestEngine
from conftest.models.policy import (
    UNTUNED_SOURCE,
    CostBenefitModel,
    PolicyDecision,
    SelectivePredictionPolicy,
)

# Alias under the name the docs use for the policy class.
AbstentionPolicy = SelectivePredictionPolicy

__all__ = [
    "AbstentionPolicy",
    "ConfTestEngine",
    "CostBenefitModel",
    "PolicyDecision",
    "SelectivePredictionPolicy",
    "UNTUNED_SOURCE",
]
