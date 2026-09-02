"""
Calibrator selection.

Split out of `scripts/calibrate_model.py` so the decision is testable on its own,
because the original one-liner was wrong in three separate ways.

It read:

    best_cal = iso_cal if iso_ece <= temp_ece else temp_cal

1. **It chose on ECE alone.** ECE is a bin-weighted *average* miscalibration, so a
   method can win it while being catastrophically wrong in one region. Measured on
   this project's own run: temperature scaling scored ECE 0.0192 -- the best of the
   three -- with MCE 0.8943, against the uncalibrated model's MCE 0.2222. A bin
   that is off by 0.89 is not a rounding error for us: the abstention policy reads
   confidence at a *specific* threshold, so a miscalibrated region is exactly the
   thing that breaks the safety argument the project is built on.

2. **It chose on the test split.** The calibrators were fitted on validation and
   then compared on test, which makes test part of model selection and biases every
   number reported from it. Selection now happens inside the validation split and
   test is touched once, for reporting.

3. **It could not decline.** Uncalibrated was never a candidate, so the script
   always installed a calibrator even when neither helped.

The replacement disqualifies any method that makes the worst-case bin or the proper
score worse than leaving the model alone, then picks the lowest ECE among what
survives. "Uncalibrated" is always a candidate, so declining to calibrate is a
possible -- and reportable -- outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CalibrationScore:
    """Metrics for one candidate, all measured on the same held-out data."""

    method: str
    ece: float
    mce: float
    brier: float


@dataclass(frozen=True)
class SelectionOutcome:
    method: str
    reason: str
    disqualified: Dict[str, str]

    @property
    def calibrated(self) -> bool:
        return self.method != "uncalibrated"


# A calibrator may not make the worst bin worse than the raw model by more than
# this. Zero would be too strict: bin boundaries move slightly between candidates,
# so an exactly-equal method could be rejected on noise. A tenth of a probability
# point is well inside that noise and far below the 0.67 regression that motivated
# the guard.
MCE_TOLERANCE = 0.10

# Same idea for the proper score. Brier is on the same scale as squared
# probability error, so this is a genuinely small allowance.
BRIER_TOLERANCE = 0.01


def select_calibrator(
    scores: Sequence[CalibrationScore],
    baseline_method: str = "uncalibrated",
) -> SelectionOutcome:
    """
    Choose a calibration method from validation-split scores.

    `scores` must include the baseline (the uncalibrated model), because the
    guards are expressed relative to it and because declining to calibrate has to
    be reachable.
    """
    by_method = {s.method: s for s in scores}
    if baseline_method not in by_method:
        raise ValueError(
            f"select_calibrator needs the {baseline_method!r} baseline among the "
            f"candidates; got {sorted(by_method)}. Without it there is nothing to "
            f"guard against and no way to decline calibration."
        )

    baseline = by_method[baseline_method]
    disqualified: Dict[str, str] = {}
    eligible: List[CalibrationScore] = []

    for score in scores:
        if score.method == baseline_method:
            continue
        if score.mce > baseline.mce + MCE_TOLERANCE:
            disqualified[score.method] = (
                f"worst-case miscalibration got worse: MCE {score.mce:.4f} vs "
                f"{baseline.mce:.4f} uncalibrated (tolerance {MCE_TOLERANCE})"
            )
            continue
        if score.brier > baseline.brier + BRIER_TOLERANCE:
            disqualified[score.method] = (
                f"proper score got worse: Brier {score.brier:.4f} vs "
                f"{baseline.brier:.4f} uncalibrated (tolerance {BRIER_TOLERANCE})"
            )
            continue
        if score.ece >= baseline.ece:
            disqualified[score.method] = (
                f"no calibration gain: ECE {score.ece:.4f} vs "
                f"{baseline.ece:.4f} uncalibrated"
            )
            continue
        eligible.append(score)

    if not eligible:
        return SelectionOutcome(
            method=baseline_method,
            reason=(
                "no candidate improved ECE without degrading worst-case "
                "calibration or the proper score; reporting the model uncalibrated"
            ),
            disqualified=disqualified,
        )

    # Ties broken by MCE then Brier, so the safer of two equally-calibrated
    # methods wins. Method name last, to keep the choice deterministic.
    winner = min(eligible, key=lambda s: (s.ece, s.mce, s.brier, s.method))
    return SelectionOutcome(
        method=winner.method,
        reason=(
            f"lowest validation ECE {winner.ece:.4f} (vs {baseline.ece:.4f} "
            f"uncalibrated) with MCE {winner.mce:.4f} and Brier {winner.brier:.4f} "
            f"both within tolerance of the uncalibrated model"
        ),
        disqualified=disqualified,
    )


def split_for_selection(n: int, holdout_fraction: float = 0.5) -> int:
    """
    Index at which to cut the validation split into fit and select halves.

    Returns the size of the fit half. Selection needs data the calibrator did not
    see, or a flexible method like isotonic regression wins by memorising.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")
    fit_size = int(round(n * (1.0 - holdout_fraction)))
    # Both halves must be non-empty for the comparison to mean anything.
    fit_size = max(1, min(n - 1, fit_size))
    return fit_size
