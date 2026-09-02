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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from conftest.evaluation.statistics import (
    cluster_draws,
    group_rows_by_cluster,
    intervals_from_replicates,
    rows_for_clusters,
)
from conftest.models.calibration import compute_ece


@dataclass(frozen=True)
class CalibrationScore:
    """
    Metrics for one candidate, all measured on the same held-out data.

    The three `*_vs_baseline` fields carry the paired difference against the
    uncalibrated model from a cluster bootstrap over mutants, when the caller had
    the cluster labels to produce one (see `score_calibrators`). Each is an
    interval dict with `point`, `ci_lower`, `ci_upper` and `excludes_zero`. They
    are None when no clustering was available, and the decision then falls back to
    fixed tolerances -- see `select_calibrator`.
    """

    method: str
    ece: float
    mce: float
    brier: float
    ece_vs_baseline: Optional[Dict[str, Any]] = None
    mce_vs_baseline: Optional[Dict[str, Any]] = None
    brier_vs_baseline: Optional[Dict[str, Any]] = None

    @property
    def has_intervals(self) -> bool:
        """Whether this candidate can be judged on measured noise rather than a guess."""
        return None not in (
            self.ece_vs_baseline, self.mce_vs_baseline, self.brier_vs_baseline
        )


@dataclass(frozen=True)
class SelectionOutcome:
    method: str
    reason: str
    disqualified: Dict[str, str]
    # "point" when the decision rested on fixed tolerances, "bootstrap" when every
    # comparison was a paired interval over resampled mutants. Worth reporting:
    # the two can disagree, and which one was used changes what the choice means.
    basis: str = "point"

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


def _worse_by_an_interval(difference: Dict[str, Any]) -> bool:
    """
    Whether a candidate-minus-baseline difference is worse than noise.

    Differences are taken as candidate - baseline, so a positive point estimate on
    a loss means the candidate lost ground. "Positive and excludes zero" is the
    whole test: a point estimate on the wrong side of zero whose interval spans it
    is not evidence of anything at this sample size.
    """
    return bool(difference.get("point", float("nan")) > 0.0 and difference.get("excludes_zero"))


def _disqualify_on_intervals(score: CalibrationScore) -> Optional[str]:
    """Reject a candidate on measured noise. None means it survives."""
    mce, brier, ece = score.mce_vs_baseline, score.brier_vs_baseline, score.ece_vs_baseline
    if _worse_by_an_interval(mce):
        return (
            f"worst-case miscalibration is measurably worse: MCE {score.mce:.4f}, "
            f"paired difference +{mce['point']:.4f} "
            f"[{mce['ci_lower']:.4f}, {mce['ci_upper']:.4f}] excludes zero"
        )
    if _worse_by_an_interval(brier):
        return (
            f"proper score is measurably worse: Brier {score.brier:.4f}, paired "
            f"difference +{brier['point']:.4f} "
            f"[{brier['ci_lower']:.4f}, {brier['ci_upper']:.4f}] excludes zero"
        )
    # A gain has to be a gain: negative difference, interval clear of zero. An ECE
    # that merely came out lower on this particular split is the noise this whole
    # path exists to stop reading as a result.
    if not (ece.get("point", float("nan")) < 0.0 and ece.get("excludes_zero")):
        return (
            f"no measurable calibration gain: ECE {score.ece:.4f}, paired "
            f"difference {ece['point']:+.4f} "
            f"[{ece['ci_lower']:.4f}, {ece['ci_upper']:.4f}] does not exclude zero"
        )
    return None


def _disqualify_on_tolerances(
    score: CalibrationScore, baseline: CalibrationScore
) -> Optional[str]:
    """
    Reject a candidate against fixed tolerances. None means it survives.

    The fallback for callers with no cluster labels to bootstrap over.
    """
    if score.mce > baseline.mce + MCE_TOLERANCE:
        return (
            f"worst-case miscalibration got worse: MCE {score.mce:.4f} vs "
            f"{baseline.mce:.4f} uncalibrated (tolerance {MCE_TOLERANCE})"
        )
    if score.brier > baseline.brier + BRIER_TOLERANCE:
        return (
            f"proper score got worse: Brier {score.brier:.4f} vs "
            f"{baseline.brier:.4f} uncalibrated (tolerance {BRIER_TOLERANCE})"
        )
    if score.ece >= baseline.ece:
        return (
            f"no calibration gain: ECE {score.ece:.4f} vs "
            f"{baseline.ece:.4f} uncalibrated"
        )
    return None


def select_calibrator(
    scores: Sequence[CalibrationScore],
    baseline_method: str = "uncalibrated",
) -> SelectionOutcome:
    """
    Choose a calibration method from validation-split scores.

    `scores` must include the baseline (the uncalibrated model), because the
    guards are expressed relative to it and because declining to calibrate has to
    be reachable.

    When every candidate carries paired bootstrap differences -- which is what
    `score_calibrators` produces -- the guards are expressed against measured
    noise: a method is rejected when its MCE or Brier is worse by an interval that
    excludes zero, and required to show an ECE gain whose interval excludes zero.
    Otherwise they fall back to the fixed tolerances below, which are a guess at
    the same noise floor. The outcome records which basis was used.
    """
    by_method = {s.method: s for s in scores}
    if baseline_method not in by_method:
        raise ValueError(
            f"select_calibrator needs the {baseline_method!r} baseline among the "
            f"candidates; got {sorted(by_method)}. Without it there is nothing to "
            f"guard against and no way to decline calibration."
        )

    baseline = by_method[baseline_method]
    candidates = [s for s in scores if s.method != baseline_method]
    use_intervals = bool(candidates) and all(s.has_intervals for s in candidates)
    disqualified: Dict[str, str] = {}
    eligible: List[CalibrationScore] = []

    for score in candidates:
        verdict = (
            _disqualify_on_intervals(score)
            if use_intervals
            else _disqualify_on_tolerances(score, baseline)
        )
        if verdict is not None:
            disqualified[score.method] = verdict
        else:
            eligible.append(score)

    basis = "bootstrap" if use_intervals else "point"
    if not eligible:
        return SelectionOutcome(
            method=baseline_method,
            reason=(
                "no candidate improved ECE without degrading worst-case "
                "calibration or the proper score; reporting the model uncalibrated"
            ),
            disqualified=disqualified,
            basis=basis,
        )

    # Ties broken by MCE then Brier, so the safer of two equally-calibrated
    # methods wins. Method name last, to keep the choice deterministic.
    winner = min(eligible, key=lambda s: (s.ece, s.mce, s.brier, s.method))
    if use_intervals:
        gain = winner.ece_vs_baseline
        reason = (
            f"lowest validation ECE {winner.ece:.4f} (vs {baseline.ece:.4f} "
            f"uncalibrated); the paired ECE improvement is "
            f"{-gain['point']:.4f} [{-gain['ci_upper']:.4f}, {-gain['ci_lower']:.4f}] "
            f"and excludes zero, and neither MCE nor Brier is measurably worse"
        )
    else:
        reason = (
            f"lowest validation ECE {winner.ece:.4f} (vs {baseline.ece:.4f} "
            f"uncalibrated) with MCE {winner.mce:.4f} and Brier {winner.brier:.4f} "
            f"both within tolerance of the uncalibrated model"
        )
    return SelectionOutcome(
        method=winner.method,
        reason=reason,
        disqualified=disqualified,
        basis=basis,
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


def _brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier score: the mean squared error of the probabilities.

    Written out rather than taken from sklearn because `brier_score_loss` infers
    the positive label from the labels present, and a bootstrap resample can
    easily contain only one class -- a normal resample here, not an error, given a
    failure rate in the single digits.
    """
    return float(np.mean((probs - y_true) ** 2))


def score_calibrators(
    y_true: np.ndarray,
    probs_by_method: Dict[str, np.ndarray],
    cluster_ids: Optional[Sequence[Any]] = None,
    baseline_method: str = "uncalibrated",
    n_bins: int = 10,
    num_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> List[CalibrationScore]:
    """
    ECE, MCE and Brier for each candidate, with paired intervals against the baseline.

    The point estimates alone cannot support the choice. On a validation half-split
    the number of *failing* rows is small -- the harvest targets a failure rate in
    the 1-15% band -- and ECE is an average over bins whose occupancy is itself
    noisy, so two methods separated by 0.002 ECE are separated by nothing. Every
    comparison the decision rests on is therefore a paired difference from a
    cluster bootstrap, and `select_calibrator` reads `excludes_zero`.

    The resampling unit is the mutant, not the row, for the same reason it is in
    the benchmark: all the tests of one mutant see one injected fault, so their
    outcomes are not independent draws, and resampling rows would report an
    interval far narrower than the evidence supports.

    Args:
        y_true: Binary labels, one per row.
        probs_by_method: method name -> probabilities over those same rows. Must
            include `baseline_method`, the uncalibrated model.
        cluster_ids: Mutant identifier per row -- `commit_sha` in the built
            dataset, which the builder sets to the mutant id. None means no
            clustering is available: point estimates only, and `select_calibrator`
            then falls back to its fixed tolerances.
        n_bins: ECE/MCE bin count, matching what the report prints.
        num_bootstraps: Resamples (default: 2000).
        ci: Confidence level (default: 0.95).
        random_seed: Reproducibility seed.

    Returns:
        One CalibrationScore per method, baseline first.
    """
    y_true = np.asarray(y_true).astype(int)
    if baseline_method not in probs_by_method:
        raise ValueError(
            f"score_calibrators needs the {baseline_method!r} probabilities among "
            f"{sorted(probs_by_method)}: every difference is taken against them."
        )
    for method, probs in probs_by_method.items():
        if len(probs) != len(y_true):
            raise ValueError(
                f"{method} has {len(probs)} probabilities for {len(y_true)} labels"
            )

    methods = [baseline_method] + [m for m in probs_by_method if m != baseline_method]
    prob_arrays = {
        m: np.asarray(probs_by_method[m], dtype=np.float64) for m in methods
    }

    def measure(method: str, rows: np.ndarray) -> Dict[str, float]:
        if len(rows) == 0:
            # An empty resample leaves all three undefined. Reporting 0.0 here
            # would read as perfect calibration measured on no data.
            return {"ece": float("nan"), "mce": float("nan"), "brier": float("nan")}
        y, p = y_true[rows], prob_arrays[method][rows]
        ece, mce, _ = compute_ece(y, p, n_bins=n_bins)
        return {"ece": float(ece), "mce": float(mce), "brier": _brier(y, p)}

    all_rows = np.arange(len(y_true), dtype=np.intp)
    points = {method: measure(method, all_rows) for method in methods}

    if cluster_ids is None:
        return [
            CalibrationScore(
                method=m,
                ece=points[m]["ece"],
                mce=points[m]["mce"],
                brier=points[m]["brier"],
            )
            for m in methods
        ]

    if len(cluster_ids) != len(y_true):
        raise ValueError(
            f"cluster_ids has {len(cluster_ids)} entries for {len(y_true)} rows"
        )
    _, row_groups = group_rows_by_cluster(cluster_ids)
    n_clusters = len(row_groups)

    metrics = ("ece", "mce", "brier")
    replicates = {
        metric: {m: np.empty(num_bootstraps, dtype=np.float64) for m in methods}
        for metric in metrics
    }
    for b, drawn in enumerate(cluster_draws(n_clusters, num_bootstraps, random_seed)):
        # Gathered once and shared by all three metrics and every method: one draw
        # per replicate is what makes the differences paired.
        rows = rows_for_clusters(row_groups, drawn)
        for method in methods:
            measured = measure(method, rows)
            for metric in metrics:
                replicates[metric][method][b] = measured[metric]

    folded = {
        metric: intervals_from_replicates(
            replicates[metric],
            {m: points[m][metric] for m in methods},
            ci=ci,
            num_units=n_clusters,
            reference=baseline_method,
        )
        for metric in metrics
    }

    return [
        CalibrationScore(
            method=m,
            ece=points[m]["ece"],
            mce=points[m]["mce"],
            brier=points[m]["brier"],
            ece_vs_baseline=folded["ece"]["differences"].get(m),
            mce_vs_baseline=folded["mce"]["differences"].get(m),
            brier_vs_baseline=folded["brier"]["differences"].get(m),
        )
        for m in methods
    ]


def split_clusters_for_selection(
    cluster_ids: Sequence[Any], holdout_fraction: float = 0.5
) -> np.ndarray:
    """
    Cut the validation split into fit and select halves along cluster boundaries.

    `split_for_selection` cuts the row array at an index, which puts some of a
    mutant's tests in the fit half and the rest in the held-out half. That is
    leakage of exactly the kind the half-split was introduced to prevent: isotonic
    regression can learn a mutant's failure pattern from the rows it was fitted on
    and be rewarded for it on that same mutant's remaining rows, which is how a
    flexible method wins a comparison it should not. Whole clusters go to one side
    or the other here.

    The halves are balanced by row count rather than by cluster count, because
    mutants differ widely in how many tests they carry.

    Args:
        cluster_ids: Mutant identifier per row.
        holdout_fraction: Share of rows to hold out for comparing candidates.

    Returns:
        Boolean mask over rows; True marks the fit half.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")
    n_rows = len(cluster_ids)
    _, row_groups = group_rows_by_cluster(cluster_ids)
    if len(row_groups) < 2:
        raise ValueError(
            f"a cluster-aware split needs at least 2 clusters; got "
            f"{len(row_groups)} over {n_rows} rows. With one cluster there is no "
            f"way to hold anything out without splitting a mutant across both "
            f"halves, which is the leak this function exists to close."
        )

    target_fit_rows = n_rows * (1.0 - holdout_fraction)
    mask = np.zeros(n_rows, dtype=bool)
    assigned = 0
    for i, rows in enumerate(row_groups):
        last = i == len(row_groups) - 1
        # Whichever side the row counts favour, the first cluster is always fitted
        # and the last is always held out, so neither half can come out empty.
        if last:
            continue
        if i == 0 or assigned + len(rows) / 2.0 <= target_fit_rows:
            mask[rows] = True
            assigned += len(rows)
    return mask
