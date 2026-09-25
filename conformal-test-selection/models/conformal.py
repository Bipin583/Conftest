"""Layer 3: split conformal prediction over calibrated failure probabilities.

This is the contribution of the project. Layers 1 and 2 produce a calibrated
``P(test fails | change)``; on their own they give no guarantee, because a
probability threshold chosen on one distribution says nothing about the recall
attained on the next. Split conformal prediction converts the score into a
selection rule with a *distribution-free, finite-sample* coverage guarantee: no
assumption about the model being correct, only that the calibration and
deployment rows are exchangeable.

**Construction.** For every *failing* calibration row the non-conformity score
is ``s = 1 - p_calibrated``. A test is selected at inference when
``1 - p <= q``, i.e. ``p >= 1 - q``. The threshold ``q`` is the ``k``-th
smallest calibration score, and everything turns on the choice of ``k``.

**The two guarantees are not the same thing.**

``marginal``
    ``k = ceil((n + 1) * (1 - alpha))``. Coverage holds *on average over
    calibration sets*: ``P(fail covered) >= 1 - alpha``. This is the standard
    construction and matches the ``np.quantile(scores, (1 - alpha) * (1 + 1/n))``
    idiom from the conformal tutorial (arXiv:2107.07511). Its weakness is that
    the realised coverage of any one deployed threshold is itself random -- for
    the calibration set you actually drew, coverage could sit below the target
    roughly half the time.

``pac`` (default)
    Training-conditional, a.k.a. a tolerance region (Vovk, 2012): pick the
    smallest ``k`` such that ``BetaInv(delta; k, n + 1 - k) >= 1 - alpha``. The
    resulting threshold satisfies "with probability ``1 - delta`` over the draw
    of the calibration set, the deployed rule covers at least ``1 - alpha`` of
    future failures". The order statistic of the coverage of a fixed conformal
    threshold is Beta-distributed, which is where the inverse-Beta comes from.

The project's stated requirement -- *95% coverage at 90% confidence* -- is a
statement with two numbers in it, so it is the PAC guarantee, not the marginal
one. ``config.yaml`` therefore sets ``conformal.guarantee: pac``; the marginal
threshold is always computed alongside it for comparison, and the gap between
them is the price of the stronger statement.

**Class-conditional (Mondrian).** The quantile is taken over failing rows only.
Marginal conformal over all rows would spend its error budget on the 95% of
rows that pass, which is not the quantity anyone wants guaranteed here: the
promise that matters is "of the tests that were going to fail, we selected at
least 95%".

Example:
    >>> from models.conformal import fit_conformal, ConformalSelector
    >>> report = fit_conformal()
    >>> report["selected_threshold"]["empirical_coverage_test"]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ArtifactError, get_logger, load_artifact, load_config, resolve_path, save_artifact  # noqa: E402
from models.calibrate import CalibrationError, load_calibrated_scorer  # noqa: E402
from models.train import load_split  # noqa: E402

LOGGER = get_logger(__name__)


class ConformalError(RuntimeError):
    """Raised when a conformal threshold cannot be computed or applied."""


# --------------------------------------------------------------------------
# Quantile levels
# --------------------------------------------------------------------------


def marginal_rank(n: int, alpha: float) -> int:
    """Order statistic giving marginal ``1 - alpha`` coverage.

    Args:
        n: Number of calibration scores.
        alpha: Miscoverage rate, e.g. ``0.05`` for 95% coverage.

    Returns:
        The rank ``k`` in ``1..n`` whose ``k``-th smallest score is the
        threshold. Clipped to ``n`` when the requested level is unreachable.

    Raises:
        ConformalError: If ``n`` is not positive or ``alpha`` is out of range.
    """
    if n <= 0:
        raise ConformalError("Cannot compute a conformal rank from an empty calibration set.")
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must lie in (0, 1); got {alpha}.")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # (n+1)(1-alpha) > n means the calibration set is too small to certify
        # this level. Falling back to the maximum score is the conservative
        # choice: it selects everything rather than silently under-covering.
        LOGGER.warning(
            "Calibration set of %d rows is too small for %.1f%% marginal coverage "
            "(needs >= %d); using the maximum score.",
            n, 100 * (1 - alpha), math.ceil(1.0 / alpha) - 1,
        )
        k = n
    return int(k)


def pac_rank(n: int, alpha: float, delta: float) -> Optional[int]:
    """Order statistic giving training-conditional (PAC) coverage.

    Finds the smallest ``k`` with ``BetaInv(delta; k, n + 1 - k) >= 1 - alpha``,
    so that with probability at least ``1 - delta`` over the calibration draw,
    the deployed threshold covers at least ``1 - alpha`` of future positives
    (Vovk, 2012).

    Args:
        n: Number of calibration scores.
        alpha: Miscoverage rate.
        delta: Failure probability of the guarantee itself, e.g. ``0.10`` for
            90% confidence.

    Returns:
        The rank ``k``, or ``None`` if no ``k <= n`` achieves the level -- the
        calibration set is simply too small for that pair of numbers.

    Raises:
        ConformalError: If SciPy is unavailable or the arguments are invalid.
    """
    if n <= 0:
        raise ConformalError("Cannot compute a conformal rank from an empty calibration set.")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ConformalError(f"alpha and delta must lie in (0, 1); got alpha={alpha}, delta={delta}.")

    try:
        from scipy.stats import beta as beta_dist
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ConformalError("SciPy is required for the PAC guarantee: pip install scipy") from exc

    target = 1.0 - alpha
    # Coverage is monotone in k, so the feasible set is an upper interval and a
    # binary search finds its left endpoint in O(log n) Beta evaluations
    # instead of the O(n) a linear scan would need on a 4600-row split.
    lo, hi = 1, n
    best: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        coverage = float(beta_dist.ppf(delta, mid, n + 1 - mid))
        if coverage >= target:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best is None:
        LOGGER.warning(
            "No rank <= %d certifies %.1f%% coverage at %.1f%% confidence; "
            "the calibration set is too small for a PAC guarantee at this level.",
            n, 100 * target, 100 * (1 - delta),
        )
    return best


def _achieved_pac_coverage(n: int, k: int, delta: float) -> Optional[float]:
    """Coverage actually certified by rank ``k`` at confidence ``1 - delta``.

    Args:
        n: Calibration set size.
        k: Chosen rank.
        delta: Failure probability of the guarantee.

    Returns:
        The certified coverage lower bound, or ``None`` if SciPy is missing.
    """
    try:
        from scipy.stats import beta as beta_dist
    except ImportError:  # pragma: no cover
        return None
    if not 1 <= k <= n:
        return None
    return float(beta_dist.ppf(delta, k, n + 1 - k))


# --------------------------------------------------------------------------
# Selector
# --------------------------------------------------------------------------


@dataclass
class ConformalSelector:
    """A fitted conformal selection rule.

    Attributes:
        threshold: Non-conformity cut-off ``q``. A test is selected when
            ``1 - p <= threshold``.
        probability_floor: The equivalent probability cut-off, ``1 - threshold``.
            Selecting on this is identical and avoids a subtraction at serving
            time.
        guarantee: ``"pac"`` or ``"marginal"``.
        coverage: Target coverage ``1 - alpha``.
        confidence: ``1 - delta`` for the PAC guarantee; ``None`` for marginal.
        rank: The order statistic used.
        n_calibration: Number of calibration scores the quantile was taken over.
        certified_coverage: Coverage the chosen rank actually certifies.
        class_conditional: Whether the quantile used failing rows only.
    """

    threshold: float
    probability_floor: float
    guarantee: str
    coverage: float
    confidence: Optional[float]
    rank: int
    n_calibration: int
    certified_coverage: Optional[float] = None
    class_conditional: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def select(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the rule to calibrated probabilities.

        Args:
            probabilities: Calibrated ``P(fail)`` per candidate test.

        Returns:
            A boolean mask; ``True`` means run the test.
        """
        scores = 1.0 - np.asarray(probabilities, dtype=float).ravel()
        return scores <= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the rule."""
        return asdict(self)


def nonconformity_scores(probabilities: np.ndarray) -> np.ndarray:
    """Convert calibrated probabilities into non-conformity scores.

    Args:
        probabilities: Calibrated ``P(fail)``.

    Returns:
        ``1 - p``: small when the model confidently expects a failure, which is
        exactly when a truly-failing row conforms to the model.
    """
    return 1.0 - np.asarray(probabilities, dtype=float).ravel()


def build_selector(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    coverage: float = 0.95,
    confidence: float = 0.90,
    guarantee: str = "pac",
    class_conditional: bool = True,
) -> Dict[str, Any]:
    """Fit conformal thresholds from a held-out calibration split.

    Both the marginal and PAC thresholds are computed; ``guarantee`` decides
    which one is returned as the selector to deploy.

    Args:
        calibration_probabilities: Calibrated ``P(fail)`` on the calibration split.
        calibration_labels: Ground-truth labels for those rows.
        coverage: Target coverage ``1 - alpha``.
        confidence: ``1 - delta`` for the PAC guarantee.
        guarantee: ``"pac"`` or ``"marginal"``.
        class_conditional: Take the quantile over failing rows only (Mondrian).

    Returns:
        A dictionary with ``selector`` (the deployed :class:`ConformalSelector`)
        and ``variants`` (both thresholds, for comparison).

    Raises:
        ConformalError: If the calibration split has no failing rows, or the
            requested guarantee is unknown.
    """
    guarantee = str(guarantee).lower()
    if guarantee not in {"pac", "marginal"}:
        raise ConformalError(f"Unknown guarantee {guarantee!r}; use 'pac' or 'marginal'.")

    probabilities = np.asarray(calibration_probabilities, dtype=float).ravel()
    labels = np.asarray(calibration_labels).ravel()
    if probabilities.size != labels.size:
        raise ConformalError(
            f"Probabilities and labels must align; got {probabilities.size} and {labels.size}."
        )

    if class_conditional:
        mask = labels == 1
        if not mask.any():
            raise ConformalError(
                "Class-conditional conformal needs failing rows in the calibration split, found none."
            )
        scores = nonconformity_scores(probabilities[mask])
    else:
        scores = nonconformity_scores(probabilities)

    scores = np.sort(scores)
    n = int(scores.size)
    alpha = 1.0 - float(coverage)
    delta = 1.0 - float(confidence)

    variants: Dict[str, Any] = {}

    k_marginal = marginal_rank(n, alpha)
    variants["marginal"] = {
        "rank": k_marginal,
        # Ranks are 1-based; array indices are 0-based.
        "threshold": float(scores[k_marginal - 1]),
        "certified_coverage": None,
        "note": "coverage holds on average over calibration draws",
    }

    k_pac = pac_rank(n, alpha, delta)
    if k_pac is None:
        variants["pac"] = {
            "rank": None,
            "threshold": None,
            "certified_coverage": None,
            "note": f"no rank <= {n} certifies {coverage:.0%} coverage at {confidence:.0%} confidence",
        }
    else:
        variants["pac"] = {
            "rank": k_pac,
            "threshold": float(scores[k_pac - 1]),
            "certified_coverage": _achieved_pac_coverage(n, k_pac, delta),
            "note": f"holds with probability {confidence:.0%} over the calibration draw",
        }

    chosen = variants[guarantee]
    if chosen["threshold"] is None:
        # Falling back keeps the pipeline running and says so loudly, rather
        # than shipping a rule whose stated guarantee is not actually met.
        LOGGER.error(
            "PAC guarantee unattainable with %d calibration failures; falling back to the "
            "marginal threshold. The %.0f%%-confidence claim does NOT hold for this run.",
            n, 100 * confidence,
        )
        guarantee = "marginal"
        chosen = variants["marginal"]

    threshold = float(chosen["threshold"])
    selector = ConformalSelector(
        threshold=threshold,
        probability_floor=1.0 - threshold,
        guarantee=guarantee,
        coverage=float(coverage),
        confidence=float(confidence) if guarantee == "pac" else None,
        rank=int(chosen["rank"]),
        n_calibration=n,
        certified_coverage=chosen["certified_coverage"],
        class_conditional=bool(class_conditional),
        metadata={"score_min": float(scores[0]), "score_max": float(scores[-1])},
    )

    LOGGER.info(
        "Conformal (%s): n=%d failures, rank=%d/%d, threshold=%.6f -> select when P(fail) >= %.6f",
        guarantee, n, selector.rank, n, selector.threshold, selector.probability_floor,
    )
    if selector.certified_coverage is not None:
        LOGGER.info(
            "Certified: >= %.2f%% coverage with %.0f%% confidence (target %.0f%%).",
            100 * selector.certified_coverage, 100 * confidence, 100 * coverage,
        )
    LOGGER.info(
        "Marginal threshold for comparison: %.6f (rank %d) -- the gap is the cost of the PAC claim.",
        variants["marginal"]["threshold"], variants["marginal"]["rank"],
    )

    return {"selector": selector, "variants": variants}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def selection_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    selector: ConformalSelector,
    groups: Optional[Sequence[Any]] = None,
    min_tests: int = 0,
    max_tests: Optional[int] = None,
) -> Dict[str, Any]:
    """Score a selection rule the way a CI owner would read it.

    Args:
        labels: Ground-truth failure labels.
        probabilities: Calibrated ``P(fail)``.
        selector: The fitted rule.
        groups: Per-row commit identifier, enabling per-commit statistics.
        min_tests: Minimum tests to run per commit, applied as a top-up.
        max_tests: Optional cap on tests per commit.

    Returns:
        A metrics dictionary covering coverage (recall over failures),
        selection rate, cost reduction, and per-commit safety.
    """
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    selected = selector.select(probabilities)

    if groups is not None:
        selected = _apply_budget(np.asarray(groups), probabilities, selected, min_tests, max_tests)

    n_total = int(labels.size)
    n_failures = int((labels == 1).sum())
    n_selected = int(selected.sum())
    caught = int(((labels == 1) & selected).sum())
    missed = n_failures - caught

    metrics: Dict[str, Any] = {
        "n_rows": n_total,
        "n_failures": n_failures,
        "n_selected": n_selected,
        "failures_caught": caught,
        "failures_missed": missed,
        # Empirical coverage == recall over failing rows: the quantity the
        # conformal guarantee is a lower bound on.
        "empirical_coverage": float(caught / n_failures) if n_failures else float("nan"),
        "selection_rate": float(n_selected / n_total) if n_total else 0.0,
        "cost_reduction": float(1.0 - n_selected / n_total) if n_total else 0.0,
        "precision": float(caught / n_selected) if n_selected else 0.0,
    }

    if groups is not None:
        metrics.update(_per_commit_metrics(np.asarray(groups), labels, selected))
    return metrics


def _apply_budget(
    groups: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    min_tests: int,
    max_tests: Optional[int],
) -> np.ndarray:
    """Enforce per-commit floors and caps on top of the conformal mask.

    The floor tops a commit up with its highest-risk unselected tests; the cap
    drops its lowest-risk selected ones. Both are operational overrides, and a
    cap in particular can break the coverage guarantee -- which is why
    :func:`fit_conformal` reports metrics with and without the budget.

    Args:
        groups: Per-row commit identifier.
        probabilities: Calibrated ``P(fail)``.
        selected: The conformal mask.
        min_tests: Per-commit minimum.
        max_tests: Per-commit maximum, or ``None``.

    Returns:
        The adjusted boolean mask.
    """
    adjusted = selected.copy()
    if min_tests <= 0 and max_tests is None:
        return adjusted

    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, groups[order][1:] != groups[order][:-1]])
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(order)]):
        idx = order[start:stop]
        chosen = adjusted[idx]
        n_chosen = int(chosen.sum())

        if min_tests and n_chosen < min_tests:
            deficit = min(min_tests, idx.size) - n_chosen
            spare = idx[~chosen]
            if spare.size:
                top_up = spare[np.argsort(probabilities[spare])[::-1][:deficit]]
                adjusted[top_up] = True

        if max_tests is not None and int(adjusted[idx].sum()) > max_tests:
            kept = idx[adjusted[idx]]
            drop = kept[np.argsort(probabilities[kept])[: int(adjusted[idx].sum()) - max_tests]]
            adjusted[drop] = False

    return adjusted


def _per_commit_metrics(groups: np.ndarray, labels: np.ndarray, selected: np.ndarray) -> Dict[str, Any]:
    """Aggregate selection quality per commit.

    A CI system is judged per push, not per row: one commit whose only failing
    test was skipped is a broken build, however good the pooled recall looks.

    Args:
        groups: Per-row commit identifier.
        labels: Ground-truth labels.
        selected: Final selection mask.

    Returns:
        Per-commit counts and the fraction of failing commits fully caught.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    n_commits = unique.size

    failures = np.bincount(inverse, weights=(labels == 1).astype(float), minlength=n_commits)
    caught = np.bincount(inverse, weights=((labels == 1) & selected).astype(float), minlength=n_commits)
    sizes = np.bincount(inverse, minlength=n_commits).astype(float)
    picked = np.bincount(inverse, weights=selected.astype(float), minlength=n_commits)

    failing = failures > 0
    fully_caught = failing & (caught == failures)
    any_caught = failing & (caught > 0)

    return {
        "n_commits": int(n_commits),
        "n_failing_commits": int(failing.sum()),
        "commits_fully_caught": int(fully_caught.sum()),
        "commit_full_catch_rate": float(fully_caught.sum() / failing.sum()) if failing.any() else float("nan"),
        "commit_any_catch_rate": float(any_caught.sum() / failing.sum()) if failing.any() else float("nan"),
        "mean_tests_selected_per_commit": float(picked.mean()),
        "mean_tests_available_per_commit": float(sizes.mean()),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def fit_conformal(
    threshold_path: Optional[Any] = None,
    report_path: Optional[Any] = None,
    processed_dir: Optional[Any] = None,
    calibration_split: str = "val",
) -> Dict[str, Any]:
    """Fit the conformal threshold and evaluate it on the test split.

    The threshold is taken on the validation split -- the same split the
    calibrator was fitted on. That is deliberate and safe: the calibration map
    is a two-parameter sigmoid, so the induced optimism is negligible next to
    the alternative of carving a fourth split out of the data. The reported
    coverage is measured on the untouched test split either way.

    Args:
        threshold_path: Destination for the threshold artefact.
        report_path: Destination for the JSON report.
        processed_dir: Directory of split CSVs.
        calibration_split: Split used for the quantile.

    Returns:
        The conformal report, also written to ``reports/conformal_report.json``.

    Raises:
        ConformalError: If the model or calibrator artefacts are missing.
    """
    cfg = load_config()
    conformal_cfg = cfg["conformal"]
    selection_cfg = cfg.get("selection", {})

    try:
        base_model, calibrator, feature_columns = load_calibrated_scorer()
    except CalibrationError as exc:
        raise ConformalError(str(exc)) from exc

    X_cal, y_cal, ids_cal = load_split(calibration_split, processed_dir)
    X_test, y_test, ids_test = load_split("test", processed_dir)

    p_cal = calibrator.transform(base_model.predict_proba(X_cal.loc[:, feature_columns])[:, 1])
    p_test = calibrator.transform(base_model.predict_proba(X_test.loc[:, feature_columns])[:, 1])

    fitted = build_selector(
        p_cal,
        y_cal,
        coverage=float(conformal_cfg.get("coverage", 0.95)),
        confidence=float(conformal_cfg.get("confidence", 0.90)),
        guarantee=str(conformal_cfg.get("guarantee", "pac")),
        class_conditional=bool(conformal_cfg.get("class_conditional", True)),
    )
    selector: ConformalSelector = fitted["selector"]

    group_column = cfg["data"].get("group_column", "commit_sha")
    groups_test = ids_test[group_column].to_numpy() if group_column in ids_test else None

    unbudgeted = selection_metrics(y_test, p_test, selector, groups=groups_test)
    budgeted = selection_metrics(
        y_test,
        p_test,
        selector,
        groups=groups_test,
        min_tests=int(selection_cfg.get("min_tests", 0) or 0),
        max_tests=selection_cfg.get("max_tests"),
    )

    report: Dict[str, Any] = {
        "calibration_split": calibration_split,
        "selector": selector.to_dict(),
        "variants": fitted["variants"],
        "test_metrics": unbudgeted,
        "test_metrics_with_budget": budgeted,
        "budget": {
            "min_tests": selection_cfg.get("min_tests"),
            "max_tests": selection_cfg.get("max_tests"),
        },
        "calibration_metrics": selection_metrics(y_cal, p_cal, selector),
    }
    report["selected_threshold"] = {
        "guarantee": selector.guarantee,
        "probability_floor": selector.probability_floor,
        "empirical_coverage_test": unbudgeted["empirical_coverage"],
        "selection_rate_test": unbudgeted["selection_rate"],
        "cost_reduction_test": unbudgeted["cost_reduction"],
    }

    LOGGER.info(
        "Test: coverage %.4f | selection rate %.4f | cost reduction %.4f | %d/%d failures caught",
        unbudgeted["empirical_coverage"], unbudgeted["selection_rate"],
        unbudgeted["cost_reduction"], unbudgeted["failures_caught"], unbudgeted["n_failures"],
    )
    if "commit_full_catch_rate" in unbudgeted:
        LOGGER.info(
            "Per-commit: %.4f of failing commits fully caught, %.1f of %.1f tests run on average.",
            unbudgeted["commit_full_catch_rate"],
            unbudgeted["mean_tests_selected_per_commit"],
            unbudgeted["mean_tests_available_per_commit"],
        )

    destination = save_artifact(
        {"selector": selector, "variants": fitted["variants"], "report": report},
        threshold_path or cfg["artifacts"]["conformal_path"],
    )
    LOGGER.info("Saved conformal threshold to %s", destination)

    target = resolve_path(report_path or Path(cfg["artifacts"]["reports_dir"]) / "conformal_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Wrote %s", target)
    return report


def load_selector(threshold_path: Optional[Any] = None) -> ConformalSelector:
    """Load a fitted conformal selector.

    Args:
        threshold_path: Artefact path. Defaults to config.

    Returns:
        The fitted :class:`ConformalSelector`.

    Raises:
        ConformalError: If the artefact is missing.
    """
    cfg = load_config()
    try:
        bundle = load_artifact(threshold_path or cfg["artifacts"]["conformal_path"])
    except ArtifactError as exc:
        raise ConformalError(f"{exc} Run 'python cli.py conformal' first.") from exc
    return bundle["selector"]


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m models.conformal``.

    Returns:
        ``0`` on success, ``1`` on a handled conformal error.
    """
    parser = argparse.ArgumentParser(description="Fit the conformal selection threshold.")
    parser.add_argument("--threshold-path", default=None, help="Destination threshold path.")
    parser.add_argument("--report-path", default=None, help="Destination report path.")
    parser.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    parser.add_argument(
        "--calibration-split",
        default="val",
        choices=["val", "test"],
        help="Split used for the quantile (default: val).",
    )
    args = parser.parse_args(argv)

    try:
        fit_conformal(
            threshold_path=args.threshold_path,
            report_path=args.report_path,
            processed_dir=args.processed_dir,
            calibration_split=args.calibration_split,
        )
    except ConformalError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Re-import under the canonical module name before running. Executing this
    # file as ``python -m models.conformal`` would otherwise bind ConformalSelector to
    # ``__main__``, and joblib records that name inside the artefact -- so the
    # pickle would only load back in a process whose ``__main__`` happens to be
    # this same file. Importing the canonical copy pins ConformalSelector.__module__
    # to "models.conformal" and makes the artefact loadable from anywhere.
    from models.conformal import main as _main

    raise SystemExit(_main())
