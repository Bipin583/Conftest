"""
ConfTest Statistical Significance & Non-Parametric Hypothesis Testing Engine.

Implements Wilcoxon Signed-Rank tests, Cliff's Delta non-parametric effect sizes,
and Bootstrap 95% Confidence Intervals for empirical RTS evaluation.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta non-parametric effect size between two distributions.

    Formula:
        delta = ( #(x > y) - #(x < y) ) / (len(x) * len(y))

    Thresholds (Romano et al., 2006):
        |delta| < 0.147: Negligible
        0.147 <= |delta| < 0.330: Small
        0.330 <= |delta| < 0.474: Medium
        |delta| >= 0.474: Large

    Args:
        x: Sample distribution 1 (e.g. ConfTest metrics).
        y: Sample distribution 2 (e.g. Baseline metrics).

    Returns:
        Tuple of (delta_float, interpretation_str).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "Negligible"

    # Efficient pairwise comparison using matrix broadcasting
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])

    delta = float((greater - less) / (n_x * n_y))
    abs_d = abs(delta)

    if abs_d < 0.147:
        magnitude = "Negligible"
    elif abs_d < 0.330:
        magnitude = "Small"
    elif abs_d < 0.474:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    return round(delta, 4), magnitude


def compute_wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float, bool]:
    """
    Compute Wilcoxon Signed-Rank test for paired non-parametric samples.

    Args:
        x: ConfTest metric array across commits.
        y: Baseline metric array across paired commits.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Tuple of (statistic_W, p_value, is_significant_at_p05).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0, False

    # Filter zero differences for Wilcoxon
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) < 5:
        # Insufficient non-zero pairs for asymptotic normal approximation
        return 0.0, 1.0, False

    try:
        res = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
        stat_w = float(res.statistic)
        p_val = float(res.pvalue)
        is_sig = bool(p_val < 0.05)
        return round(stat_w, 4), round(p_val, 5), is_sig
    except Exception as exc:
        logger.warning(f"Wilcoxon calculation fallback ({exc}).")
        return 0.0, 1.0, False


def bootstrap_confidence_interval(
    data: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Compute non-parametric percentile bootstrap confidence interval.

    Args:
        data: 1D array of sample values.
        num_bootstraps: Number of bootstrap resamples (default: 1000).
        ci: Confidence level (default: 0.95 for 95% CI).
        statistic_fn: Function computing summary statistic (default: mean).
        random_seed: Reproducibility seed.

    Returns:
        Dictionary with keys: mean, median, ci_lower, ci_upper.
    """
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = np.random.RandomState(random_seed)
    boot_stats = np.empty(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    alpha = (1.0 - ci) / 2.0
    low_pct = alpha * 100.0
    high_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(boot_stats, low_pct))
    ci_upper = float(np.percentile(boot_stats, high_pct))
    point_mean = float(statistic_fn(data))
    point_median = float(np.median(data))

    return {
        "mean": round(point_mean, 4),
        "median": round(point_median, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": ci,
    }


def _as_float(value: Any) -> float:
    """Coerce a statistic's return value to float, mapping the uncomputable to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _percentile_interval(
    replicates: np.ndarray, point: float, ci: float, num_units: int
) -> Dict[str, Any]:
    """
    Fold bootstrap replicates into a percentile interval.

    Replicates where the statistic was undefined -- an empty denominator in that
    resample, for instance no failing tests among the drawn commits -- arrive as
    NaN and are excluded rather than coerced to zero, which would drag the
    interval downwards. Their count is reported, so "undefined in 40% of
    resamples" is visible rather than something the reader has to infer from a
    suspiciously wide interval.
    """
    replicates = np.asarray(replicates, dtype=np.float64)
    finite = replicates[np.isfinite(replicates)]
    alpha = (1.0 - ci) / 2.0

    interval: Dict[str, Any] = {
        "point": point,
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "confidence_level": ci,
        "num_bootstraps": int(replicates.size),
        "num_units": int(num_units),
        "undefined_replicates": int(replicates.size - finite.size),
    }
    if finite.size == 0:
        return interval

    interval["ci_lower"] = float(np.percentile(finite, alpha * 100.0))
    interval["ci_upper"] = float(np.percentile(finite, (1.0 - alpha) * 100.0))
    return interval


def intervals_from_replicates(
    replicates: Dict[str, np.ndarray],
    points: Dict[str, float],
    ci: float = 0.95,
    num_units: int = 0,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fold named replicate vectors into intervals, plus paired differences.

    Shared by both bootstrap paths -- the callable-per-statistic one and the
    counts-matrix one -- so a difference is assembled the same way regardless of
    how the replicates were produced. Requires that replicate b of every name came
    from the same resample, which is what makes the subtraction paired.
    """
    names = list(replicates)
    if reference is not None and reference not in replicates:
        raise ValueError(f"reference {reference!r} is not one of {names}")

    result: Dict[str, Any] = {
        "resampling_unit": "cluster",
        "num_units": int(num_units),
        "num_bootstraps": int(len(replicates[names[0]])) if names else 0,
        "confidence_level": ci,
        "intervals": {
            name: _percentile_interval(replicates[name], points[name], ci, num_units)
            for name in names
        },
    }
    if reference is None:
        return result

    differences: Dict[str, Any] = {}
    for name in names:
        if name == reference:
            continue
        interval = _percentile_interval(
            np.asarray(replicates[name], dtype=np.float64)
            - np.asarray(replicates[reference], dtype=np.float64),
            points[name] - points[reference],
            ci,
            num_units,
        )
        interval["reference"] = reference
        # The only question the report actually asks of a difference.
        interval["excludes_zero"] = bool(
            np.isfinite(interval["ci_lower"])
            and np.isfinite(interval["ci_upper"])
            and (interval["ci_lower"] > 0.0 or interval["ci_upper"] < 0.0)
        )
        differences[name] = interval

    result["reference"] = reference
    result["differences"] = differences
    return result


def cluster_draws(
    num_units: int, num_bootstraps: int = 2000, random_seed: int = 42
) -> Iterator[np.ndarray]:
    """
    The resample draws themselves: one array of cluster indices per replicate.

    Every bootstrap path in this module consumes this generator, so "the counts
    matrix draws the same resamples as the callable path" is true by construction
    rather than by two functions happening to call the same RNG in the same order.
    A third consumer -- the calibration metrics, which are not ratios of sums and
    so cannot use the counts matrix -- is exactly why that stopped being safe to
    leave as a convention.

    Args:
        num_units: Number of clusters available to resample.
        num_bootstraps: Number of resamples (default: 2000).
        random_seed: Reproducibility seed.

    Yields:
        For each replicate, `num_units` cluster indices drawn with replacement.
    """
    if num_bootstraps < 1:
        raise ValueError(f"num_bootstraps must be >= 1, got {num_bootstraps}")
    num_units = int(max(0, num_units))
    rng = np.random.RandomState(random_seed)
    empty = np.empty(0, dtype=np.intp)
    for _ in range(num_bootstraps):
        yield rng.randint(0, num_units, size=num_units) if num_units > 0 else empty


def group_rows_by_cluster(
    cluster_ids: Sequence[Any],
) -> Tuple[List[Any], List[np.ndarray]]:
    """
    Row indices of each cluster, in order of first appearance.

    The bootstrap resamples clusters, but ECE, MCE and Brier are computed over
    rows, so something has to hold the mapping. First-appearance order rather than
    sorted order keeps the result stable under a column of mixed types, which a
    mutant identifier column read from CSV can easily be.

    Args:
        cluster_ids: One cluster label per row.

    Returns:
        (cluster keys, row-index array per cluster), index-aligned.
    """
    groups: Dict[Any, List[int]] = {}
    for row, key in enumerate(cluster_ids):
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    return keys, [np.asarray(groups[k], dtype=np.intp) for k in keys]


def rows_for_clusters(
    row_groups: Sequence[np.ndarray], drawn: np.ndarray
) -> np.ndarray:
    """
    Rows of the drawn clusters, with multiplicity.

    A cluster drawn twice contributes its rows twice -- that is what makes this a
    bootstrap over clusters rather than a subsample of them.
    """
    if len(drawn) == 0:
        return np.empty(0, dtype=np.intp)
    return np.concatenate([row_groups[i] for i in drawn])


def bootstrap_counts(
    num_units: int, num_bootstraps: int = 2000, random_seed: int = 42
) -> np.ndarray:
    """
    Multiplicity matrix for a cluster bootstrap: row b says how many times each
    cluster was drawn in replicate b.

    The draws come from cluster_draws, the one generator every bootstrap path in
    this module consumes, so all of them agree replicate for replicate under one
    seed. Counts are the useful form when the statistic is a ratio of sums over
    clusters: every replicate's numerator and denominator is then one matrix
    product away rather than one Python call away, which is the difference between
    seconds and minutes at 8 strategies x 5 metrics x 2000 replicates.
    """
    num_units = int(max(0, num_units))
    counts = np.zeros((num_bootstraps, num_units), dtype=np.float64)
    for b, idx in enumerate(cluster_draws(num_units, num_bootstraps, random_seed)):
        if num_units > 0:
            counts[b] = np.bincount(idx, minlength=num_units)
    return counts


def paired_cluster_bootstrap(
    num_units: int,
    statistic_fns: Dict[str, Callable[[np.ndarray], float]],
    num_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Percentile bootstrap over clusters, with every statistic sharing one resample.

    ConfTest's headline numbers -- test reduction, time reduction, failure recall
    -- are ratios pooled over commits, and the test rows within a commit are not
    independent: they see the same diff and the same injected fault. Resampling
    rows would quote an interval far narrower than the evidence supports. The unit
    of resampling here is therefore the cluster (a commit), and each statistic is
    handed the indices of the clusters drawn for a replicate and pools over them
    itself.

    Every statistic sees the *same* draw, which is what makes a difference between
    two of them paired. That matters more than the marginal intervals: the claim in
    the report is that ConfTest recovers more failures than a baseline at equal
    budget, and two overlapping marginal intervals do not settle whether the
    difference itself excludes zero.

    Args:
        num_units: Number of clusters available to resample.
        statistic_fns: Named statistics. Each takes an index array of resampled
            clusters and returns a scalar, or NaN when that resample leaves it
            undefined.
        num_bootstraps: Number of resamples (default: 2000).
        ci: Confidence level (default: 0.95).
        random_seed: Reproducibility seed.
        reference: Statistic that differences are taken against. When given,
            'differences' carries one paired interval per other statistic.

    Returns:
        Dict with 'intervals' (name -> interval), optionally 'differences', and the
        resampling metadata. Each interval carries point, ci_lower, ci_upper and
        undefined_replicates.
    """
    if num_bootstraps < 1:
        raise ValueError(f"num_bootstraps must be >= 1, got {num_bootstraps}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie in (0, 1), got {ci}")
    names = list(statistic_fns)
    if reference is not None and reference not in statistic_fns:
        raise ValueError(f"reference {reference!r} is not one of {names}")

    num_units = int(max(0, num_units))
    points = {name: _as_float(fn(np.arange(num_units))) for name, fn in statistic_fns.items()}

    replicates = {name: np.empty(num_bootstraps, dtype=np.float64) for name in names}

    # One draw per replicate, reused by every statistic: this is the pairing.
    for b, idx in enumerate(cluster_draws(num_units, num_bootstraps, random_seed)):
        for name, fn in statistic_fns.items():
            replicates[name][b] = _as_float(fn(idx))

    return intervals_from_replicates(
        replicates, points, ci=ci, num_units=num_units, reference=reference
    )

def cluster_bootstrap_ci(
    num_units: int,
    statistic_fn: Callable[[np.ndarray], float],
    num_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Percentile bootstrap interval for a single cluster-pooled statistic.

    Use this when there is one number to put an interval on. When several numbers
    will be compared against each other, use paired_cluster_bootstrap so the
    comparison is drawn from a shared resample.
    """
    return paired_cluster_bootstrap(
        num_units,
        {"statistic": statistic_fn},
        num_bootstraps=num_bootstraps,
        ci=ci,
        random_seed=random_seed,
    )["intervals"]["statistic"]


def format_interval(interval: Dict[str, Any], precision: int = 1, unit: str = "") -> str:
    """
    Render an interval as 'point [lower, upper]' for a report table.

    An undefined interval renders as 'n/a', never as 'nan': a table printing nan
    invites the reader to treat it as a measurement.
    """
    point = _as_float(interval.get("point", float("nan")))
    low = _as_float(interval.get("ci_lower", float("nan")))
    high = _as_float(interval.get("ci_upper", float("nan")))
    if not np.isfinite(point):
        return "n/a"
    head = f"{point:.{precision}f}{unit}"
    if not (np.isfinite(low) and np.isfinite(high)):
        return f"{head} [no interval]"
    return f"{head} [{low:.{precision}f}, {high:.{precision}f}]"




class StatisticalSignificanceTester:
    """Orchestrates comprehensive pairwise statistical significance testing across RTS baselines."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def evaluate_pairwise(
        self,
        conftest_metrics: Dict[str, np.ndarray],
        baseline_metrics: Dict[str, np.ndarray],
        baseline_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate ConfTest vs. a specific baseline across Failure Recall and Time Reduction.

        Args:
            conftest_metrics: Dict with keys 'failure_recall' and 'time_reduction'.
            baseline_metrics: Dict with keys 'failure_recall' and 'time_reduction'.
            baseline_name: Name of the baseline being compared.

        Returns:
            Structured statistical report dictionary.
        """
        results = {"baseline_name": baseline_name}

        for metric in ("failure_recall", "time_reduction"):
            c_vals = conftest_metrics.get(metric, np.array([]))
            b_vals = baseline_metrics.get(metric, np.array([]))

            w_stat, p_val, is_sig = compute_wilcoxon_test(c_vals, b_vals, alternative="two-sided")
            delta, magnitude = compute_cliffs_delta(c_vals, b_vals)
            c_boot = bootstrap_confidence_interval(c_vals, random_seed=self.random_seed)
            b_boot = bootstrap_confidence_interval(b_vals, random_seed=self.random_seed)

            results[metric] = {
                "wilcoxon_W": w_stat,
                "p_value": p_val,
                "statistically_significant_p05": is_sig,
                "cliffs_delta": delta,
                "effect_size": magnitude,
                "conftest_95ci": c_boot,
                "baseline_95ci": b_boot,
            }

        return results
