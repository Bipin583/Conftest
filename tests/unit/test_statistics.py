"""
Unit tests for Statistical Significance and Non-Parametric Hypothesis Testing.
"""

import numpy as np
import pytest

from conftest.evaluation.statistics import (
    cluster_draws,
    group_rows_by_cluster,
    rows_for_clusters,
    compute_cliffs_delta,
    compute_wilcoxon_test,
    bootstrap_confidence_interval,
    bootstrap_counts,
    cluster_bootstrap_ci,
    format_interval,
    paired_cluster_bootstrap,
    StatisticalSignificanceTester,
)


def test_cliffs_delta_extremes():
    """Verify Cliff's delta calculations for boundary distributions."""
    # 1. Perfectly superior distribution
    x_high = np.array([10, 11, 12, 13, 14])
    y_low = np.array([1, 2, 3, 4, 5])
    delta_1, mag_1 = compute_cliffs_delta(x_high, y_low)
    assert delta_1 == 1.0
    assert mag_1 == "Large"

    # 2. Identical distributions
    same_x = np.array([5, 5, 5, 5])
    same_y = np.array([5, 5, 5, 5])
    delta_2, mag_2 = compute_cliffs_delta(same_x, same_y)
    assert delta_2 == 0.0
    assert mag_2 == "Negligible"

    # 3. Inverted distribution
    delta_3, mag_3 = compute_cliffs_delta(y_low, x_high)
    assert delta_3 == -1.0
    assert mag_3 == "Large"


def test_wilcoxon_signed_rank_paired():
    """Verify Wilcoxon test detects significant difference on paired distributions."""
    rng = np.random.RandomState(42)
    # Distinct superior performance
    conftest_scores = rng.normal(0.98, 0.02, 30)
    baseline_scores = rng.normal(0.70, 0.05, 30)

    w_stat, p_val, is_sig = compute_wilcoxon_test(conftest_scores, baseline_scores)
    assert p_val < 0.01
    assert is_sig is True


def test_bootstrap_confidence_interval_bounds():
    """Verify Bootstrap 95% CI bounds consistency."""
    rng = np.random.RandomState(42)
    sample_data = rng.normal(50.0, 5.0, 100)

    ci_dict = bootstrap_confidence_interval(sample_data, num_bootstraps=500, ci=0.95)

    assert "mean" in ci_dict
    assert "ci_lower" in ci_dict
    assert "ci_upper" in ci_dict
    assert ci_dict["ci_lower"] <= ci_dict["mean"] <= ci_dict["ci_upper"]
    assert 48.0 < ci_dict["mean"] < 52.0


def test_statistical_significance_tester_pairwise():
    """Verify StatisticalSignificanceTester end-to-end report generation."""
    rng = np.random.RandomState(42)
    c_metrics = {
        "failure_recall": rng.uniform(0.95, 1.0, 20),
        "time_reduction": rng.uniform(0.60, 0.75, 20),
    }
    b_metrics = {
        "failure_recall": rng.uniform(0.20, 0.40, 20),
        "time_reduction": rng.uniform(0.70, 0.80, 20),
    }

    tester = StatisticalSignificanceTester()
    report = tester.evaluate_pairwise(c_metrics, b_metrics, "Random-K")

    assert report["baseline_name"] == "Random-K"
    assert "failure_recall" in report
    assert "time_reduction" in report
    assert report["failure_recall"]["statistically_significant_p05"] is True
    assert report["failure_recall"]["effect_size"] == "Large"


def test_pairing_is_what_makes_a_difference_measurable():
    """
    Two strategies whose marginal intervals overlap can still differ decisively.

    Strategy b beats a by exactly 5 points on every cluster, but both vary widely
    across clusters. Comparing the marginal intervals would call that a tie; the
    paired difference, taken on a shared resample, does not.
    """
    a = np.linspace(0.2, 0.8, 40)
    b = a + 0.05

    result = paired_cluster_bootstrap(
        len(a),
        {"a": lambda i: float(a[i].mean()), "b": lambda i: float(b[i].mean())},
        num_bootstraps=500,
        reference="b",
    )
    marginal_a = result["intervals"]["a"]
    marginal_b = result["intervals"]["b"]
    # The marginal intervals overlap: neither excludes the other's point estimate.
    assert marginal_a["ci_upper"] > marginal_b["point"]

    difference = result["differences"]["a"]
    assert difference["reference"] == "b"
    assert difference["excludes_zero"] is True
    assert difference["ci_upper"] < 0.0
    assert difference["point"] == pytest.approx(-0.05)


def test_the_counts_matrix_draws_the_same_resamples_as_the_callable_path():
    """
    bootstrap_counts claims to reproduce paired_cluster_bootstrap's draws exactly.

    The benchmark relies on it: it bootstraps 8 strategies x 5 metrics by matrix
    product rather than by 80,000 Python calls, and that shortcut is only sound if
    the draws are the same ones.
    """
    numerator = np.arange(1.0, 31.0)
    denominator = np.full(30, 4.0)

    loop = paired_cluster_bootstrap(
        30,
        {"ratio": lambda i: float(numerator[i].sum() / denominator[i].sum())},
        num_bootstraps=200,
        random_seed=7,
    )["intervals"]["ratio"]

    counts = bootstrap_counts(30, 200, random_seed=7)
    replicates = (counts @ numerator) / (counts @ denominator)

    # Same draws, so the same interval up to the order the sums are accumulated in.
    assert float(np.percentile(replicates, 2.5)) == pytest.approx(loop["ci_lower"])
    assert float(np.percentile(replicates, 97.5)) == pytest.approx(loop["ci_upper"])
    # Each replicate draws n clusters with replacement, so multiplicities sum to n.
    assert (counts.sum(axis=1) == 30).all()


def test_an_undefined_replicate_is_counted_not_silently_zeroed():
    """
    A resample can leave a ratio undefined; that is a fact to report, not a zero.

    Only one of ten clusters carries any denominator, so a resample missing it has
    nothing to divide by. Coercing those replicates to zero -- which the old
    max(1, denominator) guard effectively did -- would drag the lower bound down to
    a value no resample ever produced.
    """
    def statistic(idx: np.ndarray) -> float:
        return 5.0 if 0 in set(idx.tolist()) else float("nan")

    interval = cluster_bootstrap_ci(10, statistic, num_bootstraps=200, random_seed=3)

    assert interval["undefined_replicates"] > 0
    assert interval["num_bootstraps"] == 200
    assert interval["ci_lower"] == 5.0 and interval["ci_upper"] == 5.0


def test_a_metric_that_is_never_defined_reports_no_interval():
    """An interval that does not exist must not be rendered as a number."""
    interval = cluster_bootstrap_ci(
        5, lambda idx: float("nan"), num_bootstraps=20, random_seed=1
    )

    assert interval["undefined_replicates"] == 20
    assert np.isnan(interval["ci_lower"]) and np.isnan(interval["ci_upper"])
    assert format_interval(interval) == "n/a"


def test_format_interval_renders_point_and_bounds():
    """The report table's cell format, including the unit."""
    rendered = format_interval(
        {"point": 72.36, "ci_lower": 68.1, "ci_upper": 76.24}, precision=1, unit="%"
    )
    assert rendered == "72.4% [68.1, 76.2]"


def test_a_reference_must_name_one_of_the_statistics():
    """A typo in the reference strategy must fail loudly, not silently skip pairing."""
    with pytest.raises(ValueError, match="reference"):
        paired_cluster_bootstrap(
            5, {"a": lambda i: 1.0}, num_bootstraps=10, reference="conftest"
        )


def test_every_bootstrap_path_draws_from_one_shared_generator():
    """
    The counts matrix is the multiplicities of cluster_draws' own draws.

    An earlier version had two functions reimplement the same RNG walk and relied
    on nobody disturbing the order of the calls. The calibration metrics are a
    third consumer -- they are not ratios of sums, so they gather rows per draw
    instead of multiplying a counts matrix -- and three copies of a convention is
    one too many for the pairing to rest on.
    """
    draws = list(cluster_draws(6, 5, random_seed=11))
    counts = bootstrap_counts(6, 5, random_seed=11)

    assert len(draws) == 5
    for drawn, row in zip(draws, counts):
        assert np.array_equal(row, np.bincount(drawn, minlength=6))


def test_a_cluster_drawn_twice_contributes_its_rows_twice():
    """
    Multiplicity is the whole difference between a bootstrap and a subsample.

    Gathering the drawn clusters' rows as a set -- the obvious way to write it --
    would silently turn every replicate into a resample *without* replacement, and
    the intervals would come out too narrow for a reason no output would reveal.
    """
    keys, row_groups = group_rows_by_cluster(["m1", "m1", "m2", "m1", "m3", "m3"])

    assert keys == ["m1", "m2", "m3"]
    assert [g.tolist() for g in row_groups] == [[0, 1, 3], [2], [4, 5]]

    rows = rows_for_clusters(row_groups, np.array([0, 2, 0]))
    assert rows.tolist() == [0, 1, 3, 4, 5, 0, 1, 3]
    assert len(rows) == 8, "a cluster drawn twice must be counted twice"


def test_clusters_are_grouped_in_first_appearance_order():
    """
    Mixed-type identifiers must not decide the order, or crash it.

    A mutant id column read back from CSV can arrive as a mix of strings and
    numbers; sorting the keys would raise on that comparison in Python 3, and
    ordering by anything data-dependent would make a seeded resample depend on
    the order rows happened to be written in.
    """
    keys, row_groups = group_rows_by_cluster(["b", 2, "a", 2, "b"])

    assert keys == ["b", 2, "a"]
    assert [g.tolist() for g in row_groups] == [[0, 4], [1, 3], [2]]


def test_an_empty_draw_gathers_no_rows():
    """A resample of nothing is empty, not an error -- the caller reports it as NaN."""
    _, row_groups = group_rows_by_cluster(["m1", "m2"])
    rows = rows_for_clusters(row_groups, np.empty(0, dtype=np.intp))

    assert rows.tolist() == []
    assert rows.dtype == np.intp


def test_no_resamples_is_a_request_that_cannot_be_honoured():
    """Zero bootstraps would yield an interval computed from nothing at all."""
    with pytest.raises(ValueError, match="num_bootstraps"):
        list(cluster_draws(10, 0))
