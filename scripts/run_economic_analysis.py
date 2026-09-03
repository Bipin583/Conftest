"""
ConfTest Economic Cost-Benefit Analysis CLI.

Projects CI compute cost, developer wait cost, and escape cost under each RTS
strategy, from the strategies' MEASURED reduction and recall rates.

The rates used to be a hand-typed table in this file: Random-K at 0.75/0.28,
Changed-File at 0.68/0.78, Static Dependency Graph at 0.58/0.89, Uncalibrated ML
at 0.75/0.91, and ConfTest at 0.686 reduction / 0.995 recall with 0 escapes.
None of those came from a run. reports/baseline_comparison.csv puts the ConfTest
selector at 58.8% time reduction and 40.0% recall with 3 escaped commits, and the
Changed-File baseline at 0.0% recall rather than 78%. Those invented rates were
written into reports/economic_analysis.json, where they stopped looking invented,
and the dashboards then quoted the dollar figures.

Every dollar figure this script prints is a PROJECTION from assumed constants
(team size, hourly rate, suite duration, cost per escaped bug). What is measured
is the two rates it is applied to, and the output records which is which.

Usage:
    python scripts/run_economic_analysis.py --developers 25 --output reports/economic_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.economic_model import EnterpriseEconomicConfig, EconomicCostBenefitModel
from conftest.evaluation.headline import (
    BASELINE_COLS,
    BASELINE_CSV,
    STRATEGY_COL,
    MissingArtifact,
    as_number,
    provenance,
    read_baseline_rows,
)
from conftest.logging_config import get_logger

logger = get_logger(__name__)

INTERVALS_JSON = Path("reports/baseline_comparison_intervals.json")
SPLIT_METADATA = Path("data/splits/split_metadata.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Economic Cost-Benefit Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--developers", type=int, default=25, help="Number of active engineers.")
    parser.add_argument("--commits-per-dev", type=float, default=3.0, help="Commits per developer per day.")
    parser.add_argument("--suite-duration-mins", type=float, default=45.0, help="Full test suite duration in minutes.")
    parser.add_argument("--dev-hourly-rate", type=float, default=75.0, help="Developer hourly loaded cost in USD.")
    parser.add_argument("--bug-escape-cost", type=float, default=3500.0, help="Cost per production regression bug escape in USD.")
    parser.add_argument("--output", type=str, default="./reports/economic_analysis.json", help="Output path for JSON report.")
    return parser.parse_args()


def evaluated_commit_count(root: Path) -> Tuple[int, str]:
    """
    How many commits the escape counts in the benchmark table are out of.

    The CSV records escapes as a count, so scaling them to a year needs the
    denominator, and the CSV does not carry it. The intervals sidecar does,
    having been written by the same run; the split metadata is a fallback that
    assumes the benchmark ran on the test split, and that assumption is stated in
    the output rather than left implicit.
    """
    intervals = root / INTERVALS_JSON
    if intervals.exists():
        data = json.loads(intervals.read_text(encoding="utf-8"))
        count = data.get("num_commits")
        if count:
            return int(count), f"{INTERVALS_JSON.as_posix()} (same benchmark run)"

    metadata = root / SPLIT_METADATA
    if metadata.exists():
        data = json.loads(metadata.read_text(encoding="utf-8"))
        count = (data.get("test") or {}).get("num_commits")
        if count:
            return int(count), (
                f"{SPLIT_METADATA.as_posix()} test split, assuming the benchmark "
                "ran on it; re-run scripts/train_baseline.py with --bootstraps to "
                f"record the count in {INTERVALS_JSON.as_posix()} directly"
            )

    raise MissingArtifact(
        INTERVALS_JSON,
        "python scripts/train_baseline.py --bootstraps 2000 "
        "(needed to know how many commits the escape counts are out of)",
    )


def price_strategies(model, rows, evaluated_commits) -> list:
    """
    One priced row per strategy that carries a measured runtime saving.

    Only the compute and wait-time savings are priced. Escape cost is NOT, and
    that is deliberate: an earlier version scaled the benchmark's escaped-commit
    count to a year (3 of 5 commits x 3,750 = 11,250 escapes, $39M) and reported
    the result as a net benefit. Five commits cannot support an annual escape
    rate to three significant figures, and an injected mutant surviving a
    selection is not the same event as a regression reaching production.

    What is reported instead is the break-even count -- how many escapes a year
    would consume the savings -- next to the escape rate actually observed and
    the number of commits it was observed over. That comparison is the reader's
    to make, and it does not require inventing a scale factor.
    """
    priced = []
    for row in rows:
        name = str(row.get(STRATEGY_COL, "")).strip()
        etr = as_number(row.get(BASELINE_COLS["etr"]))
        recall = as_number(row.get(BASELINE_COLS["fr"]))
        escaped = as_number(row.get(BASELINE_COLS["escaped"])) or 0.0

        if etr is None:
            logger.warning(
                f"{name}: no measured time reduction in {BASELINE_CSV.as_posix()}, "
                "so there is no runtime saving to price. Skipped."
            )
            continue

        res = model.evaluate_rts_strategy(
            strategy_name=name,
            test_reduction_rate=etr / 100.0,
            failure_recall_rate=None if recall is None else recall / 100.0,
            annual_escapes_with_strategy=0,
        )
        # Computed with zero escape cost, so these two describe a saving before
        # any escape is paid for. Dropped rather than reported as a net figure.
        res.pop("net_annual_benefit_usd", None)
        res.pop("annual_escape_cost_usd", None)
        res.pop("strategy_total_cost_usd", None)

        res["measured_time_reduction_pct"] = etr
        res["measured_failure_recall_pct"] = recall
        res["observed_escaped_commits"] = int(escaped)
        res["evaluated_commits"] = evaluated_commits
        res["observed_escape_rate_per_commit"] = round(escaped / evaluated_commits, 4)
        res["escape_cost_priced"] = False
        res["escape_cost_note"] = (
            f"{int(escaped)} of {evaluated_commits} evaluated commits shipped an "
            "undetected failure. Not extrapolated to a year: the sample is too "
            "small to carry an annual rate, and a surviving injected mutant is "
            "not a production regression. Compare the observed rate against "
            "breakeven_escaped_bugs_threshold instead."
        )
        priced.append(res)
    return priced


def main():
    args = parse_args()

    config = EnterpriseEconomicConfig(
        num_developers=args.developers,
        commits_per_dev_per_day=args.commits_per_dev,
        full_suite_duration_minutes=args.suite_duration_mins,
        developer_hourly_rate_usd=args.dev_hourly_rate,
        cost_per_escaped_bug_usd=args.bug_escape_cost,
    )

    model = EconomicCostBenefitModel(config)
    baseline = model.calculate_full_suite_annual_cost()

    try:
        rows = read_baseline_rows(PROJECT_ROOT)
        evaluated_commits, commit_count_source = evaluated_commit_count(PROJECT_ROOT)
    except MissingArtifact as exc:
        logger.error(str(exc))
        sys.exit(1)

    annual_commits = model.total_annual_commits
    eval_results = price_strategies(model, rows, evaluated_commits)

    if not eval_results:
        logger.error(
            f"No strategy in {BASELINE_CSV.as_posix()} carries a measured time "
            "reduction. Nothing to price."
        )
        sys.exit(1)

    prov = provenance(PROJECT_ROOT)

    logger.info("\n" + "=" * 118)
    logger.info("  ConfTest Economic Projection (dollar figures are modelled, not measured)")
    logger.info(f"  Team Size: {config.num_developers} Devs | Annual Commits: {annual_commits:,} | Suite: {config.full_suite_duration_minutes:.0f}m")
    logger.info(f"  Rates measured over {evaluated_commits} commits; source: {commit_count_source}")
    logger.info("  Escape cost is not priced. Break-even is the number of yearly escapes that would consume the saving.")
    if not prov.real_labels:
        logger.warning("  Labels behind the measured rates were not produced by executing a test suite.")
    logger.info("=" * 118)
    logger.info(f"{'Strategy Name':<48} | {'ETR %':>6} | {'FR %':>6} | {'CI Savings':>12} | {'Dev Savings':>13} | {'Gross Saving':>13} | {'Breakeven':>9} | Escapes observed")
    logger.info("-" * 118)

    for item in eval_results:
        recall_text = "n/a" if item["failure_recall_rate_pct"] is None else f"{item['failure_recall_rate_pct']:.1f}"
        logger.info(
            f"{item['strategy_name']:<48} | "
            f"{item['test_reduction_rate_pct']:>6.1f} | "
            f"{recall_text:>6} | "
            f"${item['annual_ci_compute_savings_usd']:>11,.0f} | "
            f"${item['annual_developer_time_savings_usd']:>12,.0f} | "
            f"${item['gross_annual_savings_usd']:>12,.0f} | "
            f"{item['breakeven_escaped_bugs_threshold']:>9,} | "
            f"{item['observed_escaped_commits']}/{item['evaluated_commits']} commits"
        )
    logger.info("=" * 118)

    out_data = {
        "figures_are": (
            "projections from the assumed constants in 'enterprise_parameters', "
            "applied to the measured rates in 'measured_inputs'. Only the rates "
            "are measured; no dollar amount here was observed. Escape cost is not "
            "priced -- see escape_cost_note on each strategy."
        ),
        "measured_inputs": {
            "source": BASELINE_CSV.as_posix(),
            "columns_used": {
                "time_reduction": BASELINE_COLS["etr"],
                "failure_recall": BASELINE_COLS["fr"],
                "escaped_commits": BASELINE_COLS["escaped"],
            },
            "evaluated_commits": evaluated_commits,
            "evaluated_commits_source": commit_count_source,
            "escapes_extrapolated_to_a_year": False,
            "labels_measured": prov.real_labels,
            "label_provenance": prov.detail,
        },
        "enterprise_parameters": {
            "num_developers": config.num_developers,
            "commits_per_dev_per_day": config.commits_per_dev_per_day,
            "working_days_per_year": config.working_days_per_year,
            "total_annual_commits": annual_commits,
            "full_suite_duration_minutes": config.full_suite_duration_minutes,
            "ci_runner_cost_per_minute_usd": config.ci_runner_cost_per_minute_usd,
            "developer_hourly_rate_usd": config.developer_hourly_rate_usd,
            "cost_per_escaped_bug_usd": config.cost_per_escaped_bug_usd,
            "baseline_annual_bugs_escaped": config.baseline_annual_bugs_escaped,
            "developer_wait_fraction_of_ci_time": 0.30,
        },
        "baseline_full_suite_annual": baseline,
        "strategy_comparisons": eval_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    logger.info(f"\nEconomic projection saved to: {out_path}")


if __name__ == "__main__":
    main()
