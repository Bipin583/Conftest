"""
ConfTest Economic Cost-Benefit & CI/CD Financial Modeling Engine.

Quantifies financial ROI, developer wait-time productivity gains, cloud compute savings,
and regression escape risk economics across enterprise CI/CD workflows.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np

from conftest.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EnterpriseEconomicConfig:
    """Enterprise CI/CD and engineering financial parameters."""
    num_developers: int = 25
    commits_per_dev_per_day: float = 3.0
    working_days_per_year: int = 250
    developer_hourly_rate_usd: float = 75.00
    ci_runner_cost_per_minute_usd: float = 0.016  # Standard GitHub Actions Linux 4-core
    full_suite_duration_minutes: float = 45.0
    cost_per_escaped_bug_usd: float = 3500.00  # Industry average triage + patch cost
    baseline_annual_bugs_escaped: int = 4


class EconomicCostBenefitModel:
    """
    Computes economic savings, productivity gains, and risk-adjusted ROI of ConfTest.
    """

    def __init__(self, config: Optional[EnterpriseEconomicConfig] = None):
        self.config = config or EnterpriseEconomicConfig()

    @property
    def total_annual_commits(self) -> int:
        """Total commits processed through CI per year."""
        return int(
            self.config.num_developers
            * self.config.commits_per_dev_per_day
            * self.config.working_days_per_year
        )

    def calculate_full_suite_annual_cost(self) -> Dict[str, float]:
        """Calculate total baseline annual expenditure running full test suites on every commit."""
        annual_commits = self.total_annual_commits
        suite_mins = self.config.full_suite_duration_minutes

        total_ci_minutes = annual_commits * suite_mins
        annual_ci_cost = total_ci_minutes * self.config.ci_runner_cost_per_minute_usd

        # Developer blocked wait-time (assuming developer waits 30% of CI time before context switching)
        wait_hours = (total_ci_minutes * 0.30) / 60.0
        annual_developer_wait_cost = wait_hours * self.config.developer_hourly_rate_usd

        # Baseline escape cost
        annual_escape_cost = self.config.baseline_annual_bugs_escaped * self.config.cost_per_escaped_bug_usd

        total_annual_cost = annual_ci_cost + annual_developer_wait_cost + annual_escape_cost

        return {
            "total_annual_commits": annual_commits,
            "total_ci_minutes": round(total_ci_minutes, 1),
            "annual_ci_compute_cost_usd": round(annual_ci_cost, 2),
            "annual_developer_wait_cost_usd": round(annual_developer_wait_cost, 2),
            "annual_escape_cost_usd": round(annual_escape_cost, 2),
            "total_annual_cost_usd": round(total_annual_cost, 2),
        }

    def evaluate_rts_strategy(
        self,
        strategy_name: str,
        test_reduction_rate: float,
        failure_recall_rate: Optional[float],
        annual_escapes_with_strategy: int = 0,
    ) -> Dict[str, Any]:
        """
        Project net financial benefit and ROI of a specific RTS strategy.

        Every figure returned is a projection from the config's assumed constants,
        not a measurement. Only the two rates are measured, and only if the caller
        read them from the benchmark table.

        Args:
            strategy_name: Name of the strategy (e.g. 'ConfTest (Calibrated + Selective)').
            test_reduction_rate: Fraction of test WALL-CLOCK saved (0.0 to 1.0) --
                the benchmark's ETR, since the arithmetic below multiplies suite
                minutes by it. Passing the test-COUNT reduction (TRR) instead
                overstates the saving by however much the durations are skewed.
            failure_recall_rate: Fraction of regression failures caught (0.0 to 1.0),
                or None where the split offered no failure to recall. Reported, not
                used in the arithmetic.
            annual_escapes_with_strategy: Expected production escapes under this strategy.

        Returns:
            Dictionary containing projected savings, escape risk, and net ROI.
        """
        baseline = self.calculate_full_suite_annual_cost()
        annual_commits = self.total_annual_commits
        orig_suite_mins = self.config.full_suite_duration_minutes

        # Reduced test suite runtime
        reduced_mins_per_run = orig_suite_mins * (1.0 - test_reduction_rate)
        total_ci_minutes = annual_commits * reduced_mins_per_run

        ci_cost = total_ci_minutes * self.config.ci_runner_cost_per_minute_usd
        wait_hours = (total_ci_minutes * 0.30) / 60.0
        dev_wait_cost = wait_hours * self.config.developer_hourly_rate_usd

        escape_cost = annual_escapes_with_strategy * self.config.cost_per_escaped_bug_usd
        strategy_total_cost = ci_cost + dev_wait_cost + escape_cost

        # Savings
        ci_compute_savings = baseline["annual_ci_compute_cost_usd"] - ci_cost
        dev_time_savings = baseline["annual_developer_wait_cost_usd"] - dev_wait_cost
        gross_savings = ci_compute_savings + dev_time_savings
        net_financial_benefit = baseline["total_annual_cost_usd"] - strategy_total_cost

        # Break-even escape threshold: how many bugs can escape before savings are wiped out?
        breakeven_escapes = int(gross_savings / self.config.cost_per_escaped_bug_usd)

        return {
            "strategy_name": strategy_name,
            "test_reduction_rate_pct": round(test_reduction_rate * 100, 1),
            # None, not 0.0: a split with no available failure has no recall, and
            # 0.0% would read as a strategy that caught nothing.
            "failure_recall_rate_pct": (
                None if failure_recall_rate is None else round(failure_recall_rate * 100, 1)
            ),
            "annual_ci_compute_cost_usd": round(ci_cost, 2),
            "annual_developer_wait_cost_usd": round(dev_wait_cost, 2),
            "annual_escape_cost_usd": round(escape_cost, 2),
            "strategy_total_cost_usd": round(strategy_total_cost, 2),
            "annual_ci_compute_savings_usd": round(ci_compute_savings, 2),
            "annual_developer_time_savings_usd": round(dev_time_savings, 2),
            "gross_annual_savings_usd": round(gross_savings, 2),
            "net_annual_benefit_usd": round(net_financial_benefit, 2),
            "breakeven_escaped_bugs_threshold": breakeven_escapes,
        }
