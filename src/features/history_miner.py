"""
ConfTest Historical Failure & Execution Dynamics Miner
Processes historical JUnit XML / pytest JSON logs to compute time-decayed failure frequency and flakiness.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **conftest.features.history_features** (extract_history_features_from_db()).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "conftest.features.history_features"

import math
from typing import Dict, List, Any

class HistoryMiner:
    """
    Computes time-decayed failure frequency, average test duration, and flakiness scores.
    """
    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate
        # test_id -> list of execution dicts: [{"outcome": "FAIL"/"PASS", "duration": float, "age_days": float}]
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}

    def record_run(self, test_id: str, outcome: str, duration: float, age_days: float = 0.0):
        """
        Appends an execution record to a test's history.
        """
        if test_id not in self.execution_history:
            self.execution_history[test_id] = []
        self.execution_history[test_id].append({
            "outcome": outcome.upper(),
            "duration": float(duration),
            "age_days": float(age_days)
        })

    def compute_time_decayed_failure_rate(self, test_id: str) -> float:
        """
        Computes FailRate(T_i) = sum( I[Result_k = Fail] * exp(-lambda * delta_t) ) / sum( exp(-lambda * delta_t) )
        """
        runs = self.execution_history.get(test_id, [])
        if not runs:
            return 0.05  # Default low prior for new tests

        weighted_failures = 0.0
        total_weights = 0.0

        for r in runs:
            weight = math.exp(-self.decay_rate * r["age_days"])
            total_weights += weight
            if r["outcome"] == "FAIL":
                weighted_failures += weight

        if total_weights == 0:
            return 0.0

        return float(min(1.0, weighted_failures / total_weights))

    def compute_average_duration(self, test_id: str) -> float:
        """
        Computes the mean execution time in seconds.
        """
        runs = self.execution_history.get(test_id, [])
        if not runs:
            return 1.0  # Default 1.0s estimate
        durations = [r["duration"] for r in runs]
        return float(sum(durations) / len(durations))

    def compute_flakiness_score(self, test_id: str) -> float:
        """
        Computes flakiness score based on frequent Pass -> Fail -> Pass flip rate.
        """
        runs = self.execution_history.get(test_id, [])
        if len(runs) < 3:
            return 0.0

        flips = 0
        for i in range(1, len(runs)):
            if runs[i]["outcome"] != runs[i-1]["outcome"]:
                flips += 1

        return float(min(1.0, flips / (len(runs) - 1)))
