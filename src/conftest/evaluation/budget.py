"""
Failure recall under a per-commit test budget.

One implementation, imported by every evaluation path that needs the number, because the
tempting shortcut -- rank the whole dataset and take the top quarter -- scores differently
from the policy a CI system can actually run, and a project that computes it twice ends up
publishing both.
"""

from typing import Optional, Tuple

import numpy as np


def recall_at_budget_per_commit(
    y: np.ndarray,
    scores: np.ndarray,
    groups: Optional[np.ndarray],
    budget_ratio: float,
) -> Tuple[float, str]:
    """
    Failure recall when each commit may run `budget_ratio` of its own test universe.

    Ranking every row in one list and taking the top quarter is not a selection policy: at
    prediction time the only choice available is which of *this* commit's tests to run.
    A single global ranking lets one commit borrow budget from another, which no CI system
    can do, and it scores better, because the easy commits absorb the budget.

    Args:
        y: Binary labels, 1 where the test failed.
        scores: Predicted failure scores, higher meaning more suspicious.
        groups: Commit id per row. None means the caller has no commit structure.
        budget_ratio: Fraction of each commit's tests that may run.

    Returns:
        (recall pooled over commits, a note naming the denominator). NaN when no commit
        carries a failure, or when no commit ids were supplied: both are undefined, and
        0.0 would read as a measured miss.
    """
    if groups is None:
        return (
            float("nan"),
            "undefined: no commit ids were supplied, and a dataset-wide ranking is not a "
            "selection policy",
        )

    groups = np.asarray(groups).ravel()
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()
    detected = 0
    available = 0
    commits_with_failures = 0
    for commit in np.unique(groups):
        rows = np.flatnonzero(groups == commit)
        failures = int(y[rows].sum())
        if failures == 0:
            continue
        commits_with_failures += 1
        k = max(1, int(round(len(rows) * budget_ratio)))
        order = np.argsort(scores[rows])[::-1][:k]
        detected += int(y[rows][order].sum())
        available += failures

    if available == 0:
        return float("nan"), "undefined: no commit in this dataset had a failing test"
    note = (
        f"{detected} of {available} failing (commit, test) pairs, pooled over the "
        f"{commits_with_failures} commits that had at least one failure; each commit ran "
        f"the top {budget_ratio:.0%} of its own universe"
    )
    return detected / available, note
