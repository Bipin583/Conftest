"""Feature engineering for confidence-calibrated test selection.

Produces one feature vector per (commit, test) pair, grouped into the three
families the model consumes:

**Change features** describe the commit in isolation -- how big the diff is,
when it landed, what kind of files it touched. They are constant across every
test in the same commit, so on their own they can only raise or lower the whole
suite's risk.

**Test features** describe the test's own track record -- how often it has
failed recently, how long it takes, how flaky it looks. They are constant
across every commit for the same test.

**Cross features** are the ones that actually discriminate: they measure the
relationship between *this* change and *this* test. Path overlap, lexical
similarity between the changed file and the test function name, historical
co-change frequency, and static dependency distance.

All history features are computed **causally**: every window for a row at time
``t`` is built from observations strictly before ``t``. Nothing in this module
lets a test's own future outcome leak into its features, which is the usual way
a regression-test-selection benchmark accidentally reports 99% accuracy.

Example:
    >>> import pandas as pd
    >>> from data.features import build_features
    >>> raw = pd.read_csv("data/raw/test_history.csv")
    >>> features = build_features(raw)
    >>> features.to_csv("data/processed/features.csv", index=False)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, load_config, resolve_path  # noqa: E402

LOGGER = get_logger(__name__)

# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------

#: Commit-level features. Constant across tests within a commit.
CHANGE_FEATURES: List[str] = [
    "files_changed",
    "lines_added",
    "lines_removed",
    "total_churn",
    "authors_count",
    "commit_message_length",
    "num_file_extensions",
    "has_source_change",
    "has_test_change",
    "has_config_change",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

#: Test-level features. Constant across commits for a given test.
TEST_FEATURES: List[str] = [
    "failure_rate_7d",
    "failure_rate_14d",
    "failure_rate_28d",
    "failure_rate_lifetime",
    "execution_count",
    "duration_mean",
    "duration_std",
    "recent_failures",
    "runs_since_last_failure",
    "test_name_length",
    "flakiness_score",
]

#: Change-by-test interaction features. These carry the discriminative signal.
CROSS_FEATURES: List[str] = [
    "file_test_similarity",
    "co_change_frequency",
    "dependency_distance",
    "lexical_similarity",
    "path_overlap",
    "same_module",
]

#: Full ordered feature list consumed by the model (30 features).
ALL_FEATURES: List[str] = CHANGE_FEATURES + TEST_FEATURES + CROSS_FEATURES

#: Identifier columns carried through the pipeline but never fed to the model.
ID_COLUMNS: List[str] = ["repo", "commit_sha", "test_id", "timestamp"]

#: Binary target column.
LABEL_COLUMN = "label_failed"

#: Tokens that appear in nearly every test identifier and carry no signal.
_STOPWORD_TOKENS: Set[str] = {"test", "tests", "src", "lib", "py", "init", "main", "a", "the"}

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


class FeatureError(RuntimeError):
    """Raised when the input frame lacks columns the extractors require."""


# --------------------------------------------------------------------------
# Tokenisation helpers
# --------------------------------------------------------------------------


def tokenize_identifier(text: str) -> Set[str]:
    """Split a path or identifier into lowercase content tokens.

    Handles ``snake_case``, ``camelCase``, ``kebab-case`` and path separators,
    then drops single characters and boilerplate tokens such as ``test``.

    Args:
        text: A file path, module name, class name, or function name.

    Returns:
        The set of content tokens. Empty for empty or all-boilerplate input.

    Example:
        >>> sorted(tokenize_identifier("tests/test_json_parser.py"))
        ['json', 'parser']
    """
    if not isinstance(text, str) or not text:
        return set()

    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    tokens = {tok.lower() for tok in _NON_ALNUM.split(spaced) if tok}
    return {tok for tok in tokens if len(tok) > 1 and tok not in _STOPWORD_TOKENS}


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Jaccard similarity between two token sets.

    Args:
        left: First token set.
        right: Second token set.

    Returns:
        ``|A n B| / |A u B|``, or ``0.0`` when either set is empty.
    """
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def dice(left: Set[str], right: Set[str]) -> float:
    """Sorensen-Dice coefficient between two token sets.

    Less punitive than Jaccard when the sets differ greatly in size, which is
    the common case comparing a long file path against a short function name.

    Args:
        left: First token set.
        right: Second token set.

    Returns:
        ``2|A n B| / (|A| + |B|)``, or ``0.0`` when either set is empty.
    """
    if not left or not right:
        return 0.0
    total = len(left) + len(right)
    return 2.0 * len(left & right) / total if total else 0.0


def path_overlap_ratio(left_path: str, right_path: str) -> float:
    """Fraction of leading directory components two paths share.

    Args:
        left_path: First path, POSIX or Windows separators.
        right_path: Second path.

    Returns:
        Shared prefix depth divided by the deeper path's directory depth, in
        ``[0, 1]``. Two files in the same directory score ``1.0``.

    Example:
        >>> round(path_overlap_ratio("src/api/views.py", "src/api/test_views.py"), 3)
        1.0
    """
    if not isinstance(left_path, str) or not isinstance(right_path, str):
        return 0.0
    left = [p for p in re.split(r"[/\\]", left_path)[:-1] if p]
    right = [p for p in re.split(r"[/\\]", right_path)[:-1] if p]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared / max(len(left), len(right))


# --------------------------------------------------------------------------
# Change features
# --------------------------------------------------------------------------


def extract_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute commit-level features.

    Derives whatever is available from the input columns and fills the rest
    with neutral defaults, so the function works on both a full GitHub harvest
    and a leaner mined corpus.

    Args:
        frame: Rows carrying at least ``commit_sha``. Optional inputs include
            ``changed_file_path``, ``lines_added``, ``lines_removed``,
            ``commit_message``, ``author`` and ``timestamp``.

    Returns:
        A frame indexed like ``frame`` with the :data:`CHANGE_FEATURES` columns.
    """
    out = pd.DataFrame(index=frame.index)

    changed_paths = frame["changed_file_path"] if "changed_file_path" in frame else pd.Series("", index=frame.index)
    changed_paths = changed_paths.fillna("").astype(str)

    if "files_changed" in frame:
        out["files_changed"] = pd.to_numeric(frame["files_changed"], errors="coerce")
    else:
        per_commit = frame.assign(_p=changed_paths).groupby("commit_sha")["_p"].transform("nunique")
        out["files_changed"] = per_commit

    for target, candidates in (
        ("lines_added", ("lines_added", "diff_lines_added")),
        ("lines_removed", ("lines_removed", "diff_lines_deleted")),
    ):
        source = next((c for c in candidates if c in frame), None)
        out[target] = pd.to_numeric(frame[source], errors="coerce") if source else 0.0

    out["total_churn"] = out["lines_added"].fillna(0) + out["lines_removed"].fillna(0)

    if "author" in frame:
        out["authors_count"] = frame.groupby("commit_sha")["author"].transform("nunique")
    else:
        out["authors_count"] = 1.0

    if "commit_message" in frame:
        message = frame["commit_message"].fillna("").astype(str)
        out["commit_message_length"] = message.str.len()
    elif "diff_msg_length" in frame:
        out["commit_message_length"] = pd.to_numeric(frame["diff_msg_length"], errors="coerce")
    else:
        out["commit_message_length"] = 0.0

    extensions = changed_paths.str.rsplit(".", n=1).str[-1].str.lower()
    out["num_file_extensions"] = (
        frame.assign(_e=extensions).groupby("commit_sha")["_e"].transform("nunique").astype(float)
    )
    out["has_source_change"] = (~changed_paths.str.contains("test", case=False, na=False)).astype(float)
    out["has_test_change"] = changed_paths.str.contains("test", case=False, na=False).astype(float)
    out["has_config_change"] = changed_paths.str.contains(
        r"\.(?:ya?ml|toml|cfg|ini|json)$", case=False, regex=True, na=False
    ).astype(float)

    stamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True) if "timestamp" in frame else pd.Series(pd.NaT, index=frame.index)
    out["hour_of_day"] = stamps.dt.hour.astype(float)
    out["day_of_week"] = stamps.dt.dayofweek.astype(float)
    out["is_weekend"] = (stamps.dt.dayofweek >= 5).astype(float)

    return out


# --------------------------------------------------------------------------
# Test history features
# --------------------------------------------------------------------------


def extract_test_features(
    frame: pd.DataFrame,
    windows: Sequence[int] = (7, 14, 28),
    recent_k: int = 10,
) -> pd.DataFrame:
    """Compute per-test history features using only strictly prior observations.

    For each test, rows are ordered by time and every statistic is shifted by
    one position before being consumed, so a row never sees its own outcome.
    Window statistics use a time-based rolling window closed on the left.

    Args:
        frame: Rows with ``test_id``, ``timestamp`` and the label column.
            ``duration`` is optional.
        windows: Trailing window widths in days.
        recent_k: Look-back length for ``recent_failures`` and flakiness.

    Returns:
        A frame indexed like ``frame`` with the :data:`TEST_FEATURES` columns.

    Raises:
        FeatureError: If ``test_id`` or the label column is missing.
    """
    for required in ("test_id", LABEL_COLUMN):
        if required not in frame:
            raise FeatureError(f"extract_test_features requires a '{required}' column.")

    work = frame.loc[:, ["test_id", LABEL_COLUMN]].copy()
    work["_label"] = pd.to_numeric(work[LABEL_COLUMN], errors="coerce").fillna(0.0)
    work["_duration"] = (
        pd.to_numeric(frame["duration"], errors="coerce").fillna(0.0)
        if "duration" in frame
        else 0.0
    )
    work["_ts"] = (
        pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if "timestamp" in frame
        else pd.NaT
    )
    # Rows without a usable timestamp still need a stable order for the causal
    # shift; fall back to positional order within the test.
    if work["_ts"].isna().all():
        work["_ts"] = pd.to_datetime(work.groupby("test_id").cumcount(), unit="s", utc=True)

    work = work.sort_values(["test_id", "_ts"], kind="mergesort")
    grouped = work.groupby("test_id", sort=False)

    out = pd.DataFrame(index=work.index)

    # Expanding lifetime statistics, shifted so row i sees rows [0, i).
    prior_runs = grouped.cumcount()
    prior_failures = grouped["_label"].cumsum() - work["_label"]
    out["execution_count"] = prior_runs.astype(float)
    out["failure_rate_lifetime"] = np.where(prior_runs > 0, prior_failures / prior_runs.replace(0, np.nan), 0.0)

    out["duration_mean"] = grouped["_duration"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    out["duration_std"] = grouped["_duration"].apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)

    # Time-based trailing windows. The window is anchored on the current row's
    # timestamp, so the current row's own label must be subtracted back out --
    # shifting the label *before* rolling would instead drag the previous run's
    # outcome into the window regardless of how long ago it happened.
    windowed = pd.DataFrame(
        {"_ts": work["_ts"], "_label": work["_label"], "test_id": work["test_id"]}
    ).set_index("_ts")
    own_label = work["_label"].to_numpy(dtype=float)
    for days in windows:
        roller = windowed.groupby("test_id", sort=False)["_label"].rolling(f"{days}D", min_periods=1)
        window_sum = roller.sum().reset_index(level=0, drop=True).to_numpy(dtype=float)
        window_count = roller.count().reset_index(level=0, drop=True).to_numpy(dtype=float)
        prior_sum = window_sum - own_label
        prior_count = window_count - 1.0
        out[f"failure_rate_{days}d"] = np.divide(
            prior_sum,
            prior_count,
            out=np.zeros(len(work), dtype=float),
            where=prior_count > 0,
        )

    out["recent_failures"] = (
        grouped["_label"].apply(lambda s: s.shift(1).rolling(recent_k, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

    # Runs since the last observed failure; large when a test is reliably green.
    def _runs_since_failure(series: pd.Series) -> pd.Series:
        counter = 0.0
        values: List[float] = []
        for observed in series.to_numpy():
            values.append(counter)
            counter = 0.0 if observed > 0 else counter + 1.0
        return pd.Series(values, index=series.index)

    out["runs_since_last_failure"] = (
        grouped["_label"].apply(_runs_since_failure).reset_index(level=0, drop=True)
    )

    # Flakiness: how often the outcome flipped between consecutive prior runs.
    flips = grouped["_label"].apply(lambda s: s.diff().abs().shift(1).rolling(recent_k, min_periods=1).mean())
    out["flakiness_score"] = flips.reset_index(level=0, drop=True)

    name_source = frame["test_function"] if "test_function" in frame else frame["test_id"]
    out["test_name_length"] = name_source.fillna("").astype(str).str.len().reindex(work.index)

    # A test with no prior observations has a well-defined count of zero prior
    # failures. That is genuinely different from an unknown mean duration,
    # which stays NaN so preprocessing imputes it rather than asserting 0s.
    for column in ("recent_failures", "flakiness_score", "runs_since_last_failure"):
        out[column] = out[column].fillna(0.0)

    return out.reindex(frame.index)


# --------------------------------------------------------------------------
# Cross features
# --------------------------------------------------------------------------


def extract_cross_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute change-by-test interaction features.

    Args:
        frame: Rows with ``changed_file_path`` and ``test_path``. Optional
            inputs are ``test_function``, ``test_id``, the label column, and
            ``dep_shortest_path_depth`` from a static call-graph pass.

    Returns:
        A frame indexed like ``frame`` with the :data:`CROSS_FEATURES` columns.
    """
    out = pd.DataFrame(index=frame.index)

    changed = (frame["changed_file_path"] if "changed_file_path" in frame else pd.Series("", index=frame.index)).fillna("").astype(str)
    test_path = (frame["test_path"] if "test_path" in frame else pd.Series("", index=frame.index)).fillna("").astype(str)
    test_func = (frame["test_function"] if "test_function" in frame else pd.Series("", index=frame.index)).fillna("").astype(str)

    # Cache tokenisation: a corpus has far fewer distinct paths than rows.
    token_cache: Dict[str, Set[str]] = {}

    def tokens_of(value: str) -> Set[str]:
        cached = token_cache.get(value)
        if cached is None:
            cached = tokenize_identifier(value)
            token_cache[value] = cached
        return cached

    changed_tokens = [tokens_of(v) for v in changed]
    path_tokens = [tokens_of(v) for v in test_path]
    func_tokens = [tokens_of(v) for v in test_func]

    out["file_test_similarity"] = [dice(c, p) for c, p in zip(changed_tokens, path_tokens)]
    out["lexical_similarity"] = [jaccard(c, f) for c, f in zip(changed_tokens, func_tokens)]

    pair_cache: Dict[tuple, float] = {}
    overlaps: List[float] = []
    for left, right in zip(changed, test_path):
        key = (left, right)
        value = pair_cache.get(key)
        if value is None:
            value = path_overlap_ratio(left, right)
            pair_cache[key] = value
        overlaps.append(value)
    out["path_overlap"] = overlaps

    changed_module = changed.str.rsplit("/", n=1).str[0].str.rsplit("\\", n=1).str[0]
    test_module = test_path.str.rsplit("/", n=1).str[0].str.rsplit("\\", n=1).str[0]
    out["same_module"] = (changed_module == test_module).astype(float)

    if "dep_shortest_path_depth" in frame:
        depth = pd.to_numeric(frame["dep_shortest_path_depth"], errors="coerce")
        # Unreachable pairs arrive as 0 or NaN from the graph pass; treat both
        # as "far away" rather than "adjacent", which is the opposite meaning.
        reachable = frame["dep_is_reachable"] if "dep_is_reachable" in frame else (depth > 0)
        out["dependency_distance"] = depth.where(reachable.astype(bool), other=99.0).fillna(99.0)
    else:
        # Fall back to a path-distance proxy: shared-prefix depth inverted.
        out["dependency_distance"] = 1.0 - np.asarray(overlaps, dtype=float)

    out["co_change_frequency"] = _co_change_frequency(frame, changed, test_path)
    return out


def _co_change_frequency(
    frame: pd.DataFrame,
    changed: pd.Series,
    test_path: pd.Series,
) -> pd.Series:
    """Historical failure rate of a test given this file changed.

    Computed causally over the corpus ordering: the value for row ``i`` is the
    failure rate over prior rows sharing the same (changed file, test) pair.

    Args:
        frame: Source rows, used for the label and ordering.
        changed: Cleaned changed-file paths.
        test_path: Cleaned test paths.

    Returns:
        A float series aligned to ``frame.index``, ``0.0`` where no prior
        observation of the pair exists.
    """
    if LABEL_COLUMN not in frame:
        return pd.Series(0.0, index=frame.index)

    pair = changed.str.cat(test_path, sep="\x00")
    label = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").fillna(0.0)

    order = (
        pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).argsort(kind="mergesort")
        if "timestamp" in frame
        else np.arange(len(frame))
    )
    ordered_index = frame.index[order]

    pair_ordered = pair.reindex(ordered_index)
    label_ordered = label.reindex(ordered_index)

    grouped = label_ordered.groupby(pair_ordered, sort=False)
    prior_count = grouped.cumcount()
    prior_sum = grouped.cumsum() - label_ordered
    rate = np.where(prior_count > 0, prior_sum / prior_count.replace(0, np.nan), 0.0)

    return pd.Series(rate, index=ordered_index).reindex(frame.index).fillna(0.0)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_features(frame: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Run all three extractors and assemble the model-ready feature frame.

    Args:
        frame: Raw (commit, test) observations.
        show_progress: Display a tqdm bar over the extraction stages.

    Returns:
        A frame with :data:`ID_COLUMNS`, :data:`ALL_FEATURES` and the label.

    Raises:
        FeatureError: If the label column is absent.
    """
    if LABEL_COLUMN not in frame:
        raise FeatureError(f"Input frame must carry the '{LABEL_COLUMN}' column.")

    stages = (
        ("change", extract_change_features),
        ("test", extract_test_features),
        ("cross", extract_cross_features),
    )
    iterator = tqdm(stages, desc="Extracting features", unit="stage", disable=not show_progress)

    parts: List[pd.DataFrame] = []
    for name, extractor in iterator:
        try:
            parts.append(extractor(frame))
        except FeatureError:
            raise
        except Exception as exc:
            raise FeatureError(f"The '{name}' extractor failed: {exc}") from exc

    features = pd.concat(parts, axis=1)

    # Guarantee the declared schema exists and is ordered, even if an optional
    # input column was missing upstream.
    for column in ALL_FEATURES:
        if column not in features:
            LOGGER.warning("Feature '%s' could not be derived; filling with 0.0.", column)
            features[column] = 0.0
    features = features.loc[:, ALL_FEATURES].astype(float)

    carried = [c for c in ID_COLUMNS if c in frame]
    assembled = pd.concat([frame.loc[:, carried], features, frame[[LABEL_COLUMN]]], axis=1)

    LOGGER.info(
        "Built %d features for %d rows (%d change, %d test, %d cross).",
        len(ALL_FEATURES),
        len(assembled),
        len(CHANGE_FEATURES),
        len(TEST_FEATURES),
        len(CROSS_FEATURES),
    )
    return assembled


def main(argv: Optional[List[str]] = None) -> int:
    """Build ``data/processed/features.csv`` from ``data/raw/test_history.csv``.

    Returns:
        ``0`` on success, ``1`` if the raw harvest is missing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build the feature matrix.")
    parser.add_argument("--input", default=None, help="Raw history CSV.")
    parser.add_argument("--output", default=None, help="Destination features CSV.")
    args = parser.parse_args(argv)

    cfg = load_config()
    source = resolve_path(args.input or Path(cfg["data"]["raw_dir"]) / "test_history.csv")
    target = resolve_path(args.output or Path(cfg["data"]["processed_dir"]) / "features.csv")

    if not source.is_file():
        LOGGER.error("Raw history not found: %s. Run 'python cli.py collect' first.", source)
        return 1

    features = build_features(pd.read_csv(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(target, index=False)
    LOGGER.info("Wrote %s (%d rows).", target, len(features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
