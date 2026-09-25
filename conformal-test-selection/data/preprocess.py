"""Preprocessing: imputation, scaling, encoding, and the temporal split.

Two entry paths feed the same output contract:

1. **Harvest path** -- ``data/processed/features.csv`` produced by
   :mod:`data.features` from a GitHub Actions harvest. This path performs its
   own chronological 70/15/15 split, grouped by commit.

2. **Reuse path** -- an external corpus that already carries a strict
   chronological split (``data.external_splits_dir``). The adapter maps that
   corpus onto this project's feature schema and recomputes the six cross
   features from raw paths, which the source corpus does not provide.

Both paths write ``train.csv``, ``val.csv`` and ``test.csv`` into
``data/processed/`` plus a fitted ``models/preprocessor.pkl``.

**Splitting is grouped by commit, never by row.** Two rows from the same commit
share every change feature, so letting a commit straddle the train/test boundary
leaks the change signal and inflates held-out scores.

Example:
    >>> from data.preprocess import preprocess
    >>> summary = preprocess()
    >>> summary["test"]["n_rows"]
    86469
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, load_config, resolve_path, save_artifact  # noqa: E402
from data.features import (  # noqa: E402
    ALL_FEATURES,
    CROSS_FEATURES,
    ID_COLUMNS,
    LABEL_COLUMN,
    extract_cross_features,
)

LOGGER = get_logger(__name__)

#: Maps this project's schema names onto the column names used by the mined
#: ConfTest corpus. Only semantically equivalent pairs are aliased.
EXTERNAL_ALIASES: Dict[str, str] = {
    "lines_added": "diff_lines_added",
    "lines_removed": "diff_lines_deleted",
    "total_churn": "diff_total_churn",
    "files_changed": "diff_num_files_changed",
    "commit_message_length": "diff_msg_length",
    "has_test_change": "diff_num_test_files",
    "has_source_change": "diff_num_src_files",
    "has_config_change": "diff_has_config",
    "execution_count": "hist_total_prior_runs",
    "failure_rate_lifetime": "hist_lifetime_failure_rate",
    "recent_failures": "hist_recent_10_failure_rate",
    "duration_mean": "hist_avg_duration",
    "flakiness_score": "hist_flaky_score",
    "test_name_length": "ast_test_func_name_length",
}

#: Extra columns the external corpus supplies that have no schema alias but do
#: carry signal. They are appended to the model's feature list verbatim.
EXTERNAL_EXTRA_FEATURES: List[str] = [
    "diff_num_src_files",
    "diff_num_test_files",
    "diff_has_python",
    "diff_is_fix_commit",
    "diff_is_refactor_commit",
    "diff_msg_word_count",
    "ast_test_file_functions_count",
    "ast_test_file_classes_count",
    "ast_test_file_imports_count",
    "ast_test_file_complexity",
    "ast_test_is_parameterized",
    "dep_is_direct_import",
    "dep_name_heuristic_coupled",
    "dep_is_reachable",
    "dep_max_reverse_dependencies",
    "dep_test_total_out_degree",
    "hist_prior_failures",
    "hist_has_ever_failed",
    "hist_changed_files_prior_mod_count",
]

#: Schema features the external corpus cannot support, with the reason. The
#: mined corpus timestamps are synthetic mutation-ordering stamps spanning a
#: few minutes, so calendar-window and clock features are undefined on it.
EXTERNAL_UNAVAILABLE: Dict[str, str] = {
    "failure_rate_7d": "corpus timestamps are synthetic; calendar windows are degenerate",
    "failure_rate_14d": "corpus timestamps are synthetic; calendar windows are degenerate",
    "failure_rate_28d": "corpus timestamps are synthetic; calendar windows are degenerate",
    "hour_of_day": "corpus has no real wall-clock commit time",
    "day_of_week": "corpus has no real wall-clock commit time",
    "is_weekend": "corpus has no real wall-clock commit time",
    "duration_std": "corpus records mean duration only",
    "authors_count": "corpus does not record commit authorship",
    "num_file_extensions": "corpus records one changed file per row",
    "runs_since_last_failure": "not materialised by the source pipeline",
}

#: Low-cardinality string columns that are one-hot encoded.
CATEGORICAL_CANDIDATES: List[str] = ["repo", "mutant_operator"]


class PreprocessError(RuntimeError):
    """Raised when input data is missing, empty, or lacks required columns."""


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def temporal_group_split(
    frame: pd.DataFrame,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    group_column: str = "commit_sha",
    time_column: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically, keeping every row of a commit on one side.

    Commits are ordered by their earliest timestamp and cut at the requested
    ratios, so the validation and test sets sit strictly in the future of the
    training set -- the only split that reflects how the model is deployed.

    Args:
        frame: Feature rows.
        ratios: ``(train, val, test)`` proportions of *commits*, not rows.
        group_column: Column identifying a commit.
        time_column: Timestamp column used for ordering.

    Returns:
        A ``(train, val, test)`` tuple of frames.

    Raises:
        PreprocessError: If the grouping column is absent or the frame is empty.
    """
    if frame.empty:
        raise PreprocessError("Cannot split an empty frame.")
    if group_column not in frame:
        raise PreprocessError(f"Split requires a '{group_column}' column.")

    if time_column in frame:
        stamps = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        order = stamps.groupby(frame[group_column]).min().sort_values()
    else:
        LOGGER.warning("No '%s' column; falling back to first-appearance order.", time_column)
        order = pd.Series(range(frame[group_column].nunique()), index=frame[group_column].unique())

    commits = list(order.index)
    total = len(commits)
    if total < 3:
        raise PreprocessError(f"Need at least 3 commits to split, found {total}.")

    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    buckets = {
        "train": set(commits[:train_end]),
        "val": set(commits[train_end:val_end]),
        "test": set(commits[val_end:]),
    }

    parts = tuple(frame[frame[group_column].isin(buckets[name])].copy() for name in ("train", "val", "test"))
    LOGGER.info(
        "Temporal split: %d/%d/%d commits -> %d/%d/%d rows.",
        len(buckets["train"]), len(buckets["val"]), len(buckets["test"]),
        len(parts[0]), len(parts[1]), len(parts[2]),
    )
    return parts


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """Assemble the impute -> scale / impute -> encode column transformer.

    Numeric columns take a median imputer (robust to the heavy-tailed churn and
    duration distributions) followed by standardisation. Categorical columns
    take a most-frequent imputer followed by one-hot encoding that ignores
    categories unseen at fit time, so a new repository does not crash inference.

    Args:
        numeric_columns: Continuous and binary feature names.
        categorical_columns: Low-cardinality string feature names.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    # `sparse_output` replaced `sparse` in scikit-learn 1.2; this project
    # supports 1.3 through 1.9, so the newer spelling is used unconditionally.
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
        ]
    )

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))
    if not transformers:
        raise PreprocessError("No columns available to preprocess.")

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _sanitize_feature_names(names: Sequence[str]) -> List[str]:
    """Make generated column names safe for XGBoost, preserving uniqueness.

    One-hot encoding a category such as ``cmp_<=_to_>=`` yields a column name
    containing ``<`` and ``>``, which XGBoost rejects outright because those
    characters are how it serialises split conditions. Renaming happens here,
    once, so training and serving always agree on the feature list.

    Args:
        names: Raw names emitted by the fitted transformer.

    Returns:
        Sanitised names in the same order. Collisions introduced by the
        substitution are broken with a numeric suffix.
    """
    replacements = {"<": "_lt_", ">": "_gt_", "[": "_", "]": "_", ",": "_", " ": "_"}
    seen: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        clean = str(name)
        for bad, good in replacements.items():
            clean = clean.replace(bad, good)
        clean = re.sub(r"_+", "_", clean).strip("_") or "feature"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        out.append(clean)
    return out


def _transformed_frame(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    carry_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply a fitted preprocessor and reattach identifiers and the label.

    Args:
        preprocessor: Fitted transformer.
        frame: Rows to transform.
        feature_columns: Columns the transformer expects.
        carry_columns: Identifier columns to preserve alongside the features.

    Returns:
        A frame of transformed features plus carried columns and the label.
    """
    matrix = preprocessor.transform(frame.loc[:, list(feature_columns)])
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - very old sklearn
        names = [f"f{i}" for i in range(matrix.shape[1])]

    out = pd.DataFrame(matrix, columns=_sanitize_feature_names(names), index=frame.index).astype(np.float32)
    for column in carry_columns:
        if column in frame:
            out.insert(0, column, frame[column].to_numpy())
    out[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    return out


# --------------------------------------------------------------------------
# External corpus adapter
# --------------------------------------------------------------------------


def adapt_external_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Map a mined corpus onto this project's feature schema.

    Aliased columns are renamed, the six cross features are recomputed from the
    raw ``changed_file_path``/``test_path`` pair, and corpus-specific columns
    listed in :data:`EXTERNAL_EXTRA_FEATURES` are carried through.

    Args:
        frame: Raw external corpus rows.

    Returns:
        A ``(adapted_frame, feature_columns)`` pair.

    Raises:
        PreprocessError: If the label column is absent.
    """
    if LABEL_COLUMN not in frame:
        raise PreprocessError(f"External corpus lacks the '{LABEL_COLUMN}' column.")

    adapted = frame.copy()
    if "commit_timestamp" in adapted and "timestamp" not in adapted:
        adapted["timestamp"] = adapted["commit_timestamp"]

    feature_columns: List[str] = []

    for schema_name, source_name in EXTERNAL_ALIASES.items():
        if source_name in adapted:
            adapted[schema_name] = pd.to_numeric(adapted[source_name], errors="coerce")
            feature_columns.append(schema_name)

    # The cross family is this project's contribution over the source corpus:
    # the mined features are all change-only or test-only.
    cross = extract_cross_features(adapted)
    for column in CROSS_FEATURES:
        adapted[column] = cross[column]
    feature_columns.extend(CROSS_FEATURES)

    for column in EXTERNAL_EXTRA_FEATURES:
        if column in adapted and column not in feature_columns:
            adapted[column] = pd.to_numeric(adapted[column], errors="coerce")
            feature_columns.append(column)

    unavailable = sorted(set(ALL_FEATURES) - set(feature_columns))
    if unavailable:
        LOGGER.info(
            "%d schema features unavailable on this corpus (documented in "
            "EXTERNAL_UNAVAILABLE): %s",
            len(unavailable),
            ", ".join(unavailable),
        )
    return adapted, feature_columns


def _read_external_split(path: Path, name: str) -> pd.DataFrame:
    """Read one split CSV from an external corpus directory.

    Args:
        path: Directory holding ``train.csv``/``val.csv``/``test.csv``.
        name: Split name without extension.

    Returns:
        The loaded frame.

    Raises:
        PreprocessError: If the file is missing.
    """
    source = path / f"{name}.csv"
    if not source.is_file():
        raise PreprocessError(f"External split not found: {source}")
    LOGGER.info("Reading %s (%.1f MB)...", source, source.stat().st_size / 1e6)
    return pd.read_csv(source, low_memory=False)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def preprocess(
    source: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    use_external: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the full preprocessing stage and write the three split CSVs.

    Args:
        source: Override for the input. A file path selects the harvest path;
            a directory selects the reuse path.
        output_dir: Destination for the split CSVs. Defaults to
            ``data.processed_dir``.
        use_external: Force the reuse path on or off. Defaults to auto-detect
            from ``data.external_splits_dir``.

    Returns:
        A summary dictionary, also written to ``data/processed/split_summary.json``.

    Raises:
        PreprocessError: If no usable input exists.
    """
    cfg = load_config()
    data_cfg = cfg["data"]
    destination = resolve_path(output_dir or data_cfg["processed_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    external_dir = data_cfg.get("external_splits_dir")
    if use_external is None:
        use_external = bool(external_dir) and resolve_path(external_dir).is_dir()

    carry = [c for c in ID_COLUMNS + ["mutant_index"] if c]

    # ---- Load and shape the three splits ---------------------------------
    if use_external:
        root = resolve_path(source or external_dir)
        LOGGER.info("Reuse path: adapting external corpus at %s", root)
        raw_splits = {name: _read_external_split(root, name) for name in ("train", "val", "test")}

        adapted_splits: Dict[str, pd.DataFrame] = {}
        feature_columns: List[str] = []
        for name, raw in raw_splits.items():
            adapted, columns = adapt_external_frame(raw)
            adapted_splits[name] = adapted
            feature_columns = columns  # identical schema across splits
        splits = adapted_splits
        provenance = {"mode": "external", "root": str(root), "unavailable": EXTERNAL_UNAVAILABLE}
    else:
        features_path = resolve_path(source or Path(data_cfg["processed_dir"]) / "features.csv")
        if not features_path.is_file():
            raise PreprocessError(
                f"No feature matrix at {features_path} and no external corpus configured. "
                "Run 'python cli.py collect' then 'python cli.py features' first."
            )
        LOGGER.info("Harvest path: reading %s", features_path)
        frame = pd.read_csv(features_path, low_memory=False)
        train, val, test = temporal_group_split(
            frame,
            ratios=data_cfg.get("split_ratios", (0.70, 0.15, 0.15)),
            group_column=data_cfg.get("group_column", "commit_sha"),
        )
        splits = {"train": train, "val": val, "test": test}
        feature_columns = [c for c in ALL_FEATURES if c in frame]
        provenance = {"mode": "harvest", "root": str(features_path), "unavailable": {}}

    categorical = [c for c in CATEGORICAL_CANDIDATES if c in splits["train"]]
    numeric = [c for c in feature_columns if c not in categorical]
    if not numeric:
        raise PreprocessError("No numeric features survived adaptation.")

    # ---- Fit on train only, then transform all three ---------------------
    modelled = numeric + categorical
    preprocessor = build_preprocessor(numeric, categorical)
    LOGGER.info("Fitting preprocessor on %d training rows, %d columns...", len(splits["train"]), len(modelled))
    preprocessor.fit(splits["train"].loc[:, modelled])

    summary: Dict[str, Any] = {
        "provenance": provenance,
        "n_numeric_features": len(numeric),
        "n_categorical_features": len(categorical),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "cross_features_recomputed": CROSS_FEATURES,
    }

    for name, frame in splits.items():
        transformed = _transformed_frame(preprocessor, frame, modelled, carry)
        target = destination / f"{name}.csv"
        transformed.to_csv(target, index=False)
        group_column = data_cfg.get("group_column", "commit_sha")
        summary[name] = {
            "n_rows": int(len(transformed)),
            "n_commits": int(frame[group_column].nunique()) if group_column in frame else None,
            "n_failures": int(transformed[LABEL_COLUMN].sum()),
            "failure_rate": float(transformed[LABEL_COLUMN].mean()),
            "path": str(target),
        }
        LOGGER.info(
            "%-5s -> %s (%d rows, %.3f%% failing)",
            name, target.name, len(transformed), 100.0 * transformed[LABEL_COLUMN].mean(),
        )

    summary["n_model_inputs"] = int(len(preprocessor.get_feature_names_out()))
    summary["model_input_names"] = _sanitize_feature_names(preprocessor.get_feature_names_out())
    save_artifact(
        {"preprocessor": preprocessor, "modelled_columns": modelled, "summary": summary},
        cfg["artifacts"]["preprocessor_path"],
    )

    summary_path = destination / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", summary_path)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m data.preprocess``.

    Returns:
        ``0`` on success, ``1`` on a handled preprocessing error.
    """
    parser = argparse.ArgumentParser(description="Preprocess features into train/val/test splits.")
    parser.add_argument("--source", default=None, help="Input features CSV or external splits directory.")
    parser.add_argument("--output-dir", default=None, help="Destination directory.")
    parser.add_argument("--external", dest="external", action="store_true", help="Force the reuse path.")
    parser.add_argument("--no-external", dest="external", action="store_false", help="Force the harvest path.")
    parser.set_defaults(external=None)
    args = parser.parse_args(argv)

    try:
        preprocess(source=args.source, output_dir=args.output_dir, use_external=args.external)
    except PreprocessError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
