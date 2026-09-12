"""
Compatibility import path for the feature extractor.

The documented and published module name for feature extraction is
``conftest.features.extractor``; the implementation lives in
:mod:`conftest.features.pipeline`, which composes the four per-group miners
(:mod:`~conftest.features.diff_features`, :mod:`~conftest.features.ast_features`,
:mod:`~conftest.features.dependency_graph`,
:mod:`~conftest.features.history_features`) into the 32-column matrix.

This module contains no logic. It re-exports, so that the two names cannot
disagree about how a feature is computed. Import either one; prefer
``conftest.features.pipeline`` in new code, since that is where the ordering of
``FEATURE_NAMES`` is defined and the ordering is what the trained artifacts
depend on.

Note for anyone reading feature importances: 13 of the 32 columns are constant
across the training split, including all 12 diff/churn columns
(``reports/ablation_study.json``, ``diff_churn_only`` scores ROC-AUC 0.5000).
That is a defect in diff mining, not a property of churn.
"""

from __future__ import annotations

from conftest.features.ast_features import extract_ast_metrics_from_file
from conftest.features.dependency_graph import DependencyGraphBuilder
from conftest.features.diff_features import extract_diff_features
from conftest.features.history_features import extract_history_features_from_db
from conftest.features.pipeline import FEATURE_NAMES, FeatureExtractionPipeline

# Alias under the name the docs use for the class as well as the module.
FeatureExtractor = FeatureExtractionPipeline

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractionPipeline",
    "FeatureExtractor",
    "extract_diff_features",
    "extract_ast_metrics_from_file",
    "extract_history_features_from_db",
    "DependencyGraphBuilder",
]
