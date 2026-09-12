"""
ConfTest Explainability API Route.

Endpoint: POST /api/v1/explain
Returns exact SHAP TreeExplainer feature attributions and developer reason cards.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException, status

from conftest.api.schemas import ExplainRequestSchema, ExplainResponseSchema, FeatureDriverSchema
from conftest.config import settings
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.explainability.shap_explainer import ShapExplainer
from conftest.explainability.rules import RuleBasedExplainer
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/explain", tags=["Model Explainability"])

_explainer_cache: Dict[str, ShapExplainer] = {}


ENSEMBLE_PRODUCED_BY = "python scripts/train_ensemble.py"


def _first_ensemble_member() -> Path:
    """
    Resolve the ensemble member that SHAP attributions are computed against.

    The member is read from the ensemble's own metadata rather than assumed
    from a filename, because the seed is part of that filename and hardcoding
    it silently pins one training run.

    Raises:
        HTTPException: 503 when no ensemble is present, naming the script that
            would produce one. There is deliberately no fallback to some other
            model on disk: an attribution is only an explanation of the
            prediction if it comes from the model that made it, so serving one
            model's SHAP values for another model's decision would be a
            fabrication, not a degraded answer.
    """
    ensemble_dir = Path(settings.ensemble_path)
    meta_path = ensemble_dir / "ensemble_metadata.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No trained ensemble at {ensemble_dir}: {meta_path.name} is missing. "
                f"Produce one with `{ENSEMBLE_PRODUCED_BY}`."
            ),
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        member_files = meta["member_files"]
        if not member_files:
            raise ValueError("member_files is empty")
        # Metadata written before the portability fix stored absolute paths from
        # whichever machine trained the ensemble; take the basename so a foreign
        # checkout path cannot send us looking outside this one.
        member = ensemble_dir / Path(str(member_files[0])).name
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Ensemble metadata at {meta_path} is unreadable or names no members "
                f"({exc}). Re-run `{ENSEMBLE_PRODUCED_BY}`."
            ),
        )

    if not member.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Ensemble metadata names {member.name}, which is not in {ensemble_dir}. "
                f"Re-run `{ENSEMBLE_PRODUCED_BY}`."
            ),
        )
    return member


def get_shap_explainer() -> ShapExplainer:
    """Load or retrieve the cached SHAP explainer for the serving ensemble."""
    model_path = _first_ensemble_member()

    key = str(model_path)
    if key not in _explainer_cache:
        logger.info("Building SHAP explainer from ensemble member: %s", model_path)
        predictor = LightGBMTestPredictor.load(str(model_path))
        _explainer_cache[key] = ShapExplainer(predictor)
    return _explainer_cache[key]


@router.post("", response_model=ExplainResponseSchema, status_code=status.HTTP_200_OK)
def explain_test_prediction(payload: ExplainRequestSchema) -> ExplainResponseSchema:
    """
    Explain why a specific test received its failure risk score.

    Computes local Shapley values (SHAP) across all 32 features and formats natural language reasons.
    """
    try:
        explainer = get_shap_explainer()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize explainer: {str(exc)}",
        )

    # Convert features dict to ordered vector
    x_vec = np.array([float(payload.features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)

    shap_breakdown = explainer.explain_instance(x_vec, top_k=payload.top_k)
    top_pos = [
        FeatureDriverSchema(
            feature=f["feature"],
            feature_value=f["feature_value"],
            shap_attribution=f["shap_attribution"],
            impact=f["impact"],
        )
        for f in shap_breakdown["top_risk_increasing_features"]
    ]
    top_neg = [
        FeatureDriverSchema(
            feature=f["feature"],
            feature_value=f["feature_value"],
            shap_attribution=f["shap_attribution"],
            impact=f["impact"],
        )
        for f in shap_breakdown["top_risk_decreasing_features"]
    ]

    rule_explainer = RuleBasedExplainer()
    card = rule_explainer.generate_test_reason_card(
        test_id=payload.test_id,
        feature_dict=payload.features,
        shap_drivers=shap_breakdown["top_risk_increasing_features"],
        confidence=shap_breakdown["predicted_probability"],
    )

    return ExplainResponseSchema(
        test_id=payload.test_id,
        predicted_probability=shap_breakdown["predicted_probability"],
        base_expected_value=shap_breakdown["base_expected_value"],
        risk_level=card["risk_level"],
        primary_reasons=card["primary_reasons"],
        top_risk_increasing_features=top_pos,
        top_risk_decreasing_features=top_neg,
    )
