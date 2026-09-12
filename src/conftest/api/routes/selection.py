"""
ConfTest Selection API Route.

Endpoint: POST /api/v1/select
Executes confidence-calibrated regression test selection with selective abstention fallback.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from conftest.api.schemas import SelectRequestSchema, SelectResponseSchema, RankedTestSchema
from conftest.config import settings
from conftest.db.session import get_db
from conftest.db import crud
from conftest.engine.selector_engine import ConfTestEngine
from conftest.explainability.rules import RuleBasedExplainer
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/select", tags=["Test Selection"])

# Lazy global engine instance
_engine_instance: Dict[str, ConfTestEngine] = {}


def get_engine(repo_root: Optional[str] = None) -> ConfTestEngine:
    """
    Retrieve or initialize cached engine instance for a repository path.

    Artifact locations come from settings rather than literals, so which model,
    calibrator and tuned policy a deployment serves is an environment decision
    (CONFTEST_ENSEMBLE_PATH and friends) instead of a source edit.
    """
    root = repo_root or str(settings.default_repo_root)
    if root not in _engine_instance:
        _engine_instance[root] = ConfTestEngine(
            repo_root=root,
            ensemble_path=str(settings.ensemble_path),
            calibrator_path=str(settings.calibrator_path),
            policy_config_path=str(settings.policy_config_path),
        )
    return _engine_instance[root]


@router.post("", response_model=SelectResponseSchema, status_code=status.HTTP_200_OK)
def select_regression_tests(
    payload: SelectRequestSchema,
    db: Session = Depends(get_db),
) -> SelectResponseSchema:
    """
    Select regression tests for a commit diff with selective prediction fallback.

    - **FAST_SELECTED**: Model is confident and low uncertainty; selects top risk tests.
    - **SAFE_FULL_SUITE**: Model is uncertain or diff is OOD; safely runs full suite.
    """
    repo_path = payload.repo_path or str(settings.default_repo_root)
    engine = get_engine(repo_path)

    # Convert changed files to dictionary format.
    #
    # An empty diff is a client error, not something to paper over. This used to
    # substitute a fixed made-up change to "src_app/auth.py" with 15 added lines,
    # so a caller that sent no diff still received a confident-looking selection
    # with per-test confidences and an epistemic-uncertainty figure -- all of it
    # computed from features of a file the commit never touched.
    changed_files = [f.model_dump() for f in payload.changed_files]
    if not changed_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "changed_files is empty: test selection is a function of the diff, "
                "so there is nothing to rank. Send the commit's changed files, or "
                "run the full suite if the diff is genuinely unknown."
            ),
        )

    # Ensure repository is registered in DB
    repo = crud.get_or_create_repository(
        db=db,
        full_name=payload.repository_name,
        url=f"https://github.com/{payload.repository_name}",
        local_path=repo_path,
    )

    try:
        outcome = engine.analyze_and_select(
            commit_sha=payload.commit_sha,
            changed_files=changed_files,
            commit_message=payload.commit_message,
            budget_ratio=payload.budget_ratio,
            db=db,
            repository_id=repo.id,
            execute=payload.execute,
        )

        # Generate PR Markdown summary
        rule_explainer = RuleBasedExplainer()
        markdown_summary = rule_explainer.generate_commit_markdown_summary(
            commit_sha=payload.commit_sha,
            decision_dict=outcome,
            top_tests=outcome.get("ranked_tests", []),
        )

        ranked_test_schemas = [
            RankedTestSchema(
                test_id=t["test_id"],
                raw_score=t["raw_score"],
                calibrated_confidence=t["calibrated_confidence"],
                epistemic_uncertainty=t["epistemic_uncertainty"],
                is_selected=t["is_selected"],
                reasons=[f"Confidence: {t['calibrated_confidence']:.2%}, Uncertainty: {t['epistemic_uncertainty']:.4f}"],
            )
            for t in outcome.get("ranked_tests", [])
        ]

        return SelectResponseSchema(
            commit_sha=outcome["commit_sha"],
            decision_mode=outcome["decision_mode"],
            abstained=outcome["abstained"],
            selected_count=outcome["selected_count"],
            total_count=outcome["total_count"],
            test_reduction_pct=outcome["test_reduction_pct"],
            top_confidence=outcome["top_confidence"],
            epistemic_uncertainty=outcome["epistemic_uncertainty"],
            reasons=outcome["reasons"],
            selected_test_ids=outcome["selected_test_ids"],
            ranked_tests=ranked_test_schemas,
            markdown_summary=markdown_summary,
            execution_outcome=outcome.get("execution_outcome"),
        )
    except HTTPException:
        # Already carries a deliberate status code (e.g. the 422 above, or a 422
        # raised deeper in the pipeline). Re-raising unchanged keeps a client
        # error from being reported as a server fault.
        raise
    except FileNotFoundError as exc:
        # A missing ensemble, calibrator or policy file is a deployment fault and
        # must not be reported as a bad request.
        logger.error(f"Required ConfTest artifact missing: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ConfTest model artifacts are not available on this server. "
                "Check CONFTEST_ENSEMBLE_PATH, CONFTEST_CALIBRATOR_PATH and "
                "CONFTEST_POLICY_CONFIG_PATH."
            ),
        )
    except Exception as exc:
        # The message can contain absolute paths and internal identifiers, so it
        # goes to the log; the client gets the commit SHA it can quote instead.
        logger.error(f"Test selection failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Test selection pipeline failed for commit {payload.commit_sha}. "
                "See server logs for details."
            ),
        )
