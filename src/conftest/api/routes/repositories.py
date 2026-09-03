"""
ConfTest Repositories Management API Route.

Endpoints:
- GET  /api/v1/repositories
- POST /api/v1/repositories
- GET  /api/v1/repositories/{repo_id}
- GET  /api/v1/repositories/{repo_id}/commits
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from conftest.api.schemas import RepositoryCreateSchema, RepositoryResponseSchema
from conftest.db.session import get_db
from conftest.db import crud
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/repositories", tags=["Repositories"])


@router.get("", response_model=List[RepositoryResponseSchema], status_code=status.HTTP_200_OK)
def list_repositories(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)) -> List[RepositoryResponseSchema]:
    """List all registered repositories."""
    repos = crud.list_repositories(db, skip=skip, limit=limit)
    return repos


@router.post("", response_model=RepositoryResponseSchema, status_code=status.HTTP_201_CREATED)
def create_repository(payload: RepositoryCreateSchema, db: Session = Depends(get_db)) -> RepositoryResponseSchema:
    """Register a new repository for CI regression test selection."""
    existing = crud.get_repository_by_name(db, payload.full_name)
    if existing:
        return existing

    repo = crud.create_repository(
        db=db,
        full_name=payload.full_name,
        url=payload.url,
        local_path=payload.local_path,
    )
    return repo


@router.get("/{repo_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_repository_details(repo_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve repository metadata, total test count, and total commits."""
    repo = crud.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Repository {repo_id} not found.")

    test_cases = crud.list_test_cases_for_repo(db, repo_id, limit=1000)
    commits = crud.list_commits_for_repo(db, repo_id, limit=1000)

    return {
        "id": repo.id,
        "full_name": repo.full_name,
        "url": repo.url,
        "local_path": repo.local_path,
        "total_test_cases": len(test_cases),
        "total_commits_mined": len(commits),
        "created_at": repo.created_at.isoformat(),
    }


@router.get("/{repo_id}/commits", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def list_repository_commits(repo_id: int, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List recent commits and selection decisions for a repository."""
    repo = crud.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Repository {repo_id} not found.")

    commits = crud.list_commits_for_repo(db, repo_id, skip=skip, limit=limit)
    out = []
    for c in commits:
        dec = crud.get_decision_for_commit(db, c.id)
        out.append({
            "id": c.id,
            "sha": c.sha,
            "message": c.message,
            "timestamp": c.timestamp.isoformat(),
            "decision_mode": dec.mode if dec else "NOT_EVALUATED",
            "abstained": dec.abstained if dec else False,
            # A count ratio, not a duration. See analytics.py.
            "test_reduction_pct": dec.estimated_saving if dec else None,
        })
    return out
