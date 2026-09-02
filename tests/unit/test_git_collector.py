"""
Unit tests for Git repository mining, GitHub client, synthetic dataset generation, and checkpointing.
"""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest.repository.git_collector import anonymize_author, GitRepositoryMiner
from conftest.github.client import mask_token, GitHubClient
from conftest.repository.synthetic_generator import (
    SYNTHETIC_ORIGIN_TAG,
    SyntheticRepositoryGenerator,
)
from conftest.repository.collector_service import CollectorService
from conftest.db import crud


def test_anonymize_author():
    """Verify author anonymization is deterministic, privacy-safe, and handles empty inputs."""
    hash1 = anonymize_author("alice@example.com")
    hash2 = anonymize_author("alice@example.com")
    hash3 = anonymize_author("bob@example.com")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 16
    assert anonymize_author(None) == "anonymous_author"


def test_token_masking():
    """Verify GitHub API tokens are safely masked in logs."""
    token = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
    masked = mask_token(token)
    assert masked.startswith("ghp_")
    assert masked.endswith("wxyz")
    assert "1234567890" not in masked
    assert mask_token(None) == "None"


def test_synthetic_data_generator_reproducibility(db_session: Session):
    """Verify that synthetic generator is deterministic with fixed seed."""
    gen1 = SyntheticRepositoryGenerator(random_seed=1337, acknowledge_synthetic=True)
    dataset1 = gen1.generate_repository_suite(n_commits=10, n_tests=5)

    gen2 = SyntheticRepositoryGenerator(random_seed=1337, acknowledge_synthetic=True)
    dataset2 = gen2.generate_repository_suite(n_commits=10, n_tests=5)

    assert dataset1["metadata"]["data_origin"] == SYNTHETIC_ORIGIN_TAG
    assert len(dataset1["commits"]) == 10
    assert len(dataset1["test_cases"]) == 5
    assert dataset1["commits"][0]["sha"] == dataset2["commits"][0]["sha"]

    # Verify persistence to DB
    summary = gen1.persist_to_database(db_session, dataset1)
    assert summary["commits_saved"] == 10
    assert summary["test_cases"] == 5

    # Verify DB contents
    repo = crud.get_repository(db_session, summary["repository_id"])
    assert repo is not None
    commits = crud.list_commits_for_repo(db_session, repo.id)
    assert len(commits) == 10


def test_local_git_miner_with_temp_repo(tmp_path: Path):
    """Verify GitRepositoryMiner on an isolated Git repository with test commits."""
    import git

    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))

    # Commit 1: Initial commit
    file1 = repo_dir / "math_utils.py"
    file1.write_text("def add(a, b):\n    return a + b\n")
    repo.index.add(["math_utils.py"])
    repo.index.commit("feat: initial commit with add function")

    # Commit 2: Modify file and add test
    file1.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
    test_file = repo_dir / "test_math.py"
    test_file.write_text("from math_utils import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    repo.index.add(["math_utils.py", "test_math.py"])
    commit2 = repo.index.commit("feat: add sub function and unit test")

    miner = GitRepositoryMiner(str(repo_dir))
    count = miner.get_commit_count()
    assert count == 2

    commits = list(miner.iter_commits(max_commits=5))
    assert len(commits) == 2

    record = miner.mine_commit_record(commit2)
    assert record["sha"] == commit2.hexsha
    assert len(record["changed_files"]) == 2
    assert record["stats"]["lines_added"] > 0


def test_collector_service_checkpointing(tmp_path: Path):
    """Verify checkpoint save and resume functionality."""
    service = CollectorService(
        repo_path=".",
        repo_name="test/checkpoint-repo",
        output_dir=str(tmp_path),
    )

    # Initial checkpoint is empty
    cp = service.load_checkpoint()
    assert cp["processed_shas"] == []

    # Save checkpoint
    service.save_checkpoint(["sha_001", "sha_002"])
    cp_loaded = service.load_checkpoint()
    assert cp_loaded["processed_shas"] == ["sha_001", "sha_002"]
    assert cp_loaded["count"] == 2


def test_the_synthetic_generator_cannot_be_built_without_saying_so():
    # The guard exists so a fabricated dataset cannot reach an evaluation through
    # a default argument. There is no default that quietly says yes.
    with pytest.raises(ValueError, match='acknowledge_synthetic'):
        SyntheticRepositoryGenerator(random_seed=1337)


def test_every_fabricated_row_carries_the_origin_tag():
    gen = SyntheticRepositoryGenerator(random_seed=7, acknowledge_synthetic=True)
    dataset = gen.generate_repository_suite(n_commits=3, n_tests=4)

    assert dataset['metadata']['data_origin'] == SYNTHETIC_ORIGIN_TAG
    assert all(c['data_origin'] == SYNTHETIC_ORIGIN_TAG for c in dataset['commits'])
    assert all(
        run['data_origin'] == SYNTHETIC_ORIGIN_TAG
        for c in dataset['commits'] for run in c['test_runs']
    )
