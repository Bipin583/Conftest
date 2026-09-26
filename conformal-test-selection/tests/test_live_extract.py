"""Live per-PR extraction: the record shape must match what the model was fit on.

The advisory job scores a *live* PR by turning its diff into candidate records
and feeding them to the fitted pipeline. The pipeline median-imputes any absent
column, so the danger is not a crash but a silent shape mismatch: a stray key
outside ``modelled_columns`` is dropped unnoticed, and — worse — materialising a
historical feature (e.g. ``co_change_frequency``) to a literal value would tell
the completeness check a feature is "present" when it was never measured. These
tests pin the invariant that live records carry only statically-computable
columns, that the 8 history features and ``mutant_operator`` stay *absent* (so
they are imputed, not faked), and that extraction degrades to ``[]`` — never an
exception — when there is nothing to score.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import pytest

from data.live_extract import (
    _is_source_path,
    _is_test_path,
    build_live_candidates,
    diff_level_features,
)

# The historical features have no live source in CI; they must be left absent so
# the fitted preprocessor imputes them rather than trusting a fabricated value.
HISTORY_KEYS = {
    "execution_count",
    "failure_rate_lifetime",
    "recent_failures",
    "duration_mean",
    "flakiness_score",
    "hist_prior_failures",
    "hist_has_ever_failed",
    "co_change_frequency",
}
ID_KEYS = {"test_id", "test_path", "changed_file_path"}
PLACEHOLDER = "PLACEHOLDER_APPEND_BELOW"


def _git(args: List[str], cwd: Path) -> str:
    """Run git in ``cwd`` and return stdout, raising on failure (test setup only)."""
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path):
    """A throwaway repo with a base commit and a head commit editing a source file.

    Returns ``(repo_root, base_sha, head_sha)``. The head commit touches
    ``src/widget.py`` (a source change, so extraction proceeds) and the tree
    carries a ``test_*`` file that imports it, giving the extractor a candidate
    to discover and a real import edge to score.
    """
    if not _git_available():
        pytest.skip("git CLI not available")

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.email", "t@example.com"], root)
    _git(["config", "user.name", "Test"], root)

    (root / "src" / "widget.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "tests" / "test_widget.py").write_text(
        "import pytest\nfrom src.widget import add\n\n"
        "@pytest.mark.parametrize('a,b', [(1, 2)])\n"
        "def test_add(a, b):\n    assert add(a, b) == a + b\n",
        encoding="utf-8",
    )
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "base"], root)
    base = _git(["rev-parse", "HEAD"], root)

    (root / "src" / "widget.py").write_text(
        "def add(a, b):\n    # fix: guard against None\n    return (a or 0) + (b or 0)\n",
        encoding="utf-8",
    )
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "fix widget add"], root)
    head = _git(["rev-parse", "HEAD"], root)
    return str(root), base, head


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _modelled_columns():
    """The fitted preprocessor's column list, or ``None`` when artefacts are absent."""
    try:
        from models.pipeline import SelectionPipeline

        return set(SelectionPipeline.load().modelled_columns)
    except Exception:  # noqa: BLE001 - a fresh clone that has not trained yet
        return None


def test_diff_level_features_are_the_fourteen_pr_columns():
    """PR-level features are complete and the fix/refactor flags read the message."""
    diff = {
        "files": [(120, 30, "src/widget.py"), (4, 0, "tests/test_widget.py")],
        "message": "fix: guard add against None inputs",
    }
    feats = diff_level_features(diff)
    expected = {
        "lines_added", "lines_removed", "total_churn", "files_changed",
        "commit_message_length", "has_source_change", "has_test_change",
        "has_config_change", "diff_num_src_files", "diff_num_test_files",
        "diff_has_python", "diff_is_fix_commit", "diff_is_refactor_commit",
        "diff_msg_word_count",
    }
    assert set(feats) == expected
    assert feats["diff_is_fix_commit"] == 1.0
    assert feats["diff_is_refactor_commit"] == 0.0
    assert feats["has_source_change"] == 1.0 and feats["has_test_change"] == 1.0
    assert feats["total_churn"] == 154.0


def test_source_classification_ignores_test_substring_in_dir_names():
    """A source file under a dir whose name merely contains "test" stays source.

    Regression: this project lives under ``conformal-test-selection/``. A naive
    ``"test" in path`` check classified every file here as a test, so no changed
    source was ever detected and the advisory always fell back to the sample.
    """
    src = "conformal-test-selection/examples/advisory_report.py"
    assert _is_source_path(src) is True
    assert _is_test_path(src) is False
    # Genuine test files are still recognised (filename convention + tests/ dir).
    assert _is_test_path("conformal-test-selection/tests/test_live_extract.py") is True
    assert _is_test_path("pkg/foo_test.py") is True
    assert _is_test_path("tests/conftest.py") is True
    # A non-.py file is never a source candidate even outside any test dir.
    assert _is_source_path("conformal-test-selection/examples/sample.json") is False


def test_live_records_are_scored_and_carry_only_modelled_columns(git_repo):
    """Every emitted record is scoring-ready: ids + a subset of modelled columns."""
    root, base, head = git_repo
    records = build_live_candidates(root, base, head, "throwaway-repo", max_tests=50)

    assert records, "a source change with a discoverable test should yield candidates"
    modelled = _modelled_columns()
    for rec in records:
        feature_keys = set(rec) - ID_KEYS
        # No history feature or corpus artefact may be materialised — they must
        # stay absent so the preprocessor imputes rather than trusts a fake.
        assert not (feature_keys & HISTORY_KEYS), feature_keys & HISTORY_KEYS
        assert "mutant_operator" not in rec
        assert rec["repo"] == "throwaway-repo"
        assert rec["test_id"] and rec["test_path"]
        if modelled is not None:
            assert feature_keys <= modelled, feature_keys - modelled


def test_direct_import_edge_is_detected(git_repo):
    """The test imports the changed module, so the dependency proxy fires."""
    root, base, head = git_repo
    records = build_live_candidates(root, base, head, "throwaway-repo", max_tests=50)
    widget_test = [r for r in records if r["test_path"].endswith("test_widget.py")]
    assert widget_test, "the widget test should be discovered"
    assert any(r["dep_is_direct_import"] == 1.0 for r in widget_test)
    assert any(r["ast_test_is_parameterized"] == 1.0 for r in widget_test)


def test_no_source_change_returns_empty(tmp_path):
    """A diff touching only docs yields no candidates (fallback signal), not a raise."""
    if not _git_available():
        pytest.skip("git CLI not available")
    root = tmp_path / "docsonly"
    root.mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.email", "t@example.com"], root)
    _git(["config", "user.name", "Test"], root)
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "base"], root)
    base = _git(["rev-parse", "HEAD"], root)
    (root / "README.md").write_text("hello world\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "docs"], root)
    head = _git(["rev-parse", "HEAD"], root)

    assert build_live_candidates(root, base, head, "docsonly", 50) == []


def test_missing_shas_fall_back_to_last_commit(git_repo):
    """No base/head (a workflow_dispatch run) → HEAD~1..HEAD, still scoring live.

    ``pr_diff`` uses the last commit when no PR endpoints are supplied, so a
    dispatch run on a repo whose latest commit changed source still produces
    candidates rather than silently going to the sample.
    """
    root, _, _ = git_repo
    records = build_live_candidates(root, None, None, "throwaway-repo", 50)
    assert records, "the last commit changed src/widget.py, so candidates should exist"


def test_never_raises_on_bogus_repo(tmp_path):
    """A path that is not a git repo degrades to [] rather than propagating."""
    assert build_live_candidates(str(tmp_path), "aaa", "bbb", "nope", 50) == []
