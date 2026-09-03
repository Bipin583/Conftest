"""
Unit tests for the mutation harvest driver.

The driver's one safety-critical job is choosing the interpreter each subject
suite runs under. The screener builds a venv per repository with
``pip install -e .``, which is what makes a mutation in the checkout visible to
the tests; the ambient interpreter guarantees nothing and, when it happens to
hold a released copy of the same package, produces a full dataset of zeros.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from harvest_mutations import (  # noqa: E402
    accepted_from_report,
    estimate_hours,
    harvest_one,
    resolve_interpreter,
)
from conftest.groundtruth.mutation_harness import (  # noqa: E402
    SUMMARY_FILENAME,
    MutationHarness,
)


def _fake_venv(workspace: Path, name: str) -> Path:
    """Create the interpreter path the screener would have produced."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    python = workspace / f".venv_{name}" / bin_dir / exe
    python.parent.mkdir(parents=True)
    python.write_text("")
    return python


def test_resolve_interpreter_finds_the_screened_environment(tmp_path):
    expected = _fake_venv(tmp_path, "demo")

    assert resolve_interpreter(tmp_path, "demo") == expected


def test_resolve_interpreter_refuses_to_fall_back_to_sys_executable(tmp_path):
    with pytest.raises(FileNotFoundError, match="no screened environment"):
        resolve_interpreter(tmp_path, "demo")


def test_resolve_interpreter_is_per_repository(tmp_path):
    """One repo's environment must never stand in for another's."""
    _fake_venv(tmp_path, "alpha")

    assert resolve_interpreter(tmp_path, "alpha").exists()
    with pytest.raises(FileNotFoundError):
        resolve_interpreter(tmp_path, "beta")


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

def _write_report(path: Path, candidates: dict) -> Path:
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    return path


def test_only_stage3_survivors_are_eligible(tmp_path):
    report = _write_report(tmp_path / "screening_report.json", {
        "kept": {"stage1_passed": True, "stage2_passed": True, "stage3_passed": True},
        "structurally_useless": {"stage1_passed": True, "stage2_passed": True, "stage3_passed": False},
    })

    assert list(accepted_from_report(report)) == ["kept"]


def test_a_report_with_no_survivor_is_an_error_not_an_empty_run(tmp_path):
    report = _write_report(tmp_path / "screening_report.json", {
        "rejected": {"stage1_passed": True, "stage2_passed": False, "stage3_passed": False},
    })

    with pytest.raises(ValueError, match="No repository passed stage 3"):
        accepted_from_report(report)


def test_missing_report_names_the_command_that_creates_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="screen_repos.py"):
        accepted_from_report(tmp_path / "screening_report.json")


def test_cost_estimate_scales_with_suite_duration_and_mutants():
    entries = [{"suite_seconds": 3.6}, {"suite_seconds": 7.2}]

    assert estimate_hours(entries, 100) == pytest.approx(0.3)


# --------------------------------------------------------------------------
# The driver knows two things the harness does not, and both must be on disk
# --------------------------------------------------------------------------

def test_harvest_one_persists_the_summary_with_the_drivers_own_fields(tmp_path, monkeypatch):
    workspace = tmp_path / 'repos'
    (workspace / 'demo' / 'pkg').mkdir(parents=True)
    (workspace / 'demo' / 'pkg' / '__init__.py').write_text('')
    harvest_root = tmp_path / 'harvest'
    python = _fake_venv(workspace, 'demo')

    forwarded = {}

    def fake_run(self, n_mutants, resume, restore_checkout=False,
                 reprofile_baseline=False, break_lock=False):
        forwarded.update(
            restore_checkout=restore_checkout,
            reprofile_baseline=reprofile_baseline,
            break_lock=break_lock,
        )
        return {'repo': self.repo_name, 'n_sampled': n_mutants,
                'import_provenance': {'verified': True, 'modules': {'pkg': 'x'}}}

    monkeypatch.setattr(MutationHarness, 'run', fake_run)

    summary = harvest_one(
        name='demo',
        entry={'source_dirs': ['pkg'], 'commit_sha': 'cafef00d'},
        workspace=workspace,
        harvest_root=harvest_root,
        n_mutants=7,
        resume=False,
        python_executable=python,
        restore_checkout=True,
        reprofile_baseline=True,
        break_lock=True,
    )

    assert summary['n_sampled'] == 7
    # Every repair switch has to reach the harness; a driver that swallowed one
    # would silently harvest a dirty checkout against a stale baseline, or wait
    # forever on a lock the operator had already cleared.
    assert forwarded == {
        'restore_checkout': True, 'reprofile_baseline': True, 'break_lock': True,
    }
    on_disk = json.loads((harvest_root / 'demo' / SUMMARY_FILENAME).read_text())
    assert on_disk['screened_commit_sha'] == 'cafef00d'
    assert on_disk['wall_clock_seconds'] >= 0
    assert on_disk['import_provenance']['verified'] is True


def test_harvest_one_refuses_a_repo_with_no_source_dirs(tmp_path):
    workspace = tmp_path / 'repos'
    (workspace / 'demo').mkdir(parents=True)
    with pytest.raises(ValueError, match='no source_dirs'):
        harvest_one(
            name='demo', entry={}, workspace=workspace,
            harvest_root=tmp_path / 'harvest', n_mutants=1, resume=False,
            python_executable=_fake_venv(workspace, 'demo'),
        )
