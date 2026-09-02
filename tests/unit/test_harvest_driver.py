"""
Unit tests for the mutation harvest driver.

The driver's one safety-critical job is choosing the interpreter each subject
suite runs under. The screener builds a venv per repository with
``pip install -e .``, which is what makes a mutation in the checkout visible to
the tests; the ambient interpreter guarantees nothing and, when it happens to
hold a released copy of the same package, produces a full dataset of zeros.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from harvest_mutations import (  # noqa: E402
    accepted_from_report,
    estimate_hours,
    resolve_interpreter,
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
    import json

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
