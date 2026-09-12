"""Focused CLI-driver tests for scripts/harvest_bugsinpy.py."""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import harvest_bugsinpy as driver

from conftest.groundtruth.bugsinpy_adapter import CommandResult


def _metadata(tmp_path: Path, names=("tqdm", "unknown")) -> Path:
    root = tmp_path / "metadata"
    for name in names:
        (root / "projects" / name / "bugs").mkdir(parents=True)
    return root


def _result(args, code=0, stdout="", stderr="", timed_out=False):
    return CommandResult(list(args), code, stdout, stderr, 0.01, timed_out)


def test_clone_metadata_clones_then_records_exact_commit(tmp_path, monkeypatch):
    metadata = tmp_path / "meta"
    calls = []
    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[:3] == ["git", "clone", "--depth"]:
            (metadata / ".git").mkdir(parents=True)
            return _result(args)
        return _result(args, stdout="abcdef123456\n")
    monkeypatch.setattr(driver, "run_command", fake_run)
    assert driver.clone_metadata(metadata) == "abcdef123456"
    assert calls[0][0] == ["git", "clone", "--depth", "1", driver.BUGSINPY_URL, str(metadata)]
    assert calls[-1][0] == ["git", "rev-parse", "HEAD"]


def test_clone_metadata_updates_existing_checkout_and_tolerates_failed_fetch(tmp_path, monkeypatch):
    metadata = tmp_path / "meta"; (metadata / ".git").mkdir(parents=True)
    calls = []
    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "fetch":
            return _result(args, code=1, stderr="offline")
        return _result(args, stdout="deadbee\n")
    monkeypatch.setattr(driver, "run_command", fake_run)
    assert driver.clone_metadata(metadata) == "deadbee"
    assert not any(args[1:3] == ["reset", "--hard"] for args in calls)


def test_clone_metadata_failure_exits_with_captured_error(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "run_command", lambda args, **kwargs: _result(args, 1, stderr="network down"))
    with pytest.raises(SystemExit, match="git clone failed: network down"):
        driver.clone_metadata(tmp_path / "meta")


def test_resolve_projects_defaults_to_only_profiled_available_projects(tmp_path):
    metadata = _metadata(tmp_path)
    assert driver.resolve_projects(metadata, None) == ["tqdm"]


def test_resolve_projects_rejects_unknown_and_unprofiled(tmp_path):
    metadata = _metadata(tmp_path)
    with pytest.raises(SystemExit, match="is not in"):
        driver.resolve_projects(metadata, ["missing"])
    with pytest.raises(SystemExit, match="has no run profile"):
        driver.resolve_projects(metadata, ["unknown"])


def test_main_summary_mode_reads_existing_summary_only(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out"; output.mkdir()
    summary = {
        "n_records": 1, "n_labelled": 0, "n_projects_with_labels": 0,
        "projects_with_labels": [], "status_counts": {"env_error": 1},
        "base_python_version": "3.11.9", "allow_interpreter_mismatch": False,
        "per_project": {}, "records_path": "records.jsonl",
    }
    (output / "harvest_summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(driver, "clone_metadata", lambda *a: pytest.fail("must not clone"))
    assert driver.main(["--summary", "--output", str(output)]) == 0
    assert "usable as labels:   0" in capsys.readouterr().out


def test_main_summary_mode_refuses_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="nothing has been harvested"):
        driver.main(["--summary", "--output", str(tmp_path / "out")])


def test_main_survey_writes_offline_payload(tmp_path, monkeypatch):
    metadata = _metadata(tmp_path, ("tqdm",)); output = tmp_path / "out"
    payload = {
        "n_bugs": 1, "n_projects": 1, "n_with_readable_test_command": 1,
        "pinned_python_versions": {"3.8.3": 1},
        "per_project": {"tqdm": {"n_bugs": 1, "n_with_readable_test_command": 1,
                                   "has_run_profile": True, "python_versions": {"3.8.3": 1}}},
    }
    monkeypatch.setattr(driver, "survey", lambda root, projects: payload)
    assert driver.main(["--survey", "--metadata", str(metadata), "--output", str(output)]) == 0
    assert json.loads((output / "bugsinpy_survey.json").read_text()) == payload


def test_main_refuses_missing_metadata_before_any_harvest(tmp_path):
    with pytest.raises(SystemExit, match="Run with --clone first"):
        driver.main(["--metadata", str(tmp_path / "missing")])


def test_main_forwards_options_stamps_workspace_and_returns_success(tmp_path, monkeypatch):
    metadata = _metadata(tmp_path, ("tqdm",)); workspace = tmp_path / "work"; output = tmp_path / "out"
    captured = {}
    class FakeHarvester:
        def __init__(self, **kwargs): captured["init"] = kwargs
        def base_python_version(self): return "3.11.9"
        def harvest(self, **kwargs):
            captured["harvest"] = kwargs
            return {"n_records": 1, "n_labelled": 1, "n_projects_with_labels": 1,
                    "projects_with_labels": ["tqdm"], "status_counts": {"harvested": 1},
                    "base_python_version": "3.11.9", "allow_interpreter_mismatch": True,
                    "per_project": {"tqdm": {"records": 1, "harvested": 1,
                    "mean_universe_size": 3, "mean_n_killed": 1}}, "records_path": "x"}
    monkeypatch.setattr(driver, "BugsInPyHarvester", FakeHarvester)
    monkeypatch.setattr(driver, "isolate_workspace_from_project_config", lambda path: captured.setdefault("marker", path / "pytest.ini"))
    rc = driver.main(["--metadata", str(metadata), "--workspace", str(workspace), "--output", str(output),
                      "--projects", "tqdm", "--bugs", "1", "2", "--limit", "1",
                      "--baseline-runs", "3", "--suite-timeout", "17", "--base-python", sys.executable,
                      "--allow-interpreter-mismatch", "--pinned-requirements", "--force"])
    assert rc == 0 and captured["marker"] == workspace / "pytest.ini"
    assert captured["init"]["baseline_runs"] == 3 and captured["init"]["suite_timeout"] == 17
    assert captured["init"]["allow_interpreter_mismatch"] is True
    assert captured["init"]["install_pinned_requirements"] is True
    assert captured["harvest"] == {"projects": ["tqdm"], "bug_ids": [1, 2], "limit": 1, "force": True}


def test_main_returns_failure_when_no_measured_labels(tmp_path, monkeypatch):
    metadata = _metadata(tmp_path, ("tqdm",))
    class FakeHarvester:
        def __init__(self, **kwargs): pass
        def base_python_version(self): return "3.11.9"
        def harvest(self, **kwargs):
            return {"n_records": 1, "n_labelled": 0, "n_projects_with_labels": 0,
                    "projects_with_labels": [], "status_counts": {"env_error": 1},
                    "base_python_version": "3.11.9", "allow_interpreter_mismatch": False,
                    "per_project": {}, "records_path": "x"}
    monkeypatch.setattr(driver, "BugsInPyHarvester", FakeHarvester)
    monkeypatch.setattr(driver, "isolate_workspace_from_project_config", lambda path: path / "pytest.ini")
    assert driver.main(["--metadata", str(metadata), "--projects", "tqdm",
                        "--workspace", str(tmp_path / "work"), "--output", str(tmp_path / "out")]) == 1
