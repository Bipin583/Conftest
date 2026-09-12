"""Focused unit tests for the offline BugsInPy adapter and harvester safeguards."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest.groundtruth import bugsinpy_adapter as bip
from conftest.groundtruth.mutation_harness import TIMEOUT_BROKEN_STATUS, BaselineProfile


def _metadata(tmp_path: Path) -> Path:
    root = tmp_path / "BugsInPy"
    bug = root / "projects" / "tqdm" / "bugs" / "1"
    bug.mkdir(parents=True)
    (bug.parent.parent / "project.info").write_text('github_url="https://example.invalid/tqdm/"\n')
    (bug / "bug.info").write_text(
        'python_version="3.8.3"\nbuggy_commit_id="abcdef1"\n'
        'fixed_commit_id="1234567"\ntest_file="tests/a.py;tests/b.py"\npythonpath="src;lib"\n'
    )
    (bug / "run_test.sh").write_text(
        "pytest tests/a.py::TestA::test_x\npython -m unittest -q tests.test_b.TestB.test_y\n"
        "custom-runner --case unknown\n"
    )
    (bug / "bug_patch.txt").write_text(
        "--- a/tqdm/core.py\n+++ b/tqdm/core.py\n@@ -10,2 +10,3 @@\n-old()\n+new()\n+extra()\n"
    )
    (bug / "requirements.txt").write_text("# frozen\nnose==1.3.7\n")
    (bug / "setup.sh").write_text("# setup\npip install wheel\n")
    return root


def _bug(root: Path, bug_id: int = 1) -> bip.BugSpec:
    spec = bip.load_bug(root, "tqdm", 1)
    return spec if bug_id == 1 else bip.BugSpec(**{**spec.__dict__, "bug_id": bug_id})


def _harvester(tmp_path: Path, root: Path, **kwargs) -> bip.BugsInPyHarvester:
    return bip.BugsInPyHarvester(root, tmp_path / "work", tmp_path / "out", **kwargs)


def test_metadata_loader_parses_commands_patch_and_auxiliary_files(tmp_path):
    spec = _bug(_metadata(tmp_path))
    assert spec.github_url == "https://example.invalid/tqdm"
    assert spec.documented_failing_tests == (
        "tests/a::TestA::test_x", "tests/test_b::TestB::test_y"
    )
    assert spec.unreadable_test_commands == ("custom-runner --case unknown",)
    assert spec.documented_test_files() == ["tests/a.py", "tests/b.py"]
    assert spec.pinned_requirements == ("nose==1.3.7",)
    assert spec.setup_commands == ("pip install wheel",)
    assert spec.patch.changed_files == ("tqdm/core.py",)
    assert (spec.patch.lines_deleted, spec.patch.lines_added, spec.patch.first_line) == (1, 2, 10)
    assert (spec.patch.buggy_snippet, spec.patch.fixed_snippet) == ("old()", "new() / extra()")


def test_info_parser_is_data_only_and_layout_errors_are_explicit(tmp_path):
    text = 'safe="value"\n# ignored\n$(touch owned)\nnot-valid = "x"\n'
    assert bip.parse_info_file(text) == {"safe": "value"}
    root = _metadata(tmp_path)
    (root / "projects" / "tqdm" / "bugs" / "1" / "bug.info").write_text('fixed_commit_id="1234567"')
    with pytest.raises(bip.BugsInPyLayoutError, match="buggy_commit_id"):
        bip.load_bug(root, "tqdm", 1)


@pytest.mark.parametrize("raw, expected", [
    ("tests\\test_a.py::Case::test_x", "tests/test_a::Case::test_x"),
    ("tests/test_a.py", "tests/test_a.py"),
])
def test_test_id_normalisation(raw, expected):
    assert bip.normalise_test_id(raw) == expected


def test_documented_matching_handles_class_parametrisation_and_whole_files():
    candidates = ["tests/test_a.py::test_x[one]", "tests/test_a::test_x[two]", "tests/test_a::test_y"]
    matches = bip.match_documented(
        ["tests/test_a::Case::test_x", "tests/test_a.py", "tests/missing::test_z"], candidates
    )
    assert matches["tests/test_a::Case::test_x"] == candidates[:2]
    assert matches["tests/test_a.py"] == sorted(candidates)
    assert matches["tests/missing::test_z"] == []


def test_profiles_are_observed_and_unknown_projects_are_refused():
    assert bip.profile_for("tqdm").python_files == "tests_*.py"
    assert bip.profile_for("youtube-dl").test_paths == ("test",)
    with pytest.raises(bip.BugsInPyLayoutError, match="No run profile"):
        bip.profile_for("invented")


@pytest.mark.parametrize("value", ["abc", "g123456", "123456; rm -rf .", ""])
def test_sha_validation_rejects_untrusted_metadata(value):
    with pytest.raises(bip.BugsInPyLayoutError, match="not a commit id"):
        bip._require_sha(value, "commit")
    assert bip._require_sha("ABCDEF1", "commit") == "ABCDEF1"


def test_run_command_captures_timeout_and_os_errors(monkeypatch):
    monkeypatch.setattr(bip.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(k["args"] if "args" in k else a[0], 7, output="partial")
    ))
    timed = bip.run_command(["tool"], timeout=7)
    assert (timed.returncode, timed.timed_out, timed.stdout) == (124, True, "partial")
    assert "timed out after 7s" in timed.stderr
    monkeypatch.setattr(bip.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("missing")))
    failed = bip.run_command(["tool"])
    assert (failed.returncode, failed.timed_out, failed.stderr) == (127, False, "missing")


def test_pytest_environment_stamps_are_restored(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q")
    with bip.pytest_addopts("-o python_files=tests_*.py"):
        assert bip.os.environ["PYTEST_ADDOPTS"] == "-q -o python_files=tests_*.py"
    assert bip.os.environ["PYTEST_ADDOPTS"] == "-q"


def test_metadata_stamps_interpreter_and_unmeasured_rows_have_no_labels(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    monkeypatch.setattr(h, "metadata_commit", lambda: "feed123")
    monkeypatch.setattr(h, "base_python_version", lambda: "3.11.9")
    row = h.not_measured(bug, "interpreter_mismatch", "pin differs")
    assert row["bugsinpy_commit"] == "feed123" and row["python_version_used"] == "3.11.9"
    assert row["interpreter_matches_pin"] is False and row["interpreter_matches_pin_minor"] is False
    assert row["labels_measured"] is False and row["excluded_from_labels"] is True
    assert row["universe_size"] is None and row["killed"] is None and row["n_killed"] is None


def test_measure_refuses_pin_mismatch_before_checkout(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    monkeypatch.setattr(h, "base_python_version", lambda: "3.11.9")
    monkeypatch.setattr(h, "ensure_repo", lambda bug: pytest.fail("must not touch network/git"))
    row = h.measure(bug)
    assert row["status"] == "interpreter_mismatch"
    assert row["labels_measured"] is False and row["kill_ratio"] is None


def test_overlay_widens_to_changed_tests_but_never_overlays_source_patch(tmp_path):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    profile = bip.ProjectProfile(package="tqdm", test_paths=("tests",))
    changed = ["tests/helper.py", "tqdm/core.py", "docs/readme.md"]
    assert h.test_overlay_paths(bug, profile, changed) == [
        "tests/a.py", "tests/b.py", "tests/helper.py"
    ]


def test_defect_classification_counts_failures_and_missing_but_not_skips(tmp_path):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root)
    baseline = BaselineProfile(stable_passing=["a", "b", "c", "d"])
    result = SimpleNamespace(test_runs=[
        {"test_id": "a", "status": "PASSED"}, {"test_id": "b", "status": "FAILED"},
        {"test_id": "c", "status": "SKIPPED"},
    ])
    scored = h.classify_defect_run(baseline, result)
    assert scored == {
        "killed": ["b", "d"], "n_killed": 2, "kill_ratio": 0.5,
        "n_missing_at_defect": 1, "n_skipped_at_defect": 1,
        "universe_size": 4, "collected_at_defect": 3,
    }


@pytest.mark.parametrize("kwargs, expected", [
    ({"timeout_enforced": False}, TIMEOUT_BROKEN_STATUS),
    ({"timed_out": True}, "timed_out"),
    ({"collected_at_defect": 0}, "collection_failed"),
    ({"documented": ()}, "no_documented_tests"),
    ({"documented_not_in_universe": ["missing"]}, "documented_mismatch"),
    ({"documented_missing": ["did-not-fail"]}, "documented_mismatch"),
])
def test_exclusion_precedence_never_promotes_unverified_labels(tmp_path, kwargs, expected):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    if "documented" in kwargs:
        bug = bip.BugSpec(**{**bug.__dict__, "documented_failing_tests": kwargs.pop("documented")})
    status, reason = h._defect_status(
        bug, {"collected_at_defect": kwargs.pop("collected_at_defect", 1)},
        kwargs.pop("timed_out", False), kwargs.pop("timeout_enforced", True),
        kwargs.pop("documented_missing", []), kwargs.pop("documented_not_in_universe", []),
    )
    assert status == expected and reason


def test_baseline_cache_is_reused_without_running_suite(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    cached = BaselineProfile(stable_passing=["tests/a::test_x"], n_runs=2, suite_duration=1.25)
    path = h.baseline_path(bug); path.parent.mkdir(parents=True); path.write_text(json.dumps(cached.to_dict()))
    monkeypatch.setattr(h, "run_suite", lambda *a: pytest.fail("cached baseline must be reused"))
    assert h.profile_baseline(bug, bip.profile_for("tqdm"), Path("python")).stable_passing == cached.stable_passing


def test_existing_records_skips_corrupt_lines_and_write_deduplicates_order(tmp_path):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root)
    h.records_path.write_text('{"mutant_id":"bip_tqdm_2","project":"tqdm","bug_id":2}\nnot-json\n')
    assert list(h.existing_records()) == ["bip_tqdm_2"]
    h.write_records([
        {"mutant_id": "b", "project": "z", "bug_id": 2},
        {"mutant_id": "a", "project": "a", "bug_id": 3},
    ])
    rows = [json.loads(line) for line in h.records_path.read_text().splitlines()]
    assert [row["mutant_id"] for row in rows] == ["a", "b"]


def test_harvest_resumes_and_checkpoints_each_new_bug(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root)
    first, second = _bug(root), _bug(root, 2)
    existing = {first.record_id: {"mutant_id": first.record_id, "project": "tqdm", "bug_id": 1,
                                  "status": "env_error"}}
    monkeypatch.setattr(h, "existing_records", lambda: existing.copy())
    monkeypatch.setattr(bip, "load_project", lambda *a: [first, second])
    measured = []
    monkeypatch.setattr(h, "measure", lambda spec, force=False: measured.append(spec.record_id) or {
        "mutant_id": spec.record_id, "project": spec.project, "bug_id": spec.bug_id,
        "status": "interpreter_mismatch", "n_killed": None, "universe_size": None,
    })
    monkeypatch.setattr(h, "metadata_commit", lambda: "feed123")
    monkeypatch.setattr(h, "base_python_version", lambda: "3.11.9")
    summary = h.harvest(["tqdm"])
    assert measured == [second.record_id]
    assert summary["n_records"] == 2 and summary["n_attempted_this_run"] == 1
    assert summary["status_counts"] == {"env_error": 1, "interpreter_mismatch": 1}
    assert len(h.records_path.read_text().splitlines()) == 2


def test_harvest_captures_unhandled_error_as_unlabelled_record(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root); bug = _bug(root)
    monkeypatch.setattr(bip, "load_project", lambda *a: [bug])
    monkeypatch.setattr(h, "measure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(h, "metadata_commit", lambda: "feed123")
    monkeypatch.setattr(h, "base_python_version", lambda: "3.11.9")
    summary = h.harvest(["tqdm"])
    row = json.loads(h.records_path.read_text())
    assert summary["status_counts"] == {"harvest_error": 1}
    assert row["labels_measured"] is False and row["n_killed"] is None
    assert "RuntimeError: boom" in row["exclusion_reason"]


def test_summary_aggregates_only_harvested_records(tmp_path, monkeypatch):
    root = _metadata(tmp_path); h = _harvester(tmp_path, root)
    monkeypatch.setattr(h, "metadata_commit", lambda: "feed123")
    monkeypatch.setattr(h, "base_python_version", lambda: "3.11.9")
    records = {
        "a": {"project": "tqdm", "status": "harvested", "universe_size": 10, "n_killed": 2},
        "b": {"project": "tqdm", "status": "env_error", "universe_size": None, "n_killed": None},
        "c": {"project": "httpie", "status": "harvested", "universe_size": 20, "n_killed": 4},
        "d": {"project": "httpie", "status": "harvested", "universe_size": 10, "n_killed": 2},
    }
    summary = h.summarise(records, ["a", "c"], 1.26, ["tqdm", "httpie"])
    assert summary["n_labelled"] == 3 and summary["projects_with_labels"] == ["httpie", "tqdm"]
    assert summary["per_project"]["tqdm"] == {
        "records": 2, "harvested": 1, "mean_universe_size": 10.0, "mean_n_killed": 2.0
    }
    assert summary["per_project"]["httpie"]["mean_universe_size"] == 15.0
