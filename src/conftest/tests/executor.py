"""
ConfTest Safe Isolated Test Execution Engine.

Executes full test suites or selected test subsets in an isolated subprocess
with input sanitization against shell injection and JUnit XML result parsing.

Timeouts are enforced by killing the whole process tree, not just the direct
child, and the result carries `timeout_enforced` so a caller can tell a
timeout that held from one that did not. That distinction is not theoretical:
before the tree kill existed, a 180s timeout on this project measured 28,270s.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from conftest.logging_config import get_logger
from conftest.tests.discovery import validate_test_node_id

logger = get_logger(__name__)

# How long to wait for a killed pytest tree to actually be gone before giving up
# on it and saying so. Never wait unbounded: that is the defect this replaced.
KILL_GRACE_SECONDS = 30
# A run is allowed to exceed its timeout by the kill grace plus a little; beyond
# that the timeout did not hold, and the result has to admit it.
BROKEN_TIMEOUT_SLACK_SECONDS = KILL_GRACE_SECONDS + 30


class PytestExecutionResult:
    """Structured container for test execution outcomes."""

    __test__ = False

    def __init__(
        self,
        exit_code: int,
        total_duration: float,
        test_runs: List[Dict[str, Any]],
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        timeout_enforced: bool = True,
    ):
        self.exit_code = exit_code
        self.total_duration = total_duration
        self.test_runs = test_runs
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.timeout_enforced = timeout_enforced

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] == "PASSED")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] in ("FAILED", "ERROR"))

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] == "SKIPPED")

    @property
    def total_count(self) -> int:
        return len(self.test_runs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "total_duration": round(self.total_duration, 3),
            "total_tests": self.total_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "timed_out": self.timed_out,
            "timeout_enforced": self.timeout_enforced,
            "test_runs": self.test_runs,
        }


class SafeTestExecutor:
    """Executes pytest suites with subprocess isolation and timeout protection."""

    __test__ = False

    def __init__(
        self,
        repo_root: str,
        default_timeout: int = 60,
        python_executable: Optional[str] = None,
    ):
        """
        Initialize the executor.

        Args:
            repo_root: Root directory of the repository containing tests.
            default_timeout: Subprocess execution timeout in seconds.
            python_executable: Interpreter that runs pytest. Defaults to the
                current one, which is right when the suite under test belongs to
                this project. It is wrong for a foreign checkout: that suite
                must run in the environment where the checkout is installed, or
                `import <package>` resolves to whatever copy the ambient
                interpreter happens to have. Measured on this project's own
                machine: `sqlparse` is present in the ambient site-packages, so
                a mutation harvest that ran there could have imported the
                unmutated release copy and recorded every test as passing --
                fabricated labels by a route no grep for `random` would find.
        """
        self.repo_root = Path(repo_root).resolve()
        self.default_timeout = default_timeout

        if python_executable is None:
            self.python_executable = sys.executable
        else:
            candidate = Path(python_executable)
            if not candidate.is_file():
                # Silently falling back to sys.executable is the failure mode
                # this argument exists to prevent, so a bad path is fatal.
                raise FileNotFoundError(
                    f"python_executable does not exist: {candidate}. "
                    f"Create the environment first (scripts/screen_repos.py builds "
                    f"one per subject repository) rather than running the suite "
                    f"under an interpreter that may not have it installed."
                )
            self.python_executable = str(candidate.resolve())

    def _parse_junit_xml(self, xml_path: str) -> List[Dict[str, Any]]:
        """Parse JUnit XML report to extract per-test execution status and durations."""
        runs: List[Dict[str, Any]] = []
        if not os.path.exists(xml_path):
            return runs

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Iterate through all testcase elements in testsuites/testsuite
            for tc in root.iter("testcase"):
                classname = tc.attrib.get("classname", "")
                name = tc.attrib.get("name", "")
                file_attr = tc.attrib.get("file", "")
                duration = float(tc.attrib.get("time", 0.0))

                status = "PASSED"
                failure_msg = None

                failure = tc.find("failure")
                error = tc.find("error")
                skipped = tc.find("skipped")

                if failure is not None:
                    status = "FAILED"
                    failure_msg = failure.attrib.get("message", "") or failure.text
                elif error is not None:
                    status = "ERROR"
                    failure_msg = error.attrib.get("message", "") or error.text
                elif skipped is not None:
                    status = "SKIPPED"
                    failure_msg = skipped.attrib.get("message", "")

                # Construct canonical node ID: file::function or classname::function
                if file_attr:
                    clean_file = file_attr.replace("\\", "/")
                    node_id = f"{clean_file}::{name}"
                else:
                    clean_class = classname.replace(".", "/")
                    node_id = f"{clean_class}::{name}"

                runs.append({
                    "test_id": node_id,
                    "status": status,
                    "duration": duration,
                    "failure_message": failure_msg,
                })
        except Exception as exc:
            logger.error(f"Failed parsing JUnit XML ({xml_path}): {exc}")

        return runs

    def _kill_process_tree(self, proc: subprocess.Popen) -> bool:
        """
        Kill a timed-out pytest and everything it spawned. False if it survived.

        Killing the direct child is not enough. pytest's own children outlive
        it, and a surviving grandchild is not merely wasted CPU: it holds the
        checkout open and can keep writing to it, which is how a mutant run
        ends up blamed for damage a previous one did. Windows has no process
        group to signal, so the tree is walked by pid with taskkill; on POSIX
        the child was given its own session and the group is signalled.
        """
        try:
            proc.kill()
        except Exception as exc:  # already dead, or unkillable
            logger.warning(f"Direct kill of pytest {proc.pid} failed: {exc}")

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=KILL_GRACE_SECONDS,
                )
            except Exception as exc:
                logger.error(f"taskkill could not reach the pytest tree {proc.pid}: {exc}")
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception as exc:
                logger.error(f"Could not signal the pytest group for {proc.pid}: {exc}")

        try:
            proc.wait(timeout=KILL_GRACE_SECONDS)
            return True
        except subprocess.TimeoutExpired:
            logger.error(
                f"pytest {proc.pid} outlived its kill by {KILL_GRACE_SECONDS}s; "
                f"abandoning it. Anything it writes to the checkout from here is "
                f"drift, and the next mutant will be measured against it."
            )
            return False

    def _read_capture(self, path: str) -> str:
        """Read a capture file back, tolerating a subprocess that never wrote it."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""

    def run_tests(
        self,
        test_node_ids: Optional[List[str]] = None,
        test_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> PytestExecutionResult:
        """
        Execute full test suite or a selected subset of test node IDs.

        Args:
            test_node_ids: Optional list of validated test node IDs to run.
            test_dir: Optional subdirectory containing tests if running full suite.
            timeout: Subprocess timeout in seconds.

        Returns:
            PytestExecutionResult with parsed test outcomes and durations.
        """
        exec_timeout = timeout or self.default_timeout
        sanitized_targets: List[str] = []

        if test_node_ids is not None:
            # Validate every node ID to eliminate command injection risk
            for node_id in test_node_ids:
                clean_id = node_id.strip()
                if validate_test_node_id(clean_id):
                    sanitized_targets.append(clean_id)
                else:
                    logger.warning(f"Rejected invalid or unsafe test node ID: {node_id}")

            if not sanitized_targets:
                logger.warning("No valid test targets provided for execution.")
                return PytestExecutionResult(
                    exit_code=0, total_duration=0.0, test_runs=[], stdout="No valid tests to run."
                )

        # Create temporary JUnit XML report path
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
            xml_report_path = tmp_file.name

        cmd = [
            self.python_executable,
            "-m",
            "pytest",
            f"--junitxml={xml_report_path}",
            "-v",
            # Neutralise the target repo's own addopts. Many projects put
            # `--cov=... --doctest-modules` there, which (a) aborts the run
            # outright when the plugin is absent and (b) adds coverage
            # instrumentation overhead to every one of the hundreds of suite
            # executions a mutation harvest performs. Screening uses the same
            # flags, so measured timings match harvest timings.
            "-o",
            "addopts=",
            # Never write .pytest_cache into the repo under mutation; a cache
            # surviving between runs would break byte-exact restore checks.
            "-p",
            "no:cacheprovider",
        ]

        # A full-suite abstention can pass hundreds of node IDs. Putting them
        # on the command line overflows the OS argv limit (Windows
        # CreateProcess fails outright around 32K chars), so they travel via
        # the PYTEST_ADDOPTS environment variable, which pytest prepends to
        # argv without any command-line length limit. This is independent of
        # ini-file `addopts`, so the `-o addopts=` neutraliser below still
        # suppresses the target repo's own addopts.
        env = dict(os.environ)
        if sanitized_targets:
            if len(sanitized_targets) <= 20:
                cmd.extend(sanitized_targets)
            else:
                env["PYTEST_ADDOPTS"] = " ".join(sanitized_targets)
        elif test_dir:
            cmd.append(test_dir)

        start_time = time.time()
        timed_out = False
        timeout_enforced = True
        exit_code = 1
        stdout = ""
        stderr = ""

        # Capture to files, not pipes. A pipe is a shared handle: every
        # grandchild inherits it and keeps it open, and subprocess.run drains
        # the pipe *after* killing the direct child, with no timeout on the
        # drain. Measured cost of that on 2026-09-03: mut_42bfc7db58b0 forced
        # sqlparse's CLI to --inplace, a child kept the handle, and a 180s
        # timeout became 28,270s -- 7h51m of a mutated tree live on disk, in
        # which the suite truncated one of its own tracked fixtures.
        out_path = xml_report_path + ".out"
        err_path = xml_report_path + ".err"
        session_kwargs = {} if os.name == "nt" else {"start_new_session": True}

        try:
            logger.info(
                f"Running pytest ({len(sanitized_targets) if sanitized_targets else 'ALL'} "
                f"tests, timeout: {exec_timeout}s)..."
            )
            with open(out_path, "w", encoding="utf-8", errors="replace") as out_fh:
                with open(err_path, "w", encoding="utf-8", errors="replace") as err_fh:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(self.repo_root),
                        stdout=out_fh,
                        stderr=err_fh,
                        # A suite that reads stdin must get EOF, not block on a
                        # terminal that will never answer it.
                        stdin=subprocess.DEVNULL,
                        env=env,
                        **session_kwargs,
                    )
                    try:
                        exit_code = proc.wait(timeout=exec_timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        exit_code = 124  # Standard timeout exit code
                        logger.error(f"Test execution timed out after {exec_timeout}s.")
                        timeout_enforced = self._kill_process_tree(proc)
            stdout = self._read_capture(out_path)
            stderr = self._read_capture(err_path)
            if timed_out and not stderr:
                stderr = f"Execution timed out after {exec_timeout}s."
        except Exception as exc:
            logger.error(f"Test execution failed with error: {exc}")
            stderr = str(exc)
        finally:
            total_duration = max(0.001, time.time() - start_time)
            for path in (out_path, err_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        if total_duration > exec_timeout + BROKEN_TIMEOUT_SLACK_SECONDS:
            # Report the ceiling that actually held, not the one that was asked
            # for: a harvest costed in suite-timeouts is costed wrongly here.
            timeout_enforced = False
            logger.error(
                f"The {exec_timeout}s timeout did not hold: the run took "
                f"{total_duration:.0f}s, and the tree under test was live for all of it."
            )

        # Parse test outcomes from generated JUnit XML
        test_runs = self._parse_junit_xml(xml_report_path)

        # Clean up temporary XML file
        try:
            if os.path.exists(xml_report_path):
                os.remove(xml_report_path)
        except Exception:
            pass

        return PytestExecutionResult(
            exit_code=exit_code,
            total_duration=total_duration,
            test_runs=test_runs,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            timeout_enforced=timeout_enforced,
        )
