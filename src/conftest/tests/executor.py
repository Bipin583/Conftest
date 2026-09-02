"""
ConfTest Safe Isolated Test Execution Engine.

Executes full test suites or selected test subsets in an isolated subprocess
with strict timeouts, input sanitization against shell injection, and JUnit XML result parsing.
"""

import os
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
    ):
        self.exit_code = exit_code
        self.total_duration = total_duration
        self.test_runs = test_runs
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out

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
            "test_runs": self.test_runs,
        }


class SafeTestExecutor:
    """Executes pytest suites with subprocess isolation and timeout protection."""

    __test__ = False

    def __init__(self, repo_root: str, default_timeout: int = 60):
        """
        Initialize the executor.

        Args:
            repo_root: Root directory of the repository containing tests.
            default_timeout: Subprocess execution timeout in seconds.
        """
        self.repo_root = Path(repo_root).resolve()
        self.default_timeout = default_timeout

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
            sys.executable,
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

        if sanitized_targets:
            cmd.extend(sanitized_targets)
        elif test_dir:
            cmd.append(test_dir)

        start_time = time.time()
        timed_out = False
        exit_code = 1
        stdout = ""
        stderr = ""

        try:
            logger.info(f"Running pytest ({len(sanitized_targets) if sanitized_targets else 'ALL'} tests, timeout: {exec_timeout}s)...")
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=exec_timeout,
            )
            exit_code = proc.exitcode if hasattr(proc, 'exitcode') else proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124  # Standard timeout exit code
            stdout = exc.stdout or ""
            stderr = exc.stderr or f"Execution timed out after {exec_timeout}s."
            logger.error(f"Test execution timed out after {exec_timeout}s.")
        except Exception as exc:
            logger.error(f"Test execution failed with error: {exc}")
            stderr = str(exc)
        finally:
            total_duration = max(0.001, time.time() - start_time)

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
        )
