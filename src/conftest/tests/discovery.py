"""
ConfTest Pytest Discovery Engine.

Discovers regression test files, test classes, functions, and pytest node IDs
using pytest collection in isolated subprocesses and Python AST inspection.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Strict regex pattern for validating pytest node IDs (avoids command injection)
NODE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\./\\]+(::[a-zA-Z0-9_\[\]\-\.\:]+)?$")


def validate_test_node_id(node_id: str) -> bool:
    """Validate that a test node ID contains only safe characters."""
    if not node_id or len(node_id) > 1024:
        return False
    return bool(NODE_ID_PATTERN.match(node_id))


class PytestDiscovery:
    """Discovers pytest test cases across a repository."""

    __test__ = False  # Prevent pytest from treating discovery class as a test suite

    def __init__(self, repo_root: str):
        """
        Initialize test discovery.

        Args:
            repo_root: Root directory of target repository to scan.
        """
        self.repo_root = Path(repo_root).resolve()

    def discover_via_pytest(self, test_dir: Optional[str] = None, timeout: int = 120) -> List[Dict[str, Any]]:
        """
        Discover test cases using `pytest --collect-only -q`.

        The timeout is generous (this repo's own collection imports shap,
        sklearn and lightgbm across ~40 test modules and takes ~40s) because
        what happens on expiry is worse than waiting: the AST fallback below
        cannot see parametrized tests, so a timeout silently shrank this
        repo's suite from 826 collected tests to 610 AST-visible functions
        -- and a SAFE_FULL_SUITE decision then ran 75% of the suite while
        claiming to run all of it.

        Args:
            test_dir: Specific test directory to scan (relative to repo_root).
            timeout: Subprocess timeout in seconds.

        Returns:
            List of discovered test dictionaries with node IDs, paths, and function names.
        """
        target_path = (self.repo_root / test_dir) if test_dir else self.repo_root
        if not target_path.exists():
            logger.warning(f"Target test path does not exist: {target_path}")
            return []

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            # Neutralise the target repo's own addopts (set in its
            # pyproject.toml/pytest.ini): a `-v` there cancels our `-q` and
            # pytest prints a collection tree instead of one node ID per
            # line, yielding 0 parsed tests and a silent AST fallback.
            # addopts carrying --cov or --doctest-modules would likewise
            # abort collection when the plugin is absent.
            "-o",
            "addopts=",
            # Pin rootdir to the repository under discovery. Without it pytest
            # resolves rootdir from ambient config files, and a pyproject.toml
            # sitting above the checkout makes it print node IDs relative to
            # that ancestor (e.g. tests/sample_suite/tests/test_auth.py when
            # the checkout IS tests/sample_suite). The executor later runs
            # pytest with cwd=repo_root, where those IDs do not resolve, and
            # the whole run aborts with usage exit code 4 before a single
            # test executes -- while passing on any machine where no ancestor
            # config happens to exist.
            "--rootdir",
            str(self.repo_root),
            str(target_path),
        ]
        try:
            # An inherited PYTEST_ADDOPTS would be prepended to this
            # collection run's argv too: discovery nested inside a pytest
            # that an outer executor launched with hundreds of node IDs in
            # PYTEST_ADDOPTS would collect those alongside (or instead of)
            # the target, and the caller would rank a foreign suite.
            clean_env = dict(os.environ)
            clean_env.pop("PYTEST_ADDOPTS", None)
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=clean_env,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Pytest collection timed out after {timeout}s on {target_path}")
            return self.discover_via_ast(test_dir)

        test_cases: List[Dict[str, Any]] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            # Pytest -q collect-only outputs node IDs on each line, ending with summary
            if "::" in line and not line.startswith("<") and not line.endswith("collected"):
                node_id = line.split(" ")[0].replace("\\", "/")
                if validate_test_node_id(node_id):
                    parts = node_id.split("::")
                    file_path = parts[0]
                    func_name = parts[-1]
                    test_cases.append({
                        "test_id": node_id,
                        "test_path": file_path,
                        "test_function": func_name,
                        "framework": "pytest",
                    })

        if not test_cases:
            logger.info("Pytest subprocess collection yielded 0 tests. Falling back to AST scanning.")
            return self.discover_via_ast(test_dir)

        return test_cases

    def discover_via_ast(self, test_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fallback static discovery using Python's `ast` parser.
        Finds all files named `test_*.py` or `*_test.py` and functions starting with `test_`.
        """
        search_root = (self.repo_root / test_dir) if test_dir else self.repo_root
        test_cases: List[Dict[str, Any]] = []

        for root, _, files in os.walk(search_root):
            for file in files:
                # Parenthesised: without them precedence made every *_test.py
                # file match regardless of the test_ prefix rule.
                if (file.startswith("test_") or file.endswith("_test.py")) and file.endswith(".py"):
                    full_path = Path(root) / file
                    try:
                        rel_path = full_path.relative_to(self.repo_root).as_posix()
                        with open(full_path, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=str(full_path))

                        # Only iterate top-level statements. ast.walk descends
                        # into class bodies too, so a test method inside a
                        # Test class was emitted twice: once as the correct
                        # file::Class::method ID and once as a phantom
                        # file::method ID that pytest cannot resolve -- which
                        # aborts the whole run with usage exit code 4 when the
                        # IDs are passed via PYTEST_ADDOPTS.
                        for node in tree.body:
                            if isinstance(node, ast.ClassDef) and (
                                node.name.startswith("Test") or node.name.endswith("Test")
                            ):
                                for method in node.body:
                                    if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                                        node_id = f"{rel_path}::{node.name}::{method.name}"
                                        test_cases.append({
                                            "test_id": node_id,
                                            "test_path": rel_path,
                                            "test_function": method.name,
                                            "framework": "pytest",
                                        })
                            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                                node_id = f"{rel_path}::{node.name}"
                                test_cases.append({
                                    "test_id": node_id,
                                    "test_path": rel_path,
                                    "test_function": node.name,
                                    "framework": "pytest",
                                })
                    except Exception as exc:
                        logger.warning(f"AST parse error in {full_path}: {exc}")

        return test_cases
