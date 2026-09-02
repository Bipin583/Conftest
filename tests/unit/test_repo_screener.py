"""
Unit tests for the repository screener.

Each test here corresponds to a bug found while screening real repositories.
The screener decides which repos the mutation harvest runs against, so a wrong
verdict either wastes hours of compute or silently drops a usable subject.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from screen_repos import (  # noqa: E402
    MAX_SUITE_SECONDS,
    MAX_TESTS,
    MIN_TESTS,
    PYTEST_EXIT_MEANING,
    count_py_files,
    detect_c_extensions,
    locate_source_and_tests,
    parse_junit,
)


# --------------------------------------------------------------------------
# C extension detection
# --------------------------------------------------------------------------

def test_detects_real_c_extension(tmp_path):
    (tmp_path / "setup.py").write_text(
        "from setuptools import setup, Extension\n"
        "setup(ext_modules=[Extension('m', ['m.c'])])\n"
    )
    assert detect_c_extensions(tmp_path) is True


def test_detects_cython_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "fast.pyx").write_text("def f(): pass\n")
    assert detect_c_extensions(tmp_path) is True


def test_sample_header_in_examples_is_not_a_build_input(tmp_path):
    """
    pyparsing ships examples/snmp_api.h as sample data to be parsed.

    Treating it as a build input rejected a pure-Python library outright.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\nname='pyparsing'\n")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "snmp_api.h").write_text("#define FOO 1\n")
    assert detect_c_extensions(tmp_path) is False


def test_header_in_docs_is_not_a_build_input(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "snippet.c").write_text("int main(){}\n")
    assert detect_c_extensions(tmp_path) is False


def test_pure_python_project_is_pure(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    pkg = tmp_path / "x"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VERSION = '1.0'\n")
    assert detect_c_extensions(tmp_path) is False


# --------------------------------------------------------------------------
# Layout detection
# --------------------------------------------------------------------------

def _make_pkg(root: Path, name: str) -> Path:
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    return pkg


def test_src_layout_is_detected(tmp_path):
    _make_pkg(tmp_path / "src", "cachetools")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_cache.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert source_dirs == ["src"]
    assert test_dirs == ["tests"]


def test_flat_package_layout_is_detected(tmp_path):
    _make_pkg(tmp_path, "tabulate")
    tests = tmp_path / "test"
    tests.mkdir()
    (tests / "test_output.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert source_dirs == ["tabulate"]
    assert test_dirs == ["test"]


def test_single_file_test_suite_is_detected(tmp_path):
    """
    inflection and schedule ship one top-level test module, not a package.

    Requiring a tests/ DIRECTORY rejected both of them incorrectly.
    """
    _make_pkg(tmp_path, "inflection")
    (tmp_path / "test_inflection.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert source_dirs == ["inflection"]
    assert test_dirs == ["test_inflection.py"]


def test_bare_test_py_is_not_accepted(tmp_path):
    """
    python-slugify ships `test.py`, which the default pytest `python_files`
    patterns (test_*.py, *_test.py) do NOT collect. Accepting it would mean
    screening a suite the harness cannot actually run.
    """
    _make_pkg(tmp_path, "slugify")
    (tmp_path / "test.py").write_text("def test_a(): pass\n")

    _, test_dirs = locate_source_and_tests(tmp_path)
    assert test_dirs == []


def test_tests_nested_inside_package_are_found(tmp_path):
    """cerberus keeps its suite at cerberus/tests/."""
    pkg = _make_pkg(tmp_path, "cerberus")
    nested = pkg / "tests"
    nested.mkdir()
    (nested / "test_validation.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert "cerberus" in source_dirs
    assert test_dirs == ["cerberus/tests"]


def test_build_and_vcs_dirs_are_ignored(tmp_path):
    _make_pkg(tmp_path, "real")
    for noise in (".git", "build", "dist", ".tox", "docs", "examples"):
        _make_pkg(tmp_path, noise)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_a(): pass\n")

    source_dirs, _ = locate_source_and_tests(tmp_path)
    assert source_dirs == ["real"]


# --------------------------------------------------------------------------
# File counting
# --------------------------------------------------------------------------

def test_counts_source_files_excluding_tests(tmp_path):
    pkg = _make_pkg(tmp_path, "app")
    (pkg / "core.py").write_text("")
    (pkg / "util.py").write_text("")
    (pkg / "test_core.py").write_text("")

    # __init__.py + core.py + util.py
    assert count_py_files(tmp_path, ["app"], want_tests=False) == 3
    assert count_py_files(tmp_path, ["app"], want_tests=True) == 1


def test_counts_a_single_file_location(tmp_path):
    """A test location may be one module; counting must not silently skip it."""
    (tmp_path / "test_inflection.py").write_text("def test_a(): pass\n")
    assert count_py_files(tmp_path, ["test_inflection.py"], want_tests=True) == 1


def test_pycache_is_excluded(tmp_path):
    pkg = _make_pkg(tmp_path, "app")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "core.py").write_text("")
    assert count_py_files(tmp_path, ["app"], want_tests=False) == 1


# --------------------------------------------------------------------------
# JUnit parsing
# --------------------------------------------------------------------------

def test_parses_statuses(tmp_path):
    report = tmp_path / "r.xml"
    report.write_text(
        "<testsuites><testsuite>"
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_ok"/>'
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_bad">'
        '<failure message="boom"/></testcase>'
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_err">'
        '<error message="raised"/></testcase>'
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_skip">'
        "<skipped/></testcase>"
        "</testsuite></testsuites>"
    )
    outcomes = parse_junit(report)
    assert outcomes["tests/test_a.py::test_ok"] == "PASSED"
    assert outcomes["tests/test_a.py::test_bad"] == "FAILED"
    assert outcomes["tests/test_a.py::test_err"] == "ERROR"
    assert outcomes["tests/test_a.py::test_skip"] == "SKIPPED"


def test_falls_back_to_classname_when_file_attr_absent(tmp_path):
    report = tmp_path / "r.xml"
    report.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_payment" name="test_fee"/>'
        "</testsuite></testsuites>"
    )
    assert "tests/test_payment::test_fee" in parse_junit(report)


def test_missing_report_returns_empty(tmp_path):
    assert parse_junit(tmp_path / "nope.xml") == {}


def test_malformed_report_returns_empty(tmp_path):
    report = tmp_path / "r.xml"
    report.write_text("<testsuites><unclosed>")
    assert parse_junit(report) == {}


# --------------------------------------------------------------------------
# Exit-code interpretation
# --------------------------------------------------------------------------

def test_interrupted_collection_is_distinguished_from_a_small_suite():
    """
    pyparsing reported 1 test because collection aborted after one item.
    Counting rows without checking the exit code called a 2156-test suite tiny.
    """
    assert PYTEST_EXIT_MEANING[2] == "collection_interrupted"
    assert PYTEST_EXIT_MEANING[5] == "no_tests_collected"
    assert PYTEST_EXIT_MEANING[4] == "pytest_usage_error"
    assert PYTEST_EXIT_MEANING[124] == "suite_timeout"


def test_normal_exit_codes_are_not_treated_as_failures():
    """0 = all passed, 1 = some tests failed. Neither invalidates the run."""
    assert 0 not in PYTEST_EXIT_MEANING
    assert 1 not in PYTEST_EXIT_MEANING


# --------------------------------------------------------------------------
# Threshold sanity
# --------------------------------------------------------------------------

def test_thresholds_are_coherent():
    assert MIN_TESTS < MAX_TESTS
    assert MAX_SUITE_SECONDS > 0
    # A suite at the ceiling must still allow a 250-mutant harvest overnight.
    assert MAX_SUITE_SECONDS * 250 / 3600 < 7
