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
    COMMON_TEST_PLUGINS,
    STAGE2_NOTE,
    TOOLING_EXTRA_NAMES,
    DEP_FEATURE_NAMES,
    MAX_HARVEST_HOURS_PER_REPO,
    MAX_SUITE_SECONDS,
    MAX_TESTS,
    MIN_TESTS,
    MIN_VARYING_DEP_FEATURES,
    NEUTRAL_PYTEST_INI,
    PLANNED_MUTANTS_PER_REPO,
    PYTEST_EXIT_MEANING,
    collection_diagnostic,
    count_py_files,
    declared_extras,
    declared_test_groups,
    detect_c_extensions,
    drop_stage2_notes,
    isolate_workspace_from_project_config,
    _holds_collectible_tests,
    list_py_files,
    locate_source_and_tests,
    measure_dependency_variance,
    parse_junit,
    read_pyproject,
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


def test_a_vendored_fixture_directory_named_tests_is_not_a_suite(tmp_path):
    """
    jsonschema vendors the JSON-Schema-Test-Suite at json/tests/ -- thousands of
    .json fixtures, zero .py files. Taking the first rglob("tests") hit selected
    it and reported no_test_files, losing the real suite at jsonschema/tests/.
    """
    pkg = _make_pkg(tmp_path, "jsonschema")
    real = pkg / "tests"
    real.mkdir()
    (real / "test_validators.py").write_text("def test_a(): pass\n")

    vendored = tmp_path / "json" / "tests"
    vendored.mkdir(parents=True)
    (vendored / "type.json").write_text("[]\n")

    _, test_dirs = locate_source_and_tests(tmp_path)
    assert test_dirs == ["jsonschema/tests"]


def test_collectible_predicate_matches_pytest_default_patterns(tmp_path):
    empty = tmp_path / "fixtures"
    empty.mkdir()
    (empty / "case.json").write_text("{}\n")
    assert _holds_collectible_tests(empty) is False

    (empty / "helpers.py").write_text("X = 1\n")
    assert _holds_collectible_tests(empty) is False   # not a test_*/*_test name

    (empty / "suite_test.py").write_text("def test_a(): pass\n")
    assert _holds_collectible_tests(empty) is True


def test_the_shallowest_real_suite_wins(tmp_path):
    _make_pkg(tmp_path, "app")
    deep = tmp_path / "app" / "sub" / "tests"
    deep.mkdir(parents=True)
    (deep / "test_deep.py").write_text("def test_a(): pass\n")
    shallow = tmp_path / "app" / "tests"
    shallow.mkdir()
    (shallow / "test_shallow.py").write_text("def test_a(): pass\n")

    _, test_dirs = locate_source_and_tests(tmp_path)
    assert test_dirs == ["app/tests"]


def test_a_src_layout_excludes_sibling_tooling_packages(tmp_path):
    """
    packaging ships src/ plus a tasks/ package of release automation that carries
    an __init__.py. Mutating it spends budget on code the suite never imports, so
    every such mutant kills nothing and contributes only all-negative rows.
    """
    _make_pkg(tmp_path / "src", "packaging")
    _make_pkg(tmp_path, "tasks")
    (tmp_path / "tasks" / "licenses.py").write_text("def run(): pass\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_version.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert source_dirs == ["src"]
    assert test_dirs == ["tests"]


def test_a_flat_layout_still_accepts_its_top_level_package(tmp_path):
    """The src/ rule must not break repos that have no src/ at all."""
    _make_pkg(tmp_path, "cerberus")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_a(): pass\n")
    source_dirs, _ = locate_source_and_tests(tmp_path)
    assert source_dirs == ["cerberus"]


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


# --------------------------------------------------------------------------
# Isolation from this project's own pytest config
# --------------------------------------------------------------------------

def test_neutral_ini_is_written_to_the_checkout_parent(tmp_path):
    """
    Measured, not hypothetical: cachetools and sqlparse ship no pytest section,
    so pytest walked up and adopted ConfTest's own config -- handing them our
    pythonpath=["src"] and asyncio_mode="auto".
    """
    marker = isolate_workspace_from_project_config(tmp_path)
    assert marker == tmp_path / "pytest.ini"
    assert marker.read_text(encoding="utf-8") == NEUTRAL_PYTEST_INI


def test_neutral_ini_declares_a_pytest_section_and_nothing_else():
    """An empty [pytest] section is what ends the upward search harmlessly."""
    assert NEUTRAL_PYTEST_INI.strip() == "[pytest]"


def test_isolation_is_idempotent_and_repairs_a_edited_marker(tmp_path):
    marker = tmp_path / "pytest.ini"
    marker.write_text("[pytest]\naddopts = --cov\n", encoding="utf-8")
    isolate_workspace_from_project_config(tmp_path)
    assert marker.read_text(encoding="utf-8") == NEUTRAL_PYTEST_INI
    isolate_workspace_from_project_config(tmp_path)
    assert marker.read_text(encoding="utf-8") == NEUTRAL_PYTEST_INI


def test_isolation_never_writes_inside_a_checkout(tmp_path):
    """
    Nothing inside a subject repo may be touched, or byte-exact restoration
    and the G3 clean-tree check both break.
    """
    checkout = tmp_path / "cachetools"
    checkout.mkdir()
    isolate_workspace_from_project_config(tmp_path)
    assert list(checkout.iterdir()) == []


# --------------------------------------------------------------------------
# Rejection diagnostics
# --------------------------------------------------------------------------

def test_collection_diagnostic_keeps_the_error_lines():
    """
    A bare "collection_interrupted" is the opacity that let the partial-report
    bug hide. cerberus was only missing pytest-benchmark.
    """
    stdout = """\ncollecting ...\nERROR cerberus/benchmarks/test_overall_performance_1.py\nE   pytest.PytestUnknownMarkWarning: Unknown pytest.mark.benchmark\n!!!! Interrupted: 2 errors during collection !!!!\n247 tests collected, 2 errors in 0.52s\n"""
    diagnostic = collection_diagnostic(stdout, "")
    assert "test_overall_performance_1.py" in diagnostic
    assert "benchmark" in diagnostic
    assert "Interrupted" in diagnostic
    assert "collecting ..." not in diagnostic

def test_collection_diagnostic_is_empty_when_nothing_errored():
    assert collection_diagnostic("333 passed in 5.68s", "") == ""


def test_collection_diagnostic_reads_stderr_too():
    assert "ERROR boom.py" in collection_diagnostic("", "ERROR boom.py")


def test_testing_directory_is_recognised_as_tests(tmp_path):
    """pluggy keeps its suite in testing/, and was rejected as having none."""
    _make_pkg(tmp_path, "pluggy")
    testing = tmp_path / "testing"
    testing.mkdir()
    (testing / "test_hooks.py").write_text("def test_a(): pass\n")

    source_dirs, test_dirs = locate_source_and_tests(tmp_path)
    assert source_dirs == ["pluggy"]
    assert test_dirs == ["testing"]


# --------------------------------------------------------------------------
# Cost gate
# --------------------------------------------------------------------------

def test_suite_ceiling_is_derived_from_the_harvest_budget():
    """
    The gate must bound the quantity that actually costs money: a full suite
    run per mutant. Capping test COUNT rejected validators (895 tests, 4.96s,
    0.34h) while admitting sortedcontainers (296 tests, 11.58s, 0.80h).
    """
    assert MAX_SUITE_SECONDS == MAX_HARVEST_HOURS_PER_REPO * 3600 / PLANNED_MUTANTS_PER_REPO
    # A repo at the ceiling costs exactly the budget, not multiples of it.
    assert MAX_SUITE_SECONDS * PLANNED_MUTANTS_PER_REPO / 3600 == MAX_HARVEST_HOURS_PER_REPO


def test_the_previously_measured_repos_land_on_the_right_side_of_the_gate():
    """Regression on real measurements, not hypotheticals."""
    def hours(seconds):
        return seconds * PLANNED_MUTANTS_PER_REPO / 3600

    assert hours(4.96) < MAX_HARVEST_HOURS_PER_REPO      # validators, was rejected
    assert hours(11.07) < MAX_HARVEST_HOURS_PER_REPO     # humanize, was rejected
    assert hours(11.58) < MAX_HARVEST_HOURS_PER_REPO     # sortedcontainers, accepted
    assert hours(90.0) > MAX_HARVEST_HOURS_PER_REPO      # the old ceiling: 6.25h


def test_test_count_ceiling_bounds_rows_not_cost():
    """MAX_TESTS caps rows per repo (mutants x universe), so it can be generous."""
    assert MAX_TESTS > 600
    assert MAX_TESTS * PLANNED_MUTANTS_PER_REPO <= 500_000


def test_common_plugins_are_additive_only():
    """
    requirements-dev.txt is skipped because a stale pytest-flake8 registers a
    pytest_collect_file hook and aborts the run. Nothing here may do that.
    """
    assert "pytest-flake8" not in COMMON_TEST_PLUGINS
    assert "pytest-black" not in COMMON_TEST_PLUGINS
    assert "pytest-benchmark" in COMMON_TEST_PLUGINS   # cerberus needs it to import


# --------------------------------------------------------------------------
# Stage 3: is test selection a non-trivial task here?
# --------------------------------------------------------------------------

def _two_module_repo(root: Path) -> None:
    """A package where each test file imports a different module."""
    pkg = _make_pkg(root, "app")
    (pkg / "alpha.py").write_text("def a(): return 1\n")
    (pkg / "beta.py").write_text("def b(): return 2\n")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_alpha.py").write_text("from app.alpha import a\ndef test_a(): assert a()\n")
    (tests / "test_beta.py").write_text("from app.beta import b\ndef test_b(): assert b()\n")


def test_lists_source_and_test_files_separately(tmp_path):
    _two_module_repo(tmp_path)
    sources = list_py_files(tmp_path, ["app"], want_tests=False)
    tests = list_py_files(tmp_path, ["tests"], want_tests=True)
    assert sources == ["app/__init__.py", "app/alpha.py", "app/beta.py"]
    assert tests == ["tests/test_alpha.py", "tests/test_beta.py"]


def test_a_single_module_repo_has_no_dependency_variance(tmp_path):
    """
    Measured on parse and inflection: one changeable file means one dependency
    relationship, repeated for every row. Selection has nothing to select on.
    """
    pkg = _make_pkg(tmp_path, "only")
    (pkg / "__init__.py").write_text("def f(): return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_only.py").write_text("from only import f\ndef test_f(): assert f()\n")

    varying, n_vectors = measure_dependency_variance(tmp_path, ["only"], ["tests"])
    assert varying == []
    assert n_vectors == 1


def test_a_multi_module_repo_varies_its_dependency_features(tmp_path):
    _two_module_repo(tmp_path)
    varying, n_vectors = measure_dependency_variance(tmp_path, ["app"], ["tests"])
    assert len(varying) >= 1
    assert n_vectors > 1
    assert set(varying).issubset(set(DEP_FEATURE_NAMES))


def test_dependency_variance_is_empty_when_a_side_is_missing(tmp_path):
    _two_module_repo(tmp_path)
    assert measure_dependency_variance(tmp_path, ["app"], ["nonexistent"]) == ([], 0)
    assert measure_dependency_variance(tmp_path, ["nonexistent"], ["tests"]) == ([], 0)


def test_structural_threshold_demands_real_variety():
    """
    Below half the dependency features varying, the model can only learn a
    per-test fragility prior, which is not test selection.
    """
    assert 1 < MIN_VARYING_DEP_FEATURES <= len(DEP_FEATURE_NAMES) // 2 + 1


# --------------------------------------------------------------------------
# Declared dependencies
#
# An optional dependency that is absent does not fail a test, it SKIPS it, and
# a skipped test never enters the label universe. Installing only pytest
# therefore silently shrinks the dataset instead of erroring: pyjwt ran 221 of
# its 369 tests without `crypto`, and 17 of validators' tests failed on a clean
# checkout for want of `crypto-eth-addresses`.
# --------------------------------------------------------------------------

def _write_pyproject(repo: Path, body: str) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text(body, encoding="utf-8")
    return repo


def test_functional_extras_are_installed_and_tooling_is_not(tmp_path):
    repo = _write_pyproject(tmp_path / "repo", """
[project]
name = "demo"

[project.optional-dependencies]
crypto = ["cryptography>=3.4"]
widechars = ["wcwidth"]
dev = ["build"]
doc = ["sphinx", "furo"]
""")

    assert declared_extras(repo) == ["crypto", "widechars"]


def test_a_hyphenated_functional_extra_survives(tmp_path):
    """
    validators declares `crypto-eth-addresses`. A compound name is judged by
    its leading word, so that survives while `dev-docs` does not.
    """
    repo = _write_pyproject(tmp_path / "repo", """
[project]
name = "demo"

[project.optional-dependencies]
crypto-eth-addresses = ["eth-hash[pycryptodome]>=0.7.0"]
dev-docs = ["sphinx"]
""")

    assert declared_extras(repo) == ["crypto-eth-addresses"]


def test_extras_are_matched_case_and_separator_insensitively(tmp_path):
    repo = _write_pyproject(tmp_path / "repo", """
[project]
name = "demo"

[project.optional-dependencies]
Dev = ["build"]
type_checking = ["mypy"]
re2 = ["google-re2"]
""")

    # `type_checking` normalises to `type-checking`; its leading word is
    # tooling, so it goes. Matching every part instead let this one through.
    assert declared_extras(repo) == ["re2"]


def test_extras_declared_in_setup_cfg_are_found(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "setup.cfg").write_text("""
[options.extras_require]
speedups = ndg-httpsclient
lint = flake8
""", encoding="utf-8")

    assert declared_extras(repo) == ["speedups"]


def test_a_repo_declaring_nothing_yields_no_extras(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    assert declared_extras(repo) == []
    assert declared_test_groups(repo) == []


def test_test_dependency_groups_are_installed_but_dev_groups_are_not(tmp_path):
    """
    PEP 735 groups are unreachable through `.[name]` and need pip's --group.
    pyjwt keeps its test requirements in one, so ignoring groups leaves a
    declared dependency uninstalled with nothing reported anywhere.
    """
    repo = _write_pyproject(tmp_path / "repo", """
[project]
name = "demo"

[dependency-groups]
dev = ["ruff"]
docs = ["sphinx"]
tests = ["pytest", "coverage"]
""")

    assert declared_test_groups(repo) == ["tests"]


def test_a_broken_pyproject_is_survivable(tmp_path):
    """A malformed manifest must not crash screening of the other candidates."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project", encoding="utf-8")

    assert read_pyproject(repo) == {}
    assert declared_extras(repo) == []


def test_tooling_denylist_covers_the_names_that_broke_collection():
    """requirements-dev.txt is skipped for this reason; extras follow the rule."""
    for name in ("dev", "docs", "lint", "type", "build", "release", "ci"):
        assert name in TOOLING_EXTRA_NAMES
    # Test extras are NOT tooling: they carry the plugins a suite imports.
    for name in ("test", "tests", "testing"):
        assert name not in TOOLING_EXTRA_NAMES


# --------------------------------------------------------------------------
# Report hygiene
# --------------------------------------------------------------------------

def test_rerunning_stage2_replaces_its_own_notes(tmp_path):
    notes = [f"{STAGE2_NOTE}failing_on_clean: ['eth address tests']", "stage1: cloned at abc123"]

    assert drop_stage2_notes(notes) == ["stage1: cloned at abc123"]


def test_the_accepted_repos_pass_the_gates_they_are_recorded_against():
    """
    Reads the committed report, so a re-screen that quietly breaks a gate fails
    here rather than at harvest time. pyjwt's suite went from 1.94s to 8.42s
    when its `crypto` extra was installed -- still inside the budget, but the
    margin is now worth watching.
    """
    import json

    report = json.loads(
        (REPO_ROOT / "data" / "repos" / "screening_report.json").read_text(encoding="utf-8")
    )
    accepted = {
        name: entry
        for name, entry in report.get("candidates", {}).items()
        if entry.get("stage3_passed")
    }

    assert accepted, "the report records no harvestable repository"
    for name, entry in accepted.items():
        hours = entry["suite_seconds"] * PLANNED_MUTANTS_PER_REPO / 3600
        assert hours <= MAX_HARVEST_HOURS_PER_REPO, f"{name} costs {hours:.2f}h"
        assert MIN_TESTS <= entry["n_tests_collected"] <= MAX_TESTS, name
        assert entry["deterministic"], name
        assert entry["n_varying_dep_features"] >= MIN_VARYING_DEP_FEATURES, name
