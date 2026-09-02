"""
ConfTest Repository Screener.

Mutation harvesting runs the full test suite once per mutant, so a slow or
non-deterministic suite makes a repository unusable. This script screens
candidates BEFORE any compute is spent on harvesting.

Two stages, cheapest first:

  STAGE 1 (static, no install)
      clone, reject C extensions, check src/tests separation, count tests
  STAGE 2 (dynamic, installs into an isolated venv)
      install, run the suite N times, verify it is green, fast, deterministic

Usage
-----
    python scripts/screen_repos.py --stage1              # cheap triage
    python scripts/screen_repos.py --stage2              # survivors only
    python scripts/screen_repos.py --all                 # both stages
    python scripts/screen_repos.py --stage1 --repos cachetools tabulate

Output
------
    data/repos/<name>/               cloned checkouts
    data/repos/screening_report.json gate results per candidate
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conftest.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# Newline as a name, so file content assembled here never needs an escape.
NEWLINE = chr(10)


# --------------------------------------------------------------------------
# Gate thresholds
# --------------------------------------------------------------------------

# Regression test selection needs something to select on. A repository whose
# tests all relate identically to every changeable file makes the task vacuous:
# the dependency features are constant, and a model can only learn a per-test
# fragility prior. So a subject must exhibit at least this many *varying*
# dependency features across its (test file, source file) pairs.
#
# Measured, not assumed. The file-count proxy is demonstrably wrong: tabulate
# has 4 source files and varies 6/6, while cachetools has 5 and varies 2/6.
# The single-module libraries parse and inflection vary 0/6 -- one changeable
# file means one dependency relationship, repeated for every row.
MIN_VARYING_DEP_FEATURES = 3

# Dependency features whose variance is measured in stage 3.
DEP_FEATURE_NAMES = (
    "dep_is_direct_import",
    "dep_name_heuristic_coupled",
    "dep_shortest_path_depth",
    "dep_is_reachable",
    "dep_max_reverse_dependencies",
    "dep_test_total_out_degree",
)

# Harvest cost. This is the binding constraint on the whole project: the
# harness runs the full suite once per mutant, so a repository costs
# (clean suite seconds x mutants) of wall clock.
#
# The first version of this gate capped the TEST COUNT at 600, which is a proxy
# for cost -- and a bad one, because the screener measures the real thing.
# validators (895 tests, 4.96s) costs 0.34h at 250 mutants, less than half of
# accepted sortedcontainers (296 tests, 11.58s, 0.80h); the proxy rejected the
# cheap repo and admitted the expensive one. Meanwhile the 90s suite ceiling
# would have allowed a single repository to consume 6.25h. Both gates are
# replaced by one bound on the quantity that actually matters.
MAX_HARVEST_HOURS_PER_REPO = 1.0
PLANNED_MUTANTS_PER_REPO = 250
MAX_SUITE_SECONDS = MAX_HARVEST_HOURS_PER_REPO * 3600 / PLANNED_MUTANTS_PER_REPO  # 14.4s

MIN_TESTS = 50

# A ceiling on rows, not on cost: the dataset holds (mutants x universe) rows
# per repository, so 250 x 1200 is 300k rows from one subject. Cost is bounded
# by MAX_SUITE_SECONDS above; this only keeps a single repository from
# dominating the pooled dataset.
MAX_TESTS = 1200
DETERMINISM_RUNS = 3
INSTALL_TIMEOUT = 600
SUITE_TIMEOUT = 300

# Candidate pool: pure-Python, well-tested, in the right size band.
DEFAULT_CANDIDATES: Dict[str, str] = {
    "cachetools": "https://github.com/tkem/cachetools",
    "tabulate": "https://github.com/astanin/python-tabulate",
    "python-slugify": "https://github.com/un33k/python-slugify",
    "inflection": "https://github.com/jpvanhal/inflection",
    "validators": "https://github.com/python-validators/validators",
    "schedule": "https://github.com/dbader/schedule",
    "parse": "https://github.com/r1chardj0n3s/parse",
    "humanize": "https://github.com/python-humanize/humanize",
    "deepdiff": "https://github.com/seperman/deepdiff",
    "toml": "https://github.com/uiri/toml",
    "pyparsing": "https://github.com/pyparsing/pyparsing",
    "arrow": "https://github.com/arrow-py/arrow",
    # Second wave. The first wave yielded only 4 acceptances: humanize (798),
    # validators (895), arrow and deepdiff overshoot the 600 ceiling, toml
    # undershoots the 50 floor, pyparsing has 2156, and schedule calls
    # time.tzset() which does not exist on Windows.
    "sqlparse": "https://github.com/andialbrecht/sqlparse",
    "pathspec": "https://github.com/cpburnz/python-pathspec",
    "funcy": "https://github.com/Suor/funcy",
    "cerberus": "https://github.com/pyeve/cerberus",
    "sortedcontainers": "https://github.com/grantjenks/python-sortedcontainers",
    "pluggy": "https://github.com/pytest-dev/pluggy",
    "chardet": "https://github.com/chardet/chardet",
    "semver": "https://github.com/python-semver/python-semver",
    # Third wave. The structural gate (stage 3) cut the second wave from 7
    # acceptances to 3: parse and inflection are single-module libraries, so
    # every test relates identically to the only changeable file and test
    # selection has nothing to select on. This wave is chosen for the opposite
    # property -- packages split across many modules, where a change touches
    # some tests' dependencies and not others.
    "click": "https://github.com/pallets/click",
    "jinja2": "https://github.com/pallets/jinja",
    "markdown": "https://github.com/Python-Markdown/markdown",
    "packaging": "https://github.com/pypa/packaging",
    "pyjwt": "https://github.com/jpadilla/pyjwt",
    "marshmallow": "https://github.com/marshmallow-code/marshmallow",
    "tomlkit": "https://github.com/python-poetry/tomlkit",
    "mistune": "https://github.com/lepture/mistune",
    "jsonschema": "https://github.com/python-jsonschema/jsonschema",
}

# Test-only pytest plugins to install unconditionally.
#
# A test module that imports a plugin the repository never declared fails to
# IMPORT, which aborts collection and looks identical to "this repo has almost
# no tests" -- that is how cerberus was rejected with 247 perfectly good tests
# for want of pytest-benchmark.
#
# Every entry here only ADDS fixtures or markers. That is the distinction from
# requirements-dev.txt, which is still deliberately skipped: a stale
# pytest-flake8 registers a pytest_collect_file hook and aborts the whole run
# with PluginValidationError. Nothing in this list registers a collect hook.
COMMON_TEST_PLUGINS = (
    "pytest-benchmark",
    "pytest-mock",
    "pytest-timeout",
    "freezegun",
    "hypothesis",
)

# Extras and dependency groups that exist for maintainers, not for the library.
#
# Everything a repository declares as an extra is installed EXCEPT these,
# because an optional dependency that is absent does not fail a test -- it
# skips it, and a skipped test never enters the label universe. Measured on
# the accepted set: pyjwt's `crypto` extra is the difference between 221 and
# 369 collected tests, and validators' `crypto-eth-addresses` is why 17 of its
# tests fail on a clean checkout. Harvesting without them would silently
# discard 40% of pyjwt's suite.
#
# The deny-list is what requirements-dev.txt taught: linters and doc builders
# add nothing a test can call, and a stale one (pytest-flake8) can abort
# collection outright.
TOOLING_EXTRA_NAMES = frozenset({
    "dev", "devel", "development", "doc", "docs", "lint", "linting",
    "type", "types", "typing", "mypy", "build", "release", "publish",
    "ci", "cov", "coverage", "bench", "benchmark", "format", "style",
})

# Prefix for notes owned by stage 2, so re-running the stage replaces them.
# Notes are carried forward with the rest of the report entry, which once left
# validators advertising `failing_on_clean: [...17 eth tests...]` after the
# extras change had brought that count to 0 -- a note contradicting the field
# beside it is worse than no note.
STAGE2_NOTE = "stage2: "

# Markers that a package builds compiled extensions.
C_EXTENSION_MARKERS = ("ext_modules", "Extension(", "cffi", "Cython", "build_ext")
C_SOURCE_SUFFIXES = (".c", ".pyx", ".pxd", ".cpp", ".h")


@dataclass
class ScreenResult:
    """Gate outcomes for one candidate repository."""

    name: str
    url: str
    stage1_passed: bool = False
    stage2_passed: bool = False
    rejected_reason: Optional[str] = None

    # Stage 1
    cloned: bool = False
    commit_sha: Optional[str] = None
    pure_python: Optional[bool] = None
    source_dirs: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    n_source_files: int = 0
    n_test_files: int = 0

    # Stage 2
    installed: Optional[bool] = None
    n_tests_collected: Optional[int] = None
    suite_seconds: Optional[float] = None
    all_green: Optional[bool] = None
    deterministic: Optional[bool] = None
    n_flaky: Optional[int] = None
    n_failing_on_clean: Optional[int] = None
    installed_extras: List[str] = field(default_factory=list)

    # Stage 3
    stage3_passed: bool = False
    n_varying_dep_features: Optional[int] = None
    n_distinct_dep_vectors: Optional[int] = None
    varying_dep_features: List[str] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    def reject(self, reason: str) -> "ScreenResult":
        self.rejected_reason = reason
        logger.warning(f"  REJECTED {self.name}: {reason}")
        return self


# --------------------------------------------------------------------------
# Shell helper
# --------------------------------------------------------------------------

def run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Run a command, returning (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)


# --------------------------------------------------------------------------
# Stage 1: static screening
# --------------------------------------------------------------------------

def clone_repo(url: str, dest: Path) -> Tuple[bool, Optional[str]]:
    """Shallow-clone a repository. Returns (ok, commit_sha)."""
    if (dest / ".git").exists():
        logger.info(f"  already cloned: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        code, _, err = run(
            ["git", "clone", "--depth", "1", url, str(dest)], timeout=INSTALL_TIMEOUT
        )
        if code != 0:
            logger.warning(f"  clone failed: {err.strip()[:300]}")
            return False, None

    code, out, _ = run(["git", "rev-parse", "HEAD"], cwd=dest)
    return True, out.strip() if code == 0 else None


def detect_c_extensions(repo: Path) -> bool:
    """True if the repository appears to build compiled extensions."""
    for build_file in ("setup.py", "setup.cfg", "pyproject.toml"):
        path = repo / build_file
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in text for marker in C_EXTENSION_MARKERS):
            return True

    # Only C sources that could plausibly be compiled count. Sample data under
    # examples/ or docs/ is not a build input (pyparsing ships examples/*.h).
    ignored_parts = {".git", "examples", "example", "docs", "doc", "tests", "test"}
    for suffix in C_SOURCE_SUFFIXES:
        for found in repo.rglob(f"*{suffix}"):
            if ignored_parts & set(found.parts):
                continue
            if "test" in found.name.lower():
                continue
            return True

    return False


def _holds_collectible_tests(directory: Path) -> bool:
    """True when pytest's default `python_files` patterns match something here."""
    for path in directory.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            return True
    return False


def locate_source_and_tests(repo: Path) -> Tuple[List[str], List[str]]:
    """
    Identify mutable source directories and test directories.

    The dependency-graph features need source and tests to be separable, so a
    repository that interleaves them is rejected.
    """
    source_dirs: List[str] = []
    test_dirs: List[str] = []
    skip = {".git", ".github", "docs", "examples", "benchmarks", "scripts",
            "build", "dist", ".tox", ".venv", "__pycache__"}

    # Layout A: src/<pkg>
    if (repo / "src").is_dir():
        source_dirs.append("src")

    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name in skip or child.name.startswith("."):
            continue
        if child.name.endswith(".egg-info"):
            continue

        name = child.name.lower()
        # pluggy keeps its suite in testing/, which the original name list
        # missed -- the repo was rejected as having no tests at all.
        if name in ("test", "tests", "testing") or name.startswith("test_"):
            test_dirs.append(child.name)
        elif child.name != "src" and (child / "__init__.py").exists():
            # Layout B: top-level package directory.
            #
            # But only when there is no src/ layout. packaging ships src/ AND a
            # tasks/ package of release automation (select_pypi_dist.py,
            # licenses.py) with an __init__.py, and it was picked up as mutable
            # source. Nothing imports tasks/ from the suite, so every mutant
            # placed there is guaranteed to kill nothing: budget spent producing
            # rows that are all-negative for a reason unrelated to selection.
            # A project that declares src/ has already said where its library is.
            if "src" not in source_dirs:
                source_dirs.append(child.name)

    # Tests may live inside the package (pkg/tests/). A directory NAMED tests is
    # not necessarily a suite: jsonschema vendors the JSON-Schema-Test-Suite at
    # json/tests/, which holds thousands of .json fixtures and zero .py files.
    # Taking the first rglob hit picked that and reported no_test_files, losing a
    # repo whose real suite sits at jsonschema/tests/. Require collectible
    # content, and prefer the shallowest candidate that has it.
    if not test_dirs:
        candidates = [
            c for c in repo.rglob("tests")
            if c.is_dir()
            and ".git" not in c.parts
            and "__pycache__" not in c.parts
            and _holds_collectible_tests(c)
        ]
        candidates.sort(key=lambda c: (len(c.relative_to(repo).parts), c.as_posix()))
        if candidates:
            test_dirs.append(candidates[0].relative_to(repo).as_posix())

    # Small projects ship a single top-level test module instead of a package.
    # Only pytest-discoverable names count: the default `python_files` patterns
    # are `test_*.py` and `*_test.py`, so a bare `test.py` is NOT collected.
    if not test_dirs:
        for candidate in sorted(repo.glob("*.py")):
            if candidate.name.startswith("test_") or candidate.name.endswith("_test.py"):
                test_dirs.append(candidate.name)

    return source_dirs, test_dirs


def count_py_files(repo: Path, dirs: List[str], want_tests: bool) -> int:
    """Count test or non-test Python files under the given directories."""
    total = 0
    for rel in dirs:
        base = repo / rel
        if base.is_file() and base.suffix == ".py":
            is_test = base.name.startswith("test_") or base.name.endswith("_test.py")
            if is_test == want_tests:
                total += 1
            continue
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            is_test = path.name.startswith("test_") or path.name.endswith("_test.py")
            if is_test == want_tests:
                total += 1
    return total


def screen_stage1(name: str, url: str, workspace: Path) -> ScreenResult:
    """Static gates: clone, purity, layout, file counts."""
    logger.info(f"[stage 1] {name}")
    result = ScreenResult(name=name, url=url)
    dest = workspace / name

    ok, sha = clone_repo(url, dest)
    result.cloned = ok
    result.commit_sha = sha
    if not ok:
        return result.reject("clone_failed")

    result.pure_python = not detect_c_extensions(dest)
    if not result.pure_python:
        return result.reject("builds_c_extensions")

    source_dirs, test_dirs = locate_source_and_tests(dest)
    result.source_dirs = source_dirs
    result.test_dirs = test_dirs

    if not source_dirs:
        return result.reject("no_source_package_found")
    if not test_dirs:
        return result.reject("no_test_directory_found")

    result.n_source_files = count_py_files(dest, source_dirs, want_tests=False)
    result.n_test_files = count_py_files(dest, test_dirs, want_tests=True)

    if result.n_source_files == 0:
        return result.reject("no_mutable_source_files")
    if result.n_test_files == 0:
        return result.reject("no_test_files")

    result.stage1_passed = True
    logger.info(
        f"  PASS  source={source_dirs} ({result.n_source_files} files), "
        f"tests={test_dirs} ({result.n_test_files} files)"
    )
    return result


# --------------------------------------------------------------------------
# Stage 2: dynamic screening
# --------------------------------------------------------------------------

def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def read_pyproject(repo: Path) -> Dict[str, Any]:
    """Parse pyproject.toml, or return {} when there is none or it is broken."""
    path = repo / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 and older
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"  unreadable pyproject.toml in {repo.name}: {exc}")
        return {}


def declared_extras(repo: Path) -> List[str]:
    """
    Functional extras the repository declares, tooling excluded.

    Reads both metadata homes: `project.optional-dependencies` in
    pyproject.toml and `[options.extras_require]` in setup.cfg. Names are
    matched against the deny-list case-insensitively and with separators
    normalised, so `crypto-eth-addresses` survives while `Dev` does not.
    """
    names: List[str] = []

    project = read_pyproject(repo).get("project", {})
    if isinstance(project, dict):
        names.extend(str(k) for k in project.get("optional-dependencies", {}) or {})

    cfg_path = repo / "setup.cfg"
    if cfg_path.is_file():
        import configparser

        parser = configparser.ConfigParser()
        try:
            parser.read(cfg_path, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"  unreadable setup.cfg in {repo.name}: {exc}")
        else:
            if parser.has_section("options.extras_require"):
                names.extend(parser["options.extras_require"].keys())

    keep: List[str] = []
    for name in names:
        normalised = name.strip().lower().replace("_", "-")
        if normalised in TOOLING_EXTRA_NAMES:
            continue
        # A compound name is classified by its FIRST part, which is how these
        # names are built: the category leads and a qualifier follows. So
        # `type-checking` and `dev-docs` are tooling, while
        # `crypto-eth-addresses` is not. Requiring every part to be tooling let
        # `type_checking` through, since `checking` is not itself a keyword.
        head = normalised.split("-")[0]
        if head in TOOLING_EXTRA_NAMES:
            continue
        if name not in keep:
            keep.append(name)
    return keep


def declared_test_groups(repo: Path) -> List[str]:
    """
    PEP 735 dependency groups that hold test requirements.

    Groups are not extras: they are not installable via `.[name]` and need
    pip 25.1's `--group`. pyjwt keeps its test requirements in one, so ignoring
    them would leave a declared dependency uninstalled with no error anywhere.
    Only test-shaped names are taken; `dev` and `docs` are tooling.
    """
    groups = read_pyproject(repo).get("dependency-groups", {})
    if not isinstance(groups, dict):
        return []
    wanted = ("test", "tests", "testing")
    return [name for name in groups if str(name).strip().lower() in wanted]


def create_venv_and_install(repo: Path, venv_dir: Path) -> Tuple[bool, str, List[str]]:
    """Create an isolated venv and install the repository plus pytest."""
    python = venv_python(venv_dir)

    if not python.exists():
        code, _, err = run(
            [sys.executable, "-m", "venv", str(venv_dir)], timeout=INSTALL_TIMEOUT
        )
        if code != 0 or not python.exists():
            return False, f"venv_creation_failed: {err.strip()[:200]}", []

    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"],
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"pip_bootstrap_failed: {err.strip()[:200]}", []

    # Editable install so the mutated working tree is what actually executes.
    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "-e", "."],
        cwd=repo,
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"editable_install_failed: {err.strip()[:300]}", []

    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "pytest"],
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"pytest_install_failed: {err.strip()[:200]}", []

    # Additive plugins that test modules commonly import. Non-fatal: a failure
    # here leaves the repo no worse off than not trying.
    run(
        [str(python), "-m", "pip", "install", "--quiet", *COMMON_TEST_PLUGINS],
        timeout=INSTALL_TIMEOUT,
    )

    # Test-only dependencies. Without these, test modules fail to IMPORT and
    # pytest collects almost nothing -- which looks identical to "this repo has
    # too few tests". humanize needs freezegun; parse needs pytest-cov for its
    # addopts. Failures here are non-fatal: the extra may simply not exist.
    # Try every plausible extra name and do NOT stop at the first success:
    # pip exits 0 for an extra that does not exist (it only warns), so breaking
    # early silently skips the real one. humanize declares `tests`, and probing
    # `test` first would have masked it, leaving freezegun uninstalled.
    installed_extras: List[str] = []

    # Declared functional extras. An absent optional dependency does not fail a
    # test, it skips it, so this is the difference between measuring a library's
    # suite and measuring the part of it that happens to run bare.
    for extra in declared_extras(repo):
        code, _, err = run(
            [str(python), "-m", "pip", "install", "--quiet", "-e", f".[{extra}]"],
            cwd=repo,
            timeout=INSTALL_TIMEOUT,
        )
        if code == 0:
            installed_extras.append(extra)
        else:
            # Non-fatal by design: pathspec declares `hyperscan` and `re2`,
            # which have no Windows wheels. The repository is still usable, it
            # just keeps the skips those extras would have unlocked.
            logger.info(f"  extra unavailable here: {extra} ({err.strip()[:120]})")
            installed_extras.append(f"{extra}:unavailable")

    # PEP 735 groups, which `.[name]` cannot reach.
    for group in declared_test_groups(repo):
        code, _, err = run(
            [str(python), "-m", "pip", "install", "--quiet", "--group", group],
            cwd=repo,
            timeout=INSTALL_TIMEOUT,
        )
        if code == 0:
            installed_extras.append(f"group:{group}")
        else:
            logger.info(f"  group unavailable: {group} ({err.strip()[:120]})")

    # Blind probe for extras declared somewhere unreadable -- a setup.py that
    # builds `extras_require` in code, for instance. Names already handled above
    # are skipped. pip exits 0 for an extra that does not exist (it only warns),
    # so a success here proves nothing and is not recorded.
    handled = {name.split(":")[0].lower() for name in installed_extras}
    for extra in ("test", "tests", "testing"):
        if extra in handled:
            continue
        run(
            [str(python), "-m", "pip", "install", "--quiet", "-e", f".[{extra}]"],
            cwd=repo,
            timeout=INSTALL_TIMEOUT,
        )

    # Deliberately NOT installing requirements-dev.txt / dev.requirements.txt:
    # those pull linters, and a stale pytest-flake8 registers an incompatible
    # pytest_collect_file hook that aborts the entire run with
    # PluginValidationError. Only test-specific requirement files are used.
    for req_name in (
        "tests/requirements.txt",
        "test/requirements.txt",
        "requirements-test.txt",
        "test-requirements.txt",
    ):
        req_path = repo / req_name
        if req_path.exists():
            run(
                [str(python), "-m", "pip", "install", "--quiet", "-r", req_name],
                cwd=repo,
                timeout=INSTALL_TIMEOUT,
            )
            installed_extras.append(f"requirements:{req_name}")

    return True, "ok", installed_extras


def parse_junit(xml_path: Path) -> Dict[str, str]:
    """Map test node ID to status from a JUnit XML report."""
    import xml.etree.ElementTree as ET

    if not xml_path.exists():
        return {}

    outcomes: Dict[str, str] = {}
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}

    for case in root.iter("testcase"):
        file_attr = case.attrib.get("file", "")
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        node_id = (
            f"{file_attr.replace(os.sep, '/')}::{name}"
            if file_attr
            else f"{classname.replace('.', '/')}::{name}"
        )

        if case.find("failure") is not None:
            status = "FAILED"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        outcomes[node_id] = status

    return outcomes


# pytest exit codes that mean "this suite did not really run".
PYTEST_EXIT_MEANING = {
    2: "collection_interrupted",   # e.g. a test module failed to import
    3: "pytest_internal_error",
    4: "pytest_usage_error",       # e.g. addopts referencing a missing plugin
    5: "no_tests_collected",
    124: "suite_timeout",
}


NEUTRAL_PYTEST_INI = "[pytest]\n"


def isolate_workspace_from_project_config(workspace: Path) -> Path:
    """
    Stop pytest walking up out of a checkout into ConfTest's own config.

    pytest searches upward from the working directory for the first pytest.ini
    / tox.ini / setup.cfg / pyproject.toml carrying a pytest section. A subject
    repo that ships no pytest section of its own therefore adopted THIS
    project's [tool.pytest.ini_options] -- measured, not hypothesised:
    cachetools and sqlparse reported

        rootdir: <conftest project root>
        configfile: pyproject.toml

    which handed them our `pythonpath = ["src"]` (our package on their
    sys.path), our `asyncio_mode = "auto"`, and a `python_files` narrower than
    the pytest default. A subject repo must be measured under its own
    configuration, never ours.

    Writing a neutral, empty pytest.ini into the PARENT of the checkouts ends
    the upward walk there. A repo that has its own config still wins, because
    its own directory is searched first; only repos with no config at all fall
    through to this one. Nothing inside any checkout is touched, so byte-exact
    restoration and the G3 clean-tree check are unaffected.
    """
    marker = workspace / "pytest.ini"
    if not marker.exists() or marker.read_text(encoding="utf-8") != NEUTRAL_PYTEST_INI:
        marker.write_text(NEUTRAL_PYTEST_INI, encoding="utf-8")
    return marker


def collection_diagnostic(stdout: str, stderr: str) -> str:
    """
    Pull the lines that explain a collection abort out of pytest output.

    A bare "collection_interrupted" verdict is the same opacity that once let
    a screener bug hide: pyparsing looked like a 1-test project when its
    collection had actually died. Recording which modules errored, and why,
    lets a human tell a broken repo from a missing test-only plugin
    (cerberus aborts only because pytest-benchmark is absent).
    """
    lines = []
    for line in (stdout + "\n" + stderr).splitlines():
        stripped = line.strip()
        if stripped.startswith(("ERROR ", "E   ")) or "Interrupted:" in stripped:
            lines.append(stripped)
    # Keep the tail: the summary block repeats the per-module errors compactly.
    return " | ".join(lines[-6:])


def run_suite(
    python: Path, repo: Path, report: Path
) -> Tuple[Dict[str, str], float, int, str]:
    """
    Run the full suite once.

    Returns (outcomes, seconds, exit_code, diagnostic).

    The exit code matters: a run interrupted during collection still writes a
    JUnit file containing the handful of tests gathered before the abort, so
    counting rows alone reports "too few tests" for what is really a broken
    import. pyparsing appeared to have 1 test when it actually has 2156.
    """
    started = time.time()
    code, out, err = run(
        # Flags must match SafeTestExecutor exactly, or screening timings and
        # collection counts will not describe what the harvest actually runs.
        [
            str(python), "-m", "pytest", f"--junitxml={report}", "-q",
            "-o", "addopts=", "-p", "no:cacheprovider",
        ],
        cwd=repo,
        timeout=SUITE_TIMEOUT,
    )
    elapsed = time.time() - started
    diagnostic = collection_diagnostic(out, err) if code in PYTEST_EXIT_MEANING else ""
    return parse_junit(report), elapsed, code, diagnostic


def list_py_files(repo: Path, locations: List[str], want_tests: bool) -> List[str]:
    """
    Repo-relative paths of the .py files at the given locations.

    Mirrors count_py_files, including its single-file case: a test location may
    be one module (inflection ships test_inflection.py, not a tests/ package).
    """
    found: List[str] = []
    for rel in locations:
        base = repo / rel
        if base.is_file():
            if base.suffix == ".py":
                found.append(base.relative_to(repo).as_posix())
            continue
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            name = path.name
            is_test = name.startswith("test_") or name.endswith("_test.py")
            if is_test == want_tests:
                found.append(path.relative_to(repo).as_posix())
    return sorted(found)


def measure_dependency_variance(
    repo: Path, source_dirs: List[str], test_dirs: List[str]
) -> Tuple[List[str], int]:
    """
    Measure how much the dependency features actually vary in this repository.

    Enumerates every (test file, source file) pair -- the two axes a real row is
    built from -- and records which dependency features take more than one
    value. Returns (varying feature names, distinct feature-vector count).

    This is static analysis only: no install, no test run. It costs seconds,
    so it can gate before the expensive dynamic stage rather than after.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from conftest.features.dependency_graph import DependencyGraphBuilder

    sources = list_py_files(repo, source_dirs, want_tests=False)
    tests = list_py_files(repo, test_dirs, want_tests=True)
    if not tests:
        # A suite of one module whose name does not match the test patterns is
        # already rejected in stage 1; this covers the single-file case where
        # the location IS the module.
        tests = [t for t in test_dirs if (repo / t).is_file()]
    if not sources or not tests:
        return [], 0

    builder = DependencyGraphBuilder(str(repo))
    seen_values: Dict[str, set] = {name: set() for name in DEP_FEATURE_NAMES}
    vectors: set = set()

    for test_path in tests:
        for source_path in sources:
            feats = builder.compute_dependency_features(test_path, [source_path])
            vector = []
            for name in DEP_FEATURE_NAMES:
                value = round(float(feats.get(name, 0.0)), 4)
                seen_values[name].add(value)
                vector.append(value)
            vectors.add(tuple(vector))

    varying = [name for name in DEP_FEATURE_NAMES if len(seen_values[name]) > 1]
    return varying, len(vectors)


def screen_stage3(result: ScreenResult, workspace: Path) -> ScreenResult:
    """
    Structural gate: is regression test selection a non-trivial task here?

    A single-module library gives every test the same relationship to the only
    changeable file, so every dependency feature is constant and the model can
    learn nothing but a per-test fragility prior. Harvesting such a repo costs
    real hours and contributes rows that cannot exercise the method.

    Measured on the seven stage-2 survivors:

        sqlparse         21 src files   84 distinct vectors   6/6 varying
        pathspec         31            86                    5/6
        tabulate          4            12                    6/6
        cachetools        5             6                    2/6
        sortedcontainers  4             4                    1/6
        parse             1             1                    0/6
        inflection        1             1                    0/6

    Note that source-file count does not predict the outcome -- tabulate varies
    more with 4 files than cachetools does with 5 -- which is why this gate
    measures the features themselves rather than counting files.
    """
    logger.info(f"[stage 3] {result.name}")
    repo = workspace / result.name

    varying, n_vectors = measure_dependency_variance(
        repo, result.source_dirs, result.test_dirs
    )
    result.varying_dep_features = list(varying)
    result.n_varying_dep_features = len(varying)
    result.n_distinct_dep_vectors = n_vectors

    if len(varying) < MIN_VARYING_DEP_FEATURES:
        return result.reject(
            f"only_{len(varying)}_of_{len(DEP_FEATURE_NAMES)}_dependency_features_vary_"
            f"(need_{MIN_VARYING_DEP_FEATURES});_test_selection_is_vacuous_here"
        )

    result.stage3_passed = True
    logger.info(
        f"  PASS  {len(varying)}/{len(DEP_FEATURE_NAMES)} dependency features vary, "
        f"{n_vectors} distinct vectors"
    )
    return result


def drop_stage2_notes(notes: List[str]) -> List[str]:
    """
    Remove notes owned by stage 2 so a re-run replaces them instead of adding.

    Report entries are carried forward whole, which once left validators
    advertising `failing_on_clean: [...17 eth tests...]` after installing the
    extra had brought that count to 0. A note contradicting the field beside it
    is worse than no note.
    """
    return [note for note in notes if not note.startswith(STAGE2_NOTE)]


def screen_stage2(result: ScreenResult, workspace: Path) -> ScreenResult:
    """Dynamic gates: install, size, speed, greenness, determinism."""
    logger.info(f"[stage 2] {result.name}")
    result.notes = drop_stage2_notes(result.notes)
    repo = workspace / result.name
    venv_dir = workspace / f".venv_{result.name}"

    installed, message, extras = create_venv_and_install(repo, venv_dir)
    result.installed = installed
    result.installed_extras = extras
    if extras:
        logger.info(f"  extras: {', '.join(extras)}")
    if not installed:
        return result.reject(message)

    python = venv_python(venv_dir)
    report = workspace / f".report_{result.name}.xml"

    runs: List[Dict[str, str]] = []
    durations: List[float] = []

    for index in range(DETERMINISM_RUNS):
        outcomes, elapsed, exit_code, diagnostic = run_suite(python, repo, report)

        if exit_code in PYTEST_EXIT_MEANING:
            reason = PYTEST_EXIT_MEANING[exit_code]
            result.notes.append(
                f"{STAGE2_NOTE}pytest exit {exit_code} ({reason}); "
                f"{len(outcomes)} tests were in the partial report"
            )
            if diagnostic:
                result.notes.append(f"{STAGE2_NOTE}pytest said: {diagnostic}")
            return result.reject(reason)
        if not outcomes:
            return result.reject("collected_zero_tests")

        runs.append(outcomes)
        durations.append(elapsed)
        logger.info(f"  run {index + 1}/{DETERMINISM_RUNS}: {len(outcomes)} tests, {elapsed:.1f}s")

    result.n_tests_collected = len(runs[0])
    result.suite_seconds = round(sum(durations) / len(durations), 2)

    if result.n_tests_collected < MIN_TESTS:
        return result.reject(f"too_few_tests_{result.n_tests_collected}_under_{MIN_TESTS}")
    if result.n_tests_collected > MAX_TESTS:
        return result.reject(f"too_many_tests_{result.n_tests_collected}_over_{MAX_TESTS}")
    if result.suite_seconds > MAX_SUITE_SECONDS:
        hours = result.suite_seconds * PLANNED_MUTANTS_PER_REPO / 3600
        return result.reject(
            f"harvest_too_expensive_{hours:.2f}h_over_{MAX_HARVEST_HOURS_PER_REPO}h_"
            f"({result.suite_seconds}s_x_{PLANNED_MUTANTS_PER_REPO}_mutants)"
        )

    # Determinism: every test must report the same status in every run.
    all_ids = set().union(*(set(r) for r in runs))
    flaky = [
        tid for tid in all_ids
        if len({r.get(tid, "MISSING") for r in runs}) > 1
    ]
    failing = [
        tid for tid in all_ids
        if runs[0].get(tid) in ("FAILED", "ERROR")
    ]

    result.n_flaky = len(flaky)
    result.n_failing_on_clean = len(failing)
    result.deterministic = not flaky
    result.all_green = not failing

    if flaky:
        result.notes.append(f"{STAGE2_NOTE}flaky: {flaky[:5]}")
    if failing:
        result.notes.append(f"{STAGE2_NOTE}failing_on_clean: {failing[:5]}")

    # Flaky and pre-failing tests are excluded by the harness rather than
    # disqualifying the repo, provided enough stable tests remain.
    stable = result.n_tests_collected - len(flaky) - len(failing)
    if stable < MIN_TESTS:
        return result.reject(f"only_{stable}_stable_tests_under_{MIN_TESTS}")

    result.stage2_passed = True
    logger.info(
        f"  PASS  {result.n_tests_collected} tests, {result.suite_seconds}s, "
        f"{stable} stable, {len(flaky)} flaky, {len(failing)} failing"
    )
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def load_report(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Existing report unreadable, starting fresh.")
    return {"candidates": {}}


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline: the report is committed, and a file without one shows a
    # spurious change on every diff.
    path.write_text(json.dumps(report, indent=2) + NEWLINE)


def print_summary(report: Dict[str, Any]) -> None:
    """Print the gate table and the harness-ready configuration."""
    candidates = report.get("candidates", {})
    if not candidates:
        print("\nNo candidates screened yet.")
        return

    print("\n" + "=" * 78)
    print("SCREENING SUMMARY")
    print("=" * 78)
    header = (f"{'repo':<18} {'s1':<4} {'s2':<4} {'s3':<4} {'tests':>6} {'secs':>7} "
              f"{'dep':>5}  {'reason / notes'}")
    print(header)
    print("-" * 78)

    accepted: List[Dict[str, Any]] = []
    for name in sorted(candidates):
        entry = candidates[name]
        s1 = "ok" if entry.get("stage1_passed") else "--"
        s2 = "ok" if entry.get("stage2_passed") else "--"
        s3 = "ok" if entry.get("stage3_passed") else "--"
        tests = entry.get("n_tests_collected") or "-"
        secs = entry.get("suite_seconds") or "-"
        n_vary = entry.get("n_varying_dep_features")
        dep = f"{n_vary}/{len(DEP_FEATURE_NAMES)}" if n_vary is not None else "-"
        reason = entry.get("rejected_reason") or ";".join(entry.get("notes", [])) or ""
        print(f"{name:<18} {s1:<4} {s2:<4} {s3:<4} {str(tests):>6} {str(secs):>7} "
              f"{dep:>5}  {reason[:32]}")
        if entry.get("stage3_passed"):
            accepted.append(entry)

    print("-" * 78)
    print(f"Stage 1 passed: {sum(1 for e in candidates.values() if e.get('stage1_passed'))}")
    print(f"Stage 2 passed: {sum(1 for e in candidates.values() if e.get('stage2_passed'))}")
    print(f"Stage 3 passed: {len(accepted)}   (target: 5)   <- eligible for harvest")

    if accepted:
        print("\nHARNESS CONFIGURATION for accepted repositories:")
        print("-" * 78)
        for entry in accepted:
            print(f"  {{'repo_name': {entry['name']!r},")
            print(f"    'repo_root': 'data/repos/{entry['name']}',")
            print(f"    'source_dirs': {entry['source_dirs']!r},")
            print(f"    'commit_sha': {entry.get('commit_sha')!r}}},")

        estimated = sum(
            (entry.get("suite_seconds") or 0) * 250 / 3600 for entry in accepted
        )
        print(f"\nEstimated harvest cost at 250 mutants/repo: {estimated:.1f} hours total")
        print(f"Split across 4 laptops: {estimated / 4:.1f} hours each")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen repositories for mutation harvesting.")
    parser.add_argument("--stage1", action="store_true", help="run static screening")
    parser.add_argument("--stage2", action="store_true", help="run dynamic screening on survivors")
    parser.add_argument("--stage3", action="store_true",
                        help="run the structural (dependency-variance) screen on survivors")
    parser.add_argument("--all", action="store_true", help="run all three stages")
    parser.add_argument("--repos", nargs="*", help="limit to these candidate names")
    parser.add_argument("--workspace", default="data/repos", help="clone destination")
    parser.add_argument("--summary", action="store_true", help="print the existing report only")
    args = parser.parse_args()

    configure_logging()

    workspace = (REPO_ROOT / args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    isolate_workspace_from_project_config(workspace)
    report_path = workspace / "screening_report.json"
    report = load_report(report_path)

    if args.summary:
        print_summary(report)
        return 0

    stage1 = args.stage1 or args.all
    stage2 = args.stage2 or args.all
    stage3 = args.stage3 or args.all
    if not (stage1 or stage2 or stage3):
        parser.error("choose --stage1, --stage2, --stage3, --all, or --summary")

    selected = args.repos or list(DEFAULT_CANDIDATES)
    unknown = [name for name in selected if name not in DEFAULT_CANDIDATES]
    if unknown:
        parser.error(f"unknown candidates: {unknown}")

    if stage1:
        logger.info(f"Stage 1: static screening of {len(selected)} candidates")
        for name in selected:
            try:
                result = screen_stage1(name, DEFAULT_CANDIDATES[name], workspace)
            except Exception as exc:
                logger.error(f"  stage 1 crashed for {name}: {exc}")
                result = ScreenResult(name=name, url=DEFAULT_CANDIDATES[name])
                result.reject(f"stage1_crash: {exc}")
            report["candidates"][name] = asdict(result)
            save_report(report_path, report)

    if stage2:
        survivors = [
            name for name in selected
            if report["candidates"].get(name, {}).get("stage1_passed")
        ]
        logger.info(f"Stage 2: dynamic screening of {len(survivors)} survivors")

        for name in survivors:
            entry = report["candidates"][name]
            result = ScreenResult(**entry)
            try:
                result = screen_stage2(result, workspace)
            except Exception as exc:
                logger.error(f"  stage 2 crashed for {name}: {exc}")
                result.reject(f"stage2_crash: {exc}")
            report["candidates"][name] = asdict(result)
            save_report(report_path, report)

    if stage3:
        # Stage 3 is static and cheap, so it runs on stage-2 survivors rather
        # than gating before them: the dynamic gates are what decide whether a
        # repo is usable at all, and a repo that fails them never reaches here.
        survivors = [
            name for name in selected
            if report["candidates"].get(name, {}).get("stage2_passed")
        ]
        logger.info(f"Stage 3: structural screening of {len(survivors)} survivors")

        for name in survivors:
            entry = report["candidates"][name]
            result = ScreenResult(**entry)
            try:
                result = screen_stage3(result, workspace)
            except Exception as exc:
                logger.error(f"  stage 3 crashed for {name}: {exc}")
                result.reject(f"stage3_crash: {exc}")
            report["candidates"][name] = asdict(result)
            save_report(report_path, report)

    report["screened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_report(report_path, report)
    print_summary(report)
    logger.info(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
