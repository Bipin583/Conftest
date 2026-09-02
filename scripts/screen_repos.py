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


# --------------------------------------------------------------------------
# Gate thresholds
# --------------------------------------------------------------------------

MAX_SUITE_SECONDS = 90.0
MIN_TESTS = 50
MAX_TESTS = 600
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
}

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
        if name in ("test", "tests") or name.startswith("test_"):
            test_dirs.append(child.name)
        elif child.name != "src" and (child / "__init__.py").exists():
            # Layout B: top-level package directory
            source_dirs.append(child.name)

    # Tests may live inside the package (pkg/tests/).
    if not test_dirs:
        for candidate in repo.rglob("tests"):
            if candidate.is_dir() and ".git" not in candidate.parts:
                test_dirs.append(candidate.relative_to(repo).as_posix())
                break

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


def create_venv_and_install(repo: Path, venv_dir: Path) -> Tuple[bool, str]:
    """Create an isolated venv and install the repository plus pytest."""
    python = venv_python(venv_dir)

    if not python.exists():
        code, _, err = run(
            [sys.executable, "-m", "venv", str(venv_dir)], timeout=INSTALL_TIMEOUT
        )
        if code != 0 or not python.exists():
            return False, f"venv_creation_failed: {err.strip()[:200]}"

    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"],
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"pip_bootstrap_failed: {err.strip()[:200]}"

    # Editable install so the mutated working tree is what actually executes.
    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "-e", "."],
        cwd=repo,
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"editable_install_failed: {err.strip()[:300]}"

    code, _, err = run(
        [str(python), "-m", "pip", "install", "--quiet", "pytest"],
        timeout=INSTALL_TIMEOUT,
    )
    if code != 0:
        return False, f"pytest_install_failed: {err.strip()[:200]}"

    # Test-only dependencies. Without these, test modules fail to IMPORT and
    # pytest collects almost nothing -- which looks identical to "this repo has
    # too few tests". humanize needs freezegun; parse needs pytest-cov for its
    # addopts. Failures here are non-fatal: the extra may simply not exist.
    # Try every plausible extra name and do NOT stop at the first success:
    # pip exits 0 for an extra that does not exist (it only warns), so breaking
    # early silently skips the real one. humanize declares `tests`, and probing
    # `test` first would have masked it, leaving freezegun uninstalled.
    for extra in ("test", "tests", "testing"):
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

    return True, "ok"


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


def screen_stage2(result: ScreenResult, workspace: Path) -> ScreenResult:
    """Dynamic gates: install, size, speed, greenness, determinism."""
    logger.info(f"[stage 2] {result.name}")
    repo = workspace / result.name
    venv_dir = workspace / f".venv_{result.name}"

    installed, message = create_venv_and_install(repo, venv_dir)
    result.installed = installed
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
                f"pytest exit {exit_code} ({reason}); "
                f"{len(outcomes)} tests were in the partial report"
            )
            if diagnostic:
                result.notes.append(f"pytest said: {diagnostic}")
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
        return result.reject(f"suite_too_slow_{result.suite_seconds}s_over_{MAX_SUITE_SECONDS}s")

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
        result.notes.append(f"flaky: {flaky[:5]}")
    if failing:
        result.notes.append(f"failing_on_clean: {failing[:5]}")

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
    path.write_text(json.dumps(report, indent=2))


def print_summary(report: Dict[str, Any]) -> None:
    """Print the gate table and the harness-ready configuration."""
    candidates = report.get("candidates", {})
    if not candidates:
        print("\nNo candidates screened yet.")
        return

    print("\n" + "=" * 78)
    print("SCREENING SUMMARY")
    print("=" * 78)
    header = f"{'repo':<18} {'s1':<4} {'s2':<4} {'tests':>6} {'secs':>7}  {'reason / notes'}"
    print(header)
    print("-" * 78)

    accepted: List[Dict[str, Any]] = []
    for name in sorted(candidates):
        entry = candidates[name]
        s1 = "ok" if entry.get("stage1_passed") else "--"
        s2 = "ok" if entry.get("stage2_passed") else "--"
        tests = entry.get("n_tests_collected") or "-"
        secs = entry.get("suite_seconds") or "-"
        reason = entry.get("rejected_reason") or ";".join(entry.get("notes", [])) or ""
        print(f"{name:<18} {s1:<4} {s2:<4} {str(tests):>6} {str(secs):>7}  {reason[:32]}")
        if entry.get("stage2_passed"):
            accepted.append(entry)

    print("-" * 78)
    print(f"Stage 1 passed: {sum(1 for e in candidates.values() if e.get('stage1_passed'))}")
    print(f"Stage 2 passed: {len(accepted)}   (target: 5)")

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
    parser.add_argument("--all", action="store_true", help="run both stages")
    parser.add_argument("--repos", nargs="*", help="limit to these candidate names")
    parser.add_argument("--workspace", default="data/repos", help="clone destination")
    parser.add_argument("--summary", action="store_true", help="print the existing report only")
    args = parser.parse_args()

    configure_logging()

    workspace = (REPO_ROOT / args.workspace).resolve()
    report_path = workspace / "screening_report.json"
    report = load_report(report_path)

    if args.summary:
        print_summary(report)
        return 0

    stage1 = args.stage1 or args.all
    stage2 = args.stage2 or args.all
    if not (stage1 or stage2):
        parser.error("choose --stage1, --stage2, --all, or --summary")

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

    report["screened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_report(report_path, report)
    print_summary(report)
    logger.info(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
