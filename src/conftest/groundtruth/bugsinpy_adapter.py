"""
BugsInPy real bugs as ground truth, in the schema the mutation harvest emits.

BugsInPy (https://github.com/soarsmu/BugsInPy) records 501 real defects from 17
Python projects. Each defect names a buggy commit, the fix commit, and the test(s)
that the fix makes pass. That is the external-validity set this project needs: the
mutation harvest measures reactions to synthetic one-line edits, and a reviewer is
entitled to ask whether the same selector works on defects a human actually shipped.

What BugsInPy supplies and what it does not:

  supplied   buggy commit, fixed commit, the failing test command, the fix diff,
             a pinned Python version, a frozen requirements file
  NOT        the test universe, per-test durations, which other tests fail

The three missing quantities are the ones test selection is scored on -- you cannot
compute a reduction without knowing what you declined to run -- so they have to be
measured here by running the suite. This module therefore does two separable things,
and keeps them separable:

  1. reads the metadata (offline, no execution, no environment)
  2. measures a bug by running the suite at both commits (needs an environment)

Step 1 alone yields no labels. A record that has not been through step 2 carries
`status` != "harvested" and no universe, rather than a zero or an empty list that a
downstream mean would silently absorb.

The measurement mirrors `MutationHarness`: the universe is the set of tests that pass
on *every* baseline run at the fixed commit, so a flaky test cannot be recorded as
broken by the bug. The bug's failing set is then the universe members that fail at the
buggy commit. BugsInPy's documented failing tests are treated as a claim to check
against that measurement, not as a label to copy -- a record is only `harvested` when
every documented test was observed to fail. Anything else is published with the
disagreement spelled out.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from conftest.groundtruth.mutation_harness import (
    BROKE_SUITE_KILL_RATIO,
    FAILING_STATUSES,
    TIMEOUT_BROKEN_STATUS,
    BaselineProfile,
    classify_runs,
    restore_tracked,
    tracked_modifications,
)
from conftest.logging_config import get_logger
from conftest.tests.executor import SafeTestExecutor

logger = get_logger(__name__)

BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy"

# Emitted on every record so a reader of data/groundtruth/bugsinpy/real_bugs.jsonl
# can regenerate it without reading this file.
PRODUCED_BY = "python scripts/harvest_bugsinpy.py"

# The four shapes BugsInPy actually uses in run_test.sh, measured over all 501 bugs
# at metadata commit 11c5f1ee: 445 `pytest <node>`, 142 `python -m unittest -q <dotted>`,
# 10 `py.test <node>`, 10 `python3 -m pytest <node>`, 5 `tox <node>`.
# Newline used inside f-strings that build multi-line diagnostics.
NEWLINE = chr(10)

_PYTEST_TOKENS = ("pytest", "py.test")
_UNITTEST_TOKEN = "unittest"


class BugsInPyLayoutError(RuntimeError):
    """The metadata checkout is not shaped the way this adapter reads it."""


def parse_info_file(text: str) -> Dict[str, str]:
    """
    Read BugsInPy's `key="value"` info files.

    They are shell fragments, so they are parsed as such rather than eval'd: a
    metadata repository is untrusted input, and `bug.info` is a file this project
    downloads from a third party.
    """
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?(.*?)"?\s*$', line)
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def normalise_test_id(raw: str) -> str:
    """
    Put a test id in the form `path/to/file::name`, the form the JUnit parser emits.

    BugsInPy writes `tqdm/tests/tests_contrib.py::test_enumerate`; the executor
    reconstructs ids from the JUnit report and may or may not carry the `.py`,
    depending on whether pytest filled in the `file` attribute. Comparing the two
    sides without normalising is how a confirmed bug looks unconfirmed.
    """
    ident = raw.strip().replace("\\", "/")
    if "::" not in ident:
        return ident
    head, sep, tail = ident.partition("::")
    if head.endswith(".py"):
        head = head[: -len(".py")]
    return f"{head}{sep}{tail}"


def parse_run_test(text: str) -> Tuple[List[str], List[str]]:
    """
    Turn run_test.sh into pytest node ids, and say which lines were not understood.

    Returns (node_ids, unreadable_lines). A line this function cannot map is returned
    rather than dropped: a bug whose failing test we cannot name is a bug we cannot
    verify, and silently returning the ones we did parse would make it look verified.

    `python -m unittest tests.test_black.BlackTestCase.test_x` is mapped to
    `tests/test_black::BlackTestCase::test_x` on the assumption that the dotted path
    is package.module.Class.method. The assumption is not always right -- a dotted
    module deeper than its file, or a test function at module level, shifts the split
    -- so the result is checked against the measured universe before it is trusted,
    and a miss is reported as a disagreement rather than a failure to find.
    """
    node_ids: List[str] = []
    unreadable: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        lowered = [t.lower() for t in tokens]

        if any(t in _PYTEST_TOKENS for t in lowered) or lowered[0] == "tox":
            # Arguments that are not flags and look like a target.
            found = [t for t in tokens if "::" in t or t.endswith(".py")]
            if not found:
                unreadable.append(line)
                continue
            node_ids.extend(normalise_test_id(t) for t in found)
            continue

        if _UNITTEST_TOKEN in lowered:
            dotted = [t for t in tokens[1:] if "." in t and not t.startswith("-")]
            if not dotted:
                unreadable.append(line)
                continue
            for target in dotted:
                parts = target.split(".")
                if len(parts) < 2:
                    unreadable.append(line)
                    continue
                # Walk from the right: the trailing parts that begin with a capital or
                # with `test` are the class and method; the rest is the module path.
                split_at = len(parts) - 1
                while split_at > 1 and (
                    parts[split_at - 1][:1].isupper() or parts[split_at - 1].startswith("test")
                ):
                    split_at -= 1
                module = "/".join(parts[:split_at])
                node_ids.append("::".join([module] + parts[split_at:]))
            continue

        unreadable.append(line)

    return node_ids, unreadable


@dataclass(frozen=True)
class PatchSummary:
    """The fix diff, reduced to the fields the feature pipeline asks a commit for."""

    changed_files: Tuple[str, ...]
    lines_added: int
    lines_deleted: int
    first_file: Optional[str]
    first_line: Optional[int]
    buggy_snippet: Optional[str]
    fixed_snippet: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_bug_patch(text: str) -> PatchSummary:
    """
    Read the fix diff. `a/` is the buggy side, so `-` lines are the defect.

    Every field here is counted off the diff. The mutation harvest could not do this:
    a mutant is one line in one file by construction, which is why 12 of the 32
    features were constant in training. A real fix touches what it touches, so these
    counts are the first thing this dataset adds that the mutants could not.
    """
    changed: List[str] = []
    added = deleted = 0
    first_file: Optional[str] = None
    first_line: Optional[int] = None
    buggy: List[str] = []
    fixed: List[str] = []

    for line in text.splitlines():
        if line.startswith("--- a/") or line.startswith("--- /dev/null"):
            continue
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            changed.append(path)
            if first_file is None:
                first_file = path
            continue
        if line.startswith("@@"):
            match = re.search(r"^@@ -(\d+)", line)
            if match and first_line is None:
                first_line = int(match.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
            if len(fixed) < 4:
                fixed.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            if len(buggy) < 4:
                buggy.append(line[1:].strip())

    return PatchSummary(
        changed_files=tuple(dict.fromkeys(changed)),
        lines_added=added,
        lines_deleted=deleted,
        first_file=first_file,
        first_line=first_line,
        buggy_snippet=" / ".join(buggy) if buggy else None,
        fixed_snippet=" / ".join(fixed) if fixed else None,
    )


@dataclass(frozen=True)
class BugSpec:
    """One BugsInPy defect, as read off disk. No measurement in here."""

    project: str
    bug_id: int
    github_url: str
    python_version: str
    buggy_commit: str
    fixed_commit: str
    test_file: Optional[str]
    pythonpath: Optional[str]
    documented_failing_tests: Tuple[str, ...]
    unreadable_test_commands: Tuple[str, ...]
    patch: PatchSummary
    pinned_requirements: Tuple[str, ...]
    setup_commands: Tuple[str, ...]

    @property
    def record_id(self) -> str:
        return f"bip_{self.project}_{self.bug_id}"

    def documented_test_files(self) -> List[str]:
        """The `test_file` list from bug.info, as repo-relative slash paths."""
        if not self.test_file:
            return []
        return [
            part.strip().replace("\\", "/")
            for part in self.test_file.split(";")
            if part.strip()
        ]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["patch"] = self.patch.to_dict()
        payload["record_id"] = self.record_id
        return payload


def _read(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def project_dir(bugsinpy_root: str | Path, project: str) -> Path:
    root = Path(bugsinpy_root)
    candidate = root / "projects" / project
    if not candidate.is_dir():
        raise BugsInPyLayoutError(
            f"No project '{project}' under {root / 'projects'}. Clone the metadata with "
            f"`git clone --depth 1 {BUGSINPY_URL} <root>` and pass its path."
        )
    return candidate


def list_projects(bugsinpy_root: str | Path) -> List[str]:
    """Every project directory in the metadata checkout, sorted."""
    projects = Path(bugsinpy_root) / "projects"
    if not projects.is_dir():
        raise BugsInPyLayoutError(
            f"{projects} does not exist. Expected a BugsInPy checkout; clone it with "
            f"`git clone --depth 1 {BUGSINPY_URL} {Path(bugsinpy_root)}`."
        )
    return sorted(p.name for p in projects.iterdir() if p.is_dir() and (p / "bugs").is_dir())


def list_bug_ids(bugsinpy_root: str | Path, project: str) -> List[int]:
    """Bug ids for one project, numerically sorted. Non-numeric entries are skipped."""
    bugs = project_dir(bugsinpy_root, project) / "bugs"
    return sorted(int(p.name) for p in bugs.iterdir() if p.is_dir() and p.name.isdigit())


def load_bug(bugsinpy_root: str | Path, project: str, bug_id: int) -> BugSpec:
    """Read one bug's metadata. Raises when a field a measurement needs is absent."""
    proj = project_dir(bugsinpy_root, project)
    bug = proj / "bugs" / str(bug_id)
    if not bug.is_dir():
        raise BugsInPyLayoutError(f"No bug {bug_id} for project {project} (looked in {bug}).")

    info_text = _read(bug / "bug.info")
    if info_text is None:
        raise BugsInPyLayoutError(f"{bug / 'bug.info'} is missing; the bug cannot be checked out.")
    info = parse_info_file(info_text)

    project_info = parse_info_file(_read(proj / "project.info") or "")
    github_url = project_info.get("github_url", "").strip().rstrip("/")
    if not github_url:
        raise BugsInPyLayoutError(f"{proj / 'project.info'} has no github_url.")

    for required in ("buggy_commit_id", "fixed_commit_id"):
        if not info.get(required):
            raise BugsInPyLayoutError(
                f"{bug / 'bug.info'} has no {required}. Without both commits there is "
                "nothing to compare, and this adapter does not invent a parent."
            )

    run_test_text = _read(bug / "run_test.sh") or ""
    documented, unreadable = parse_run_test(run_test_text)

    requirements = tuple(
        line.strip()
        for line in (_read(bug / "requirements.txt") or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    setup_commands = tuple(
        line.strip()
        for line in (_read(bug / "setup.sh") or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )

    return BugSpec(
        project=project,
        bug_id=bug_id,
        github_url=github_url,
        python_version=info.get("python_version", ""),
        buggy_commit=info["buggy_commit_id"],
        fixed_commit=info["fixed_commit_id"],
        test_file=info.get("test_file") or None,
        pythonpath=info.get("pythonpath") or None,
        documented_failing_tests=tuple(dict.fromkeys(documented)),
        unreadable_test_commands=tuple(unreadable),
        patch=parse_bug_patch(_read(bug / "bug_patch.txt") or ""),
        pinned_requirements=requirements,
        setup_commands=setup_commands,
    )


def load_project(bugsinpy_root: str | Path, project: str) -> List[BugSpec]:
    """Every readable bug for one project, in id order."""
    specs: List[BugSpec] = []
    for bug_id in list_bug_ids(bugsinpy_root, project):
        specs.append(load_bug(bugsinpy_root, project, bug_id))
    return specs


@dataclass(frozen=True)
class ProjectProfile:
    """
    How to run one project's suite, recorded from an observed run rather than guessed.

    BugsInPy's own framework answers this with per-bug shell scripts that assume the
    pinned interpreter and a Linux box. This project needs the whole universe rather
    than the one failing test, which the scripts never collect, so the few facts that
    differ per project are written down here and stamped onto every record.

    `python_files` exists because tqdm names its tests `tests_*.py` and configured
    nothing: pytest collects such a file when it is named on the command line, which
    is all BugsInPy ever does, and collects nothing at all when pointed at the
    directory. A universe of zero is the failure this field prevents.
    """

    package: str
    test_paths: Tuple[str, ...]
    python_files: Optional[str] = None
    extra_requirements: Tuple[str, ...] = ()
    editable_install: bool = True
    notes: str = ""


# Only projects whose suite has actually been collected on this machine appear here.
# The CLI refuses a project with no profile rather than guessing its layout, because a
# guess that collects nothing looks exactly like a project with no tests.
PROJECT_PROFILES: Dict[str, ProjectProfile] = {
    "tqdm": ProjectProfile(
        package="tqdm",
        test_paths=("tqdm/tests",),
        python_files="tests_*.py",
        extra_requirements=("nose",),
        notes=(
            "Upstream ran these under nose, so tests_tqdm.py imports nose.with_setup "
            "and the files do not match pytest's default python_files."
        ),
    ),
    "PySnooper": ProjectProfile(
        package="pysnooper",
        test_paths=("tests",),
        extra_requirements=("python_toolbox", "decorator"),
        notes="Both extras come from the bugs' own setup.sh.",
    ),
    "cookiecutter": ProjectProfile(
        package="cookiecutter",
        test_paths=("tests",),
        extra_requirements=("pytest-mock", "freezegun"),
        notes="run_test.sh drives tox; the underlying runner is pytest.",
    ),
    "httpie": ProjectProfile(
        package="httpie",
        test_paths=("tests",),
        extra_requirements=("pytest-httpbin", "responses"),
        notes="",
    ),
    "thefuck": ProjectProfile(
        package="thefuck",
        test_paths=("tests",),
        extra_requirements=("pytest-mock", "mock"),
        notes="",
    ),
    "youtube-dl": ProjectProfile(
        package="youtube_dl",
        test_paths=("test",),
        extra_requirements=(),
        notes=(
            "Stdlib only. test/test_download.py reaches the network and is excluded by "
            "the baseline screen when it fails, not by an allow-list here."
        ),
    ),
}


def profile_for(project: str) -> ProjectProfile:
    profile = PROJECT_PROFILES.get(project)
    if profile is None:
        raise BugsInPyLayoutError(
            f"No run profile for '{project}'. A profile records the importable package "
            f"name, where the tests live and any test-time dependency the project's own "
            f"install does not pull in -- facts that have to be observed once per "
            f"project. Add one to PROJECT_PROFILES after collecting its suite by hand; "
            f"guessing produces an empty universe that reads like a passing measurement. "
            f"Profiled so far: {', '.join(sorted(PROJECT_PROFILES))}."
        )
    return profile


# ----------------------------------------------------------------------
# Step 2: measurement
# ----------------------------------------------------------------------


class EnvironmentSetupError(RuntimeError):
    """A checkout could not be fetched, installed, or imported."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def venv_python(venv_dir: Path) -> Path:
    """Interpreter inside a venv, for whichever platform built it."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@contextmanager
def pytest_addopts(extra: str) -> Iterator[None]:
    """
    Temporarily extend PYTEST_ADDOPTS for child pytest processes.

    `SafeTestExecutor` builds its own command line and passes no `env=`, so the
    child inherits this process's environment: this is the seam through which a
    per-project `-o python_files=...` reaches pytest without the executor
    growing a parameter that only BugsInPy needs. `-o addopts=` in the
    executor's command neutralises the *ini* addopts and leaves PYTEST_ADDOPTS
    alone, so the two do not fight.
    """
    if not extra:
        yield
        return

    key = "PYTEST_ADDOPTS"
    previous = os.environ.get(key)
    os.environ[key] = f"{previous} {extra}".strip() if previous else extra
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@dataclass
class CommandResult:
    """One external command, kept so a failure can be published verbatim."""

    args: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, limit: int = 600) -> str:
        """Last of stderr, falling back to stdout: pip reports both ways."""
        text = (self.stderr or "").strip() or (self.stdout or "").strip()
        return text[-limit:]


def run_command(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """
    Run one command with no shell, capturing both streams.

    No `shell=True` anywhere in this module: every argument here is either a
    literal or a value read out of third-party metadata (commit ids, project
    names, requirement strings), and a shell would make that metadata
    executable. Commit ids are additionally validated by `_require_sha`.
    """
    started = time.time()
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            args=list(args),
            returncode=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=f"timed out after {timeout}s",
            duration=time.time() - started,
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(
            args=list(args),
            returncode=127,
            stdout="",
            stderr=str(exc),
            duration=time.time() - started,
        )

    return CommandResult(
        args=list(args),
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration=time.time() - started,
    )


_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _require_sha(value: str, label: str) -> str:
    if not _SHA_RE.match(value or ""):
        raise BugsInPyLayoutError(
            f"{label} is not a commit id: {value!r}. It is read from BugsInPy "
            f"metadata and handed to git, so it is checked rather than trusted."
        )
    return value


def canonical_test_key(test_id: str) -> Tuple[str, str]:
    """
    Reduce a test id to (module path, test name) for comparison only.

    Three things differ between the id BugsInPy documents and the id the JUnit
    report yields, none of which mean a different test: the `.py` suffix (present
    when pytest fills in the `file` attribute, absent when the id was rebuilt
    from a dotted classname), the class segment (the JUnit parser writes
    `file::method` and drops the class), and the parametrisation suffix
    (`test_x[1-2]` is one case of `test_x`). Comparing raw strings across that
    gap reports every confirmed bug as unconfirmed.

    Stored ids are never rewritten by this: the record publishes what the runner
    reported. This key exists solely to decide whether two names refer to the
    same test.
    """
    ident = test_id.strip().replace("\\", "/")
    head, _, tail = ident.partition("::")
    if head.endswith(".py"):
        head = head[: -len(".py")]
    last = tail.split("::")[-1] if tail else ""
    return head, last.split("[")[0]


def match_documented(
    documented: Sequence[str], candidates: Iterable[str]
) -> Dict[str, List[str]]:
    """
    Map each documented test id onto the measured ids it refers to.

    A documented id with no `::` names a whole file, which BugsInPy does use, so
    it matches every measured test in that file. An empty list means the
    documented test was not among the candidates -- reported as a disagreement,
    never resolved by picking something close.
    """
    by_key: Dict[Tuple[str, str], List[str]] = {}
    by_head: Dict[str, List[str]] = {}
    for candidate in candidates:
        head, name = canonical_test_key(candidate)
        by_key.setdefault((head, name), []).append(candidate)
        by_head.setdefault(head, []).append(candidate)

    matches: Dict[str, List[str]] = {}
    for doc_id in documented:
        head, name = canonical_test_key(doc_id)
        if name:
            matches[doc_id] = sorted(by_key.get((head, name), []))
        else:
            matches[doc_id] = sorted(by_head.get(head, []))
    return matches


class BugsInPyHarvester:
    """
    Measures BugsInPy bugs into the record schema `data/harvest/*/mutants.jsonl` uses.

    One bug is measured the way one mutant is: profile the passing revision to
    establish a universe, then run the defective revision once and record which
    universe members broke. The difference is that a mutant's two revisions
    differ by one line this project wrote, while a bug's two revisions are two
    commits someone else pushed -- so each revision needs its own checkout, and
    the checkout has to be installed before its own tests can import it.
    """

    def __init__(
        self,
        bugsinpy_root: Path,
        workspace: Path,
        output_dir: Path,
        baseline_runs: int = 2,
        suite_timeout: int = 900,
        install_timeout: int = 1200,
        git_timeout: int = 900,
        allow_interpreter_mismatch: bool = False,
        install_pinned_requirements: bool = False,
        base_python: Optional[str] = None,
    ):
        """
        Args:
            bugsinpy_root: Clone of the BugsInPy metadata repository.
            workspace: Where per-bug checkouts and virtualenvs are built. Not
                tracked by git: it is an input, reproducible from the pinned
                metadata commit recorded in the output.
            output_dir: Where records and per-bug baselines are written.
            baseline_runs: Runs at the fixed commit. Two is the minimum that can
                detect flakiness at all; the mutation harvest uses three because
                it reuses one baseline across hundreds of mutants, whereas here
                every bug pays for its own.
            suite_timeout: Per suite run.
            install_timeout: Per pip invocation. 2019-era projects sometimes
                build a wheel from source, which is slow but not hung.
            allow_interpreter_mismatch: Measure even when the interpreter is not
                the pinned one. Off by default. When on, the mismatch is stamped
                on every record rather than being silently tolerated.
            install_pinned_requirements: Attempt the bug's frozen requirements
                file. Off by default because those pins were resolved for the
                pinned interpreter and mostly cannot build under a newer one;
                the record says which source was used.
            base_python: Interpreter used to create each venv. Defaults to the
                one running this code.
        """
        self.bugsinpy_root = Path(bugsinpy_root).resolve()
        self.workspace = Path(workspace).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.baseline_runs = max(1, int(baseline_runs))
        self.suite_timeout = suite_timeout
        self.install_timeout = install_timeout
        self.git_timeout = git_timeout
        self.allow_interpreter_mismatch = allow_interpreter_mismatch
        self.install_pinned_requirements = install_pinned_requirements
        self.base_python = str(Path(base_python).resolve()) if base_python else sys.executable

        if not (self.bugsinpy_root / "projects").is_dir():
            raise BugsInPyLayoutError(
                f"{self.bugsinpy_root} has no projects/ directory. Clone the metadata "
                f"repository first: git clone {BUGSINPY_URL}"
            )

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Provenance of the metadata itself
    # ------------------------------------------------------------------

    def metadata_commit(self) -> Optional[str]:
        """
        The BugsInPy commit these records were read from.

        BugsInPy is a live repository: bugs get added and test commands get
        corrected. A record that does not name the metadata commit cannot be
        reproduced, only approximated.
        """
        result = run_command(
            ["git", "-C", str(self.bugsinpy_root), "rev-parse", "HEAD"], timeout=60
        )
        return result.stdout.strip() if result.ok else None

    def interpreter_version(self, python_executable: str) -> Optional[str]:
        probe = "import sys; print('.'.join(map(str, sys.version_info[:3])))"
        result = run_command([python_executable, "-c", probe], timeout=120)
        return result.stdout.strip() if result.ok else None

    # ------------------------------------------------------------------
    # Per-bug directories
    # ------------------------------------------------------------------

    def bug_workspace(self, bug: BugSpec) -> Path:
        return self.workspace / bug.project / str(bug.bug_id)

    def checkout_dir(self, bug: BugSpec) -> Path:
        return self.bug_workspace(bug) / "repo"

    def venv_dir(self, bug: BugSpec) -> Path:
        return self.bug_workspace(bug) / "venv"

    def baseline_path(self, bug: BugSpec) -> Path:
        return self.output_dir / bug.project / "baselines" / f"{bug.bug_id}.json"

    # ------------------------------------------------------------------
    # Checkout
    # ------------------------------------------------------------------

    def ensure_repo(self, bug: BugSpec) -> Path:
        """
        Create (or reuse) a checkout able to reach both of the bug's commits.

        Two commits out of one project's history are all that is needed, so the
        clone is built by fetching those two objects rather than by cloning the
        whole history: pandas is a gigabyte, and 169 of the 501 bugs are pandas.
        """
        repo_dir = self.checkout_dir(bug)
        repo_dir.mkdir(parents=True, exist_ok=True)

        if not (repo_dir / ".git").is_dir():
            init = run_command(["git", "init", "-q"], cwd=repo_dir, timeout=120)
            if not init.ok:
                raise EnvironmentSetupError(f"git init failed in {repo_dir}: {init.tail()}")
            add = run_command(
                ["git", "remote", "add", "origin", bug.github_url], cwd=repo_dir, timeout=120
            )
            if not add.ok:
                raise EnvironmentSetupError(f"git remote add failed: {add.tail()}")

        commits = (("buggy_commit", bug.buggy_commit), ("fixed_commit", bug.fixed_commit))
        for label, sha in commits:
            _require_sha(sha, label)
            have = run_command(
                ["git", "cat-file", "-e", sha + "^{commit}"], cwd=repo_dir, timeout=120
            )
            if have.ok:
                continue
            fetch = run_command(
                ["git", "fetch", "--depth", "1", "--quiet", "origin", sha],
                cwd=repo_dir,
                timeout=self.git_timeout,
            )
            if not fetch.ok:
                raise EnvironmentSetupError(
                    f"Could not fetch {label} {sha[:10]} of {bug.github_url}: {fetch.tail()}"
                )

        return repo_dir

    def checkout_commit(self, bug: BugSpec, sha: str) -> str:
        """
        Move the checkout to one commit, discarding whatever the last suite did.

        `--force` is deliberate: the previous revision's suite may have written
        to tracked files (the mutation harvest measured one subject suite
        truncating its own fixture), and the next measurement has to start from
        the revision as published, not from that residue.
        """
        repo_dir = self.checkout_dir(bug)
        _require_sha(sha, "commit")
        result = run_command(
            ["git", "-c", "advice.detachedHead=false", "checkout", "--force", "--quiet", sha],
            cwd=repo_dir,
            timeout=self.git_timeout,
        )
        if not result.ok:
            raise EnvironmentSetupError(f"git checkout {sha[:10]} failed: {result.tail()}")

        head = run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=120)
        resolved = head.stdout.strip() if head.ok else ""
        if not resolved.startswith(sha[:7]):
            raise EnvironmentSetupError(
                f"Checkout is at {resolved[:10] or 'unknown'} after asking for {sha[:10]}."
            )
        return resolved

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def ensure_env(self, bug: BugSpec) -> Path:
        """Create the bug's virtualenv if it is not already there."""
        venv = self.venv_dir(bug)
        python = venv_python(venv)
        if python.is_file():
            return python

        result = run_command(
            [self.base_python, "-m", "venv", str(venv)], timeout=self.install_timeout
        )
        if not result.ok or not python.is_file():
            raise EnvironmentSetupError(f"Could not create a virtualenv at {venv}: {result.tail()}")
        return python

    def install_checkout(
        self, bug: BugSpec, profile: ProjectProfile, python: Path
    ) -> Dict[str, Any]:
        """
        Install the checkout and its test-time dependencies into the bug's venv.

        The install is what makes `import <package>` resolve to the checkout, so
        it is not optional and its failure is not recoverable by running the
        suite anyway: pytest would then import whatever copy the ambient
        environment holds, and every test would pass at both commits.

        `requirements_source` is recorded because it is a real methodological
        choice. BugsInPy freezes a full environment per bug, resolved against a
        2019-2020 interpreter; those pins mostly cannot build here, so by default
        this installs the project itself plus the profile's named extras and lets
        pip resolve the rest. That is a weaker reproduction than BugsInPy
        intends, and the record says so rather than implying the frozen file was
        honoured.
        """
        steps: List[Dict[str, Any]] = []

        def _pip(*args: str, label: str) -> CommandResult:
            result = run_command(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check", *args],
                cwd=self.checkout_dir(bug),
                timeout=self.install_timeout,
            )
            steps.append(
                {
                    "label": label,
                    "args": list(args),
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "tail": "" if result.ok else result.tail(400),
                }
            )
            return result

        requirements_source = "profile"
        if self.install_pinned_requirements and bug.pinned_requirements:
            req_path = self.bug_workspace(bug) / "requirements_pinned.txt"
            req_path.write_text(NEWLINE.join(bug.pinned_requirements), encoding="utf-8")
            frozen = _pip("-r", str(req_path), label="pinned_requirements")
            requirements_source = "bugsinpy_frozen" if frozen.ok else "bugsinpy_frozen_failed"

        project_args = ["-e", "."] if profile.editable_install else ["."]
        project = _pip(*project_args, label="project")
        if not project.ok:
            raise EnvironmentSetupError(
                f"pip install of the checkout failed for {bug.record_id}: {project.tail()}"
            )

        runner = _pip("pytest", label="pytest")
        if not runner.ok:
            raise EnvironmentSetupError(f"pip install pytest failed: {runner.tail()}")

        for requirement in profile.extra_requirements:
            extra = _pip(requirement, label="extra:" + requirement)
            if not extra.ok:
                raise EnvironmentSetupError(
                    f"pip install {requirement} failed (the profile lists it as required "
                    f"for this suite to run at all): {extra.tail()}"
                )

        probe = run_command([str(python), "-m", "pytest", "--version"], timeout=300)
        pytest_version = None
        if probe.ok and probe.stdout.strip():
            pytest_version = probe.stdout.strip().splitlines()[0]

        return {
            "requirements_source": requirements_source,
            "pytest_version": pytest_version,
            "steps": steps,
        }

    def verify_import_provenance(
        self, bug: BugSpec, profile: ProjectProfile, python: Path
    ) -> Dict[str, Any]:
        """
        Require that the venv imports the checkout, not an ambient copy.

        Same check the mutation harness runs, for the same reason and with more
        at stake here: these package names (`tqdm`, `httpie`) are ordinary PyPI
        packages that may well be installed globally on the machine doing the
        measuring. If the tests import that copy, the fixed and the buggy commit
        are the same program, and the bug is recorded as breaking nothing.

        The probe runs in a temporary directory, not in the workspace: the
        workspace holds one directory per project, so `workspace/tqdm/` shadows
        the real `tqdm` as a namespace package and the probe reports the shadow.
        Measured on the first run of this harvester, which is why a namespace
        resolution is now its own rejection rather than a path comparison.
        """
        repo_dir = self.checkout_dir(bug).resolve()
        probe = NEWLINE.join(
            [
                "import importlib",
                "mod = importlib.import_module(%r)" % profile.package,
                "print(getattr(mod, '__file__', None) or '<namespace>')",
            ]
        )
        with tempfile.TemporaryDirectory() as neutral_cwd:
            result = run_command([str(python), "-c", probe], cwd=Path(neutral_cwd), timeout=300)

        if not result.ok:
            raise EnvironmentSetupError(
                f"{bug.record_id}: cannot import {profile.package} in its own venv: "
                f"{result.tail()}"
            )

        printed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        where = printed[-1] if printed else ""
        if not where:
            raise EnvironmentSetupError(
                f"{bug.record_id}: the import probe printed nothing for "
                f"{profile.package}. stderr: {result.tail(300)}"
            )
        if where == "<namespace>":
            raise EnvironmentSetupError(
                f"{bug.record_id}: {profile.package} resolved to a namespace package "
                f"with no module file, so nothing says which code the suite imports."
            )

        resolved = Path(where).resolve().parent
        inside = resolved == repo_dir or repo_dir in resolved.parents
        if not inside:
            raise EnvironmentSetupError(
                f"{bug.record_id}: {profile.package} imports {where}, which is outside "
                f"the checkout {repo_dir}. The suite would measure that copy, and both "
                f"commits would score identically."
            )
        return {"verified": True, "module": profile.package, "path": str(resolved)}

    # ------------------------------------------------------------------
    # Running the suite
    # ------------------------------------------------------------------

    def run_suite(self, bug: BugSpec, profile: ProjectProfile, python: Path) -> Any:
        """
        Run the whole suite once in the bug's own environment.

        The universe has to be the whole suite, not the documented failing test:
        a selector is scored on what it declined to run, so a measurement that
        only ever sees the failing test cannot say anything about reduction.
        This is the one thing BugsInPy's own `bugsinpy-test` never does -- it
        runs the single node id from run_test.sh.

        PYTHONPATH follows bug.info's `pythonpath` field, which is how several
        projects make themselves importable without an install.
        """
        executor = SafeTestExecutor(
            repo_root=str(self.checkout_dir(bug)),
            default_timeout=self.suite_timeout,
            python_executable=str(python),
        )

        addopts = f"-o python_files={profile.python_files}" if profile.python_files else ""
        previous_pythonpath = os.environ.get("PYTHONPATH")
        if bug.pythonpath:
            parts = [part for part in bug.pythonpath.split(";") if part]
            roots = [str(self.checkout_dir(bug) / part) for part in parts]
            joined = os.pathsep.join(roots)
            os.environ["PYTHONPATH"] = (
                f"{joined}{os.pathsep}{previous_pythonpath}" if previous_pythonpath else joined
            )
        try:
            with pytest_addopts(addopts):
                return executor.run_tests(
                    test_node_ids=list(profile.test_paths), timeout=self.suite_timeout
                )
        finally:
            if bug.pythonpath:
                if previous_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous_pythonpath

    def profile_baseline(
        self, bug: BugSpec, profile: ProjectProfile, python: Path, force: bool = False
    ) -> BaselineProfile:
        """
        Establish the universe at the fixed commit, exactly as the mutation harvest does.

        A test enters the universe only by passing every run, so a flaky test
        cannot be recorded as broken by the bug. The tree is restored between
        runs because some suites write to their own tracked fixtures, which would
        otherwise make run 2 a measurement of run 1's damage.

        Cached per bug: each bug is its own revision of its own project, so
        unlike the mutation harvest there is no single baseline to amortise over
        hundreds of records -- but a resumed run should not pay for it twice.
        """
        cache = self.baseline_path(bug)
        if cache.exists() and not force:
            logger.info(f"{bug.record_id}: reusing cached baseline {cache}")
            return BaselineProfile.from_dict(json.loads(cache.read_text(encoding="utf-8")))

        repo_dir = self.checkout_dir(bug)
        per_run_status: List[Dict[str, str]] = []
        per_run_duration: List[Dict[str, float]] = []
        suite_durations: List[float] = []
        self_damage: Set[str] = set()
        drift_checkable = tracked_modifications(repo_dir) is not None

        for run_index in range(self.baseline_runs):
            result = self.run_suite(bug, profile, python)

            if drift_checkable:
                drifted = tracked_modifications(repo_dir)
                if drifted:
                    self_damage.update(drifted)
                    logger.warning(
                        f"  run {run_index + 1} modified tracked file(s): "
                        f"{', '.join(drifted[:5])}. Restoring."
                    )
                    restore_tracked(repo_dir)

            if result.timed_out:
                raise EnvironmentSetupError(
                    f"{bug.record_id}: baseline run {run_index + 1} timed out after "
                    f"{self.suite_timeout}s at the fixed commit."
                )
            if not result.test_runs:
                raise EnvironmentSetupError(
                    f"{bug.record_id}: baseline run {run_index + 1} collected zero tests "
                    f"from {profile.test_paths}. Either the profile points at the wrong "
                    f"directory or the files do not match pytest's python_files.\n"
                    f"stderr: {result.stderr[:500]}"
                )

            per_run_status.append({r["test_id"]: r["status"] for r in result.test_runs})
            per_run_duration.append({r["test_id"]: float(r["duration"]) for r in result.test_runs})
            suite_durations.append(result.total_duration)
            logger.info(
                f"  {bug.record_id} baseline {run_index + 1}/{self.baseline_runs}: "
                f"{result.passed_count} passed, {result.failed_count} failed, "
                f"{result.skipped_count} skipped ({result.total_duration:.1f}s)"
            )

        stable_passing, excluded = classify_runs(per_run_status)
        mean_durations = {
            test_id: sum(d.get(test_id, 0.0) for d in per_run_duration) / len(per_run_duration)
            for test_id in stable_passing
        }
        baseline = BaselineProfile(
            stable_passing=stable_passing,
            excluded=excluded,
            mean_durations=mean_durations,
            suite_duration=sum(suite_durations) / len(suite_durations),
            n_runs=self.baseline_runs,
            self_damage=sorted(self_damage),
        )
        if not baseline.stable_passing:
            raise EnvironmentSetupError(
                f"{bug.record_id}: no test passed all {self.baseline_runs} runs at the "
                f"fixed commit, so nothing can be labelled."
            )

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
        return baseline

    # ------------------------------------------------------------------
    # The defect: buggy source, fixed tests
    # ------------------------------------------------------------------

    def changed_between_commits(self, bug: BugSpec) -> List[str]:
        """Repo-relative paths the fix commit touched, as git sees the pair."""
        result = run_command(
            ["git", "diff", "--name-only", bug.buggy_commit, bug.fixed_commit],
            cwd=self.checkout_dir(bug),
            timeout=self.git_timeout,
        )
        if not result.ok:
            raise EnvironmentSetupError(f"git diff between the two commits failed: {result.tail()}")
        return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})

    def buggy_is_parent_of_fixed(self, bug: BugSpec) -> Optional[bool]:
        """
        Whether the pair is (parent, child), which is what makes the diff the fix.

        When it holds, the difference between the two checkouts is one commit and
        the defect is exactly what the fix reversed. When it does not, the diff
        may carry unrelated work, and the record says so instead of implying a
        clean single-cause comparison.
        """
        result = run_command(
            ["git", "rev-parse", bug.fixed_commit + "^"],
            cwd=self.checkout_dir(bug),
            timeout=self.git_timeout,
        )
        if not result.ok:
            # A depth-1 fetch has no parent object locally; unknown, not false.
            return None
        parent = result.stdout.strip()
        return bool(parent) and parent.startswith(bug.buggy_commit[:7])

    def test_overlay_paths(
        self, bug: BugSpec, profile: ProjectProfile, changed: Sequence[str]
    ) -> List[str]:
        """
        The test files to take from the fixed commit into the buggy checkout.

        This is BugsInPy's own protocol, read out of framework/bin/bugsinpy-checkout:
        it resets to the buggy commit and then moves the fixed commit's copy of
        `test_file` in. The reason is that a fix usually ships with the test that
        catches it, so at the buggy commit alone that test does not exist -- and
        a universe measured at the fixed commit would lose members for a reason
        that has nothing to do with the defect.

        Two deliberate departures, both recorded on the record:

          * the declared `test_file` list is widened to every file the fix touched
            under the profile's test paths, so a conftest or a shared helper the
            fix also changed travels with it and the universe stays intact;
          * anything the source patch touches is excluded, because that is the
            defect itself and overlaying it would erase what is being measured.
        """
        declared = bug.documented_test_files()
        def _under_test_paths(path: str) -> bool:
            return any(
                path == root or path.startswith(root.rstrip("/") + "/")
                for root in profile.test_paths
            )

        under_test_paths = [path for path in changed if _under_test_paths(path)]
        source_patched = set(bug.patch.changed_files)
        overlay = {
            path.strip().replace("\\", "/")
            for path in (*declared, *under_test_paths)
            if path.strip() and path.strip().replace("\\", "/") not in source_patched
        }
        return sorted(overlay)

    def apply_defect(self, bug: BugSpec, overlay: Sequence[str]) -> Dict[str, Any]:
        """
        Put the checkout into the state the defect shipped in: buggy source, fixed tests.

        Returns what was actually overlaid, verified against git rather than
        assumed: a path that the fixed commit does not contain (renamed test
        files do occur) is reported, not silently dropped.
        """
        repo_dir = self.checkout_dir(bug)
        self.checkout_commit(bug, bug.buggy_commit)

        # BugsInPy runs `git clean -f -d` here. Keep egg-info: the editable
        # install's metadata lives in the tree, and a suite that asks
        # pkg_resources for its own version needs it.
        cleaned = run_command(
            ["git", "clean", "-fdq", "-e", "*.egg-info"], cwd=repo_dir, timeout=self.git_timeout
        )
        if not cleaned.ok:
            logger.warning(f"{bug.record_id}: git clean reported {cleaned.tail(200)}")

        applied: List[str] = []
        missing: List[str] = []
        for path in overlay:
            result = run_command(
                ["git", "checkout", bug.fixed_commit, "--", path],
                cwd=repo_dir,
                timeout=self.git_timeout,
            )
            (applied if result.ok else missing).append(path)

        if missing:
            logger.warning(
                f"{bug.record_id}: {len(missing)} declared test path(s) absent from the "
                f"fixed commit: {', '.join(missing[:3])}"
            )
        return {"overlaid": applied, "overlay_failed": missing}

    # ------------------------------------------------------------------
    # Scoring one defect
    # ------------------------------------------------------------------

    def classify_defect_run(self, baseline: BaselineProfile, result: Any) -> Dict[str, Any]:
        """
        Which universe tests the defect breaks, by the mutation harvest's own rule.

        A universe test that fails, errors, or vanishes from collection counts as
        killed. Vanishing counts because the test files are identical at both
        revisions -- the overlay guarantees that -- so a test that no longer
        collects did not disappear, it broke at import time, which is as
        observable a failure as an assertion.

        SKIPPED is not killed, matching the harvest: a test the defect causes to
        skip has not detected anything.
        """
        observed = {r["test_id"]: r["status"] for r in result.test_runs}
        universe = list(baseline.stable_passing)

        killed = [
            test_id
            for test_id in universe
            if observed.get(test_id, "MISSING") in FAILING_STATUSES or test_id not in observed
        ]
        missing = [test_id for test_id in universe if test_id not in observed]
        skipped = [test_id for test_id in universe if observed.get(test_id) == "SKIPPED"]
        kill_ratio = len(killed) / len(universe) if universe else float("nan")

        return {
            "killed": killed,
            "n_killed": len(killed),
            "kill_ratio": kill_ratio,
            "n_missing_at_defect": len(missing),
            "n_skipped_at_defect": len(skipped),
            "universe_size": len(universe),
            "collected_at_defect": len(result.test_runs),
        }

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def base_python_version(self) -> Optional[str]:
        """Version of the interpreter that will build every venv, probed once."""
        if not hasattr(self, "_base_python_version"):
            self._base_python_version = self.interpreter_version(self.base_python)
        return self._base_python_version

    def metadata_fields(self, bug: BugSpec) -> Dict[str, Any]:
        """
        Everything knowable about a bug without running anything.

        Present on every record, measured or not, so a skipped bug is still a
        published row with a reason rather than a silent absence. The diff counts
        are real numbers here: under mutation, 12 of the 32 features were constant
        because a mutant is always one line in one file, and a real fix is not.
        """
        if not hasattr(self, "_metadata_commit_cached"):
            self._metadata_commit_cached = self.metadata_commit()

        used = self.base_python_version()
        pinned = bug.python_version or None
        matches = bool(pinned and used and used == pinned)
        same_minor = bool(
            pinned and used and pinned.split(".")[:2] == used.split(".")[:2]
        )

        return {
            "mutant_id": bug.record_id,
            "repo": bug.project,
            "source": "bugsinpy",
            "project": bug.project,
            "bug_id": bug.bug_id,
            "github_url": bug.github_url,
            "buggy_commit": bug.buggy_commit,
            "fixed_commit": bug.fixed_commit,
            "operator": "real_bug",
            "file_path": bug.patch.first_file,
            "line": bug.patch.first_line,
            # The passing revision is the fixed one, so it plays the part the
            # pristine source plays under mutation, and the buggy side is the
            # defect. Naming them the other way round would invert every label.
            "original_snippet": bug.patch.fixed_snippet,
            "mutated_snippet": bug.patch.buggy_snippet,
            "changed_files": list(bug.patch.changed_files),
            "n_changed_files": len(bug.patch.changed_files),
            "lines_added": bug.patch.lines_added,
            "lines_deleted": bug.patch.lines_deleted,
            "documented_failing_tests": list(bug.documented_failing_tests),
            "unreadable_test_commands": list(bug.unreadable_test_commands),
            "python_version_pinned": pinned,
            "python_version_used": used,
            "interpreter_matches_pin": matches,
            "interpreter_matches_pin_minor": same_minor,
            "bugsinpy_commit": self._metadata_commit_cached,
            "produced_by": PRODUCED_BY,
        }

    def not_measured(self, bug: BugSpec, status: str, reason: str) -> Dict[str, Any]:
        """
        A published row for a bug that was not measured, carrying no labels.

        The absent fields are absent, not zero: `universe_size` of 0 and an empty
        `killed` list would average into any downstream summary as a bug that
        broke nothing, which is the fabrication this whole layer exists to stop.
        """
        record = self.metadata_fields(bug)
        record.update(
            {
                "status": status,
                "labels_measured": False,
                "excluded_from_labels": True,
                "exclusion_reason": reason,
                "universe_size": None,
                "n_killed": None,
                "kill_ratio": None,
                "killed": None,
                "suite_duration": None,
                "measured_at": utc_now(),
            }
        )
        return record

    def measure(self, bug: BugSpec, force: bool = False) -> Dict[str, Any]:
        """
        Measure one bug end to end and return its record.

        Order matters and is the same order the mutation harvest uses: prove the
        environment imports the checkout, establish the universe on the passing
        revision, then observe the defect once. Doing it the other way round --
        running the defect first and calling whatever fails the label -- cannot
        distinguish a bug from a test that was already broken.

        Environment failures are returned as records, not raised: a project that
        will not install on this machine is a fact about the harvest worth
        publishing, and 500 other bugs should not stop for it.
        """
        try:
            profile = profile_for(bug.project)
        except BugsInPyLayoutError as exc:
            return self.not_measured(bug, "no_profile", str(exc)[:400])

        used = self.base_python_version()
        pinned = bug.python_version or None
        if pinned and used != pinned and not self.allow_interpreter_mismatch:
            return self.not_measured(
                bug,
                "interpreter_mismatch",
                f"bug pins Python {pinned}; this machine offers {used}. Rerun with "
                f"allow_interpreter_mismatch to measure anyway (the mismatch is then "
                f"stamped on the record).",
            )

        try:
            self.ensure_repo(bug)
            fixed_head = self.checkout_commit(bug, bug.fixed_commit)
            python = self.ensure_env(bug)
            install = self.install_checkout(bug, profile, python)
            provenance = self.verify_import_provenance(bug, profile, python)
            baseline = self.profile_baseline(bug, profile, python, force=force)
            changed = self.changed_between_commits(bug)
            parentage = self.buggy_is_parent_of_fixed(bug)
            overlay = self.test_overlay_paths(bug, profile, changed)
            applied = self.apply_defect(bug, overlay)
            # The defect is now on disk. Reinstalling is not needed for an
            # editable install (the path is what is on sys.path), and is needed
            # when the packaging files themselves moved between the commits.
            if any(
                path in ("setup.py", "setup.cfg", "pyproject.toml") for path in changed
            ):
                install = self.install_checkout(bug, profile, python)
            defect_result = self.run_suite(bug, profile, python)
        except EnvironmentSetupError as exc:
            return self.not_measured(bug, "env_error", str(exc)[:600])

        scored = self.classify_defect_run(baseline, defect_result)
        enforced = getattr(defect_result, "timeout_enforced", True)

        matches = match_documented(bug.documented_failing_tests, baseline.stable_passing)
        killed_set = set(scored["killed"])
        observed_failing: List[str] = []
        documented_missing: List[str] = []
        documented_not_in_universe: List[str] = []
        for doc_id, candidates in matches.items():
            if not candidates:
                documented_not_in_universe.append(doc_id)
            elif any(candidate in killed_set for candidate in candidates):
                observed_failing.append(doc_id)
            else:
                documented_missing.append(doc_id)

        status, exclusion = self._defect_status(
            bug=bug,
            scored=scored,
            timed_out=defect_result.timed_out,
            timeout_enforced=enforced,
            documented_missing=documented_missing,
            documented_not_in_universe=documented_not_in_universe,
        )

        record = self.metadata_fields(bug)
        record.update(
            {
                "status": status,
                "labels_measured": status == "harvested",
                "excluded_from_labels": status != "harvested",
                "exclusion_reason": exclusion,
                "timed_out": defect_result.timed_out,
                "timeout_enforced": enforced,
                "exit_code": defect_result.exit_code,
                "suite_duration": round(defect_result.total_duration, 3),
                "baseline_suite_duration": round(baseline.suite_duration, 3),
                "baseline_runs": baseline.n_runs,
                "universe_size": scored["universe_size"],
                "n_killed": scored["n_killed"],
                "kill_ratio": (
                    None
                    if math.isnan(scored["kill_ratio"])
                    else round(scored["kill_ratio"], 4)
                ),
                "killed": scored["killed"],
                "n_missing_at_defect": scored["n_missing_at_defect"],
                "n_skipped_at_defect": scored["n_skipped_at_defect"],
                "collected_at_defect": scored["collected_at_defect"],
                # Flag, not a discard: a real defect that breaks four fifths of
                # the suite is still a real defect, but it is degenerate for
                # selection -- any test at all finds it -- so a consumer that
                # wants a harder set can filter on this.
                "broke_suite": (
                    not math.isnan(scored["kill_ratio"])
                    and scored["kill_ratio"] >= BROKE_SUITE_KILL_RATIO
                ),
                "documented_observed_failing": sorted(observed_failing),
                "documented_missing": sorted(documented_missing),
                "documented_not_in_universe": sorted(documented_not_in_universe),
                "documented_matches": {k: v for k, v in sorted(matches.items())},
                "buggy_is_parent_of_fixed": parentage,
                "fixed_head": fixed_head,
                "test_overlay": applied["overlaid"],
                "test_overlay_failed": applied["overlay_failed"],
                "test_overlay_widened": sorted(
                    set(applied["overlaid"]) - set(bug.documented_test_files())
                ),
                "excluded_from_universe": len(baseline.excluded),
                "baseline_self_damage": baseline.self_damage,
                "import_provenance": provenance,
                "requirements_source": install["requirements_source"],
                "pytest_version": install["pytest_version"],
                "install_steps": install["steps"],
                "measured_at": utc_now(),
            }
        )
        return record

    def _defect_status(
        self,
        bug: BugSpec,
        scored: Dict[str, Any],
        timed_out: bool,
        timeout_enforced: bool,
        documented_missing: Sequence[str],
        documented_not_in_universe: Sequence[str],
    ) -> Tuple[str, Optional[str]]:
        """
        Decide whether this measurement may be used as a label, and say why not.

        The cross-check is the point of this function. BugsInPy asserts which test
        the fix makes pass; this harvest observed which tests the defect breaks.
        When the two agree, the record is evidence about a real defect. When they
        do not, something in the chain is wrong -- the wrong commit pair, a test
        the newer interpreter skips, an environment that shims the failure away --
        and no amount of downstream averaging can tell which. The disagreement is
        published and the record is kept out of the labels, exactly as the
        mutation harvest keeps `timed_out` records out.
        """
        if not timeout_enforced:
            return TIMEOUT_BROKEN_STATUS, (
                f"the {self.suite_timeout}s suite timeout did not hold, so these "
                f"outcomes were read against a checkout something else still had open"
            )
        if timed_out:
            return "timed_out", f"the defect run exceeded {self.suite_timeout}s"
        if scored["collected_at_defect"] == 0:
            return "collection_failed", (
                "pytest collected nothing at the buggy commit, so every universe test "
                "counts as vanished and the documented test could not run either -- "
                "the cross-check would pass vacuously"
            )
        if not bug.documented_failing_tests:
            return "no_documented_tests", (
                "run_test.sh yielded no readable node id, so the measurement has "
                "nothing to be checked against"
            )
        if documented_not_in_universe:
            return "documented_mismatch", (
                f"documented failing test(s) absent from the measured universe: "
                f"{', '.join(sorted(documented_not_in_universe)[:3])}"
            )
        if documented_missing:
            return "documented_mismatch", (
                f"documented failing test(s) did not fail at the buggy commit: "
                f"{', '.join(sorted(documented_missing)[:3])}"
            )
        return "harvested", None

    # ------------------------------------------------------------------
    # Whole runs
    # ------------------------------------------------------------------

    @property
    def records_path(self) -> Path:
        return self.output_dir / "real_bugs.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "harvest_summary.json"

    def existing_records(self) -> Dict[str, Dict[str, Any]]:
        """Records already on disk, keyed by id, so a run can resume."""
        if not self.records_path.exists():
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping an unreadable line in real_bugs.jsonl")
                continue
            records[payload.get("mutant_id", "")] = payload
        records.pop("", None)
        return records

    def write_records(self, records: Sequence[Dict[str, Any]]) -> Path:
        """
        Rewrite the whole file from the records held in memory.

        Append would be cheaper, but a resumed run re-measures some ids and the
        file would then hold two rows for one bug -- and a mean over that file
        would weight those bugs twice.
        """
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=lambda r: (r.get("project", ""), r.get("bug_id", 0)))
        lines = [json.dumps(record, sort_keys=True) for record in ordered]
        self.records_path.write_text(NEWLINE.join(lines) + NEWLINE, encoding="utf-8")
        return self.records_path

    def harvest(
        self,
        projects: Sequence[str],
        bug_ids: Optional[Sequence[int]] = None,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Measure the requested bugs, writing after each one, and return the summary.

        Written after every bug rather than at the end: one bug costs an install
        plus three suite runs, so a run that dies at bug 20 must not lose 19
        measurements. This mirrors the mutation harvest's checkpointing, minus the
        need for a lock -- each bug owns its own checkout, so two harvests cannot
        corrupt each other's tree.
        """
        started = time.time()
        records = self.existing_records()
        attempted: List[str] = []

        for project in projects:
            specs = load_project(self.bugsinpy_root, project)
            if bug_ids:
                wanted = set(int(b) for b in bug_ids)
                specs = [spec for spec in specs if spec.bug_id in wanted]
            if limit is not None:
                specs = specs[:limit]

            logger.info(f"{project}: {len(specs)} bug(s) selected")
            for spec in specs:
                if spec.record_id in records and not force:
                    logger.info(f"{spec.record_id}: already recorded, skipping")
                    continue
                logger.info(f"--- {spec.record_id} ---")
                try:
                    record = self.measure(spec, force=force)
                except Exception as exc:  # noqa: BLE001 - one bug must not end the run
                    logger.exception(f"{spec.record_id}: unhandled failure")
                    record = self.not_measured(
                        spec, "harvest_error", f"{type(exc).__name__}: {exc}"[:600]
                    )
                records[spec.record_id] = record
                attempted.append(spec.record_id)
                self.write_records(list(records.values()))
                logger.info(
                    f"{spec.record_id}: {record['status']}"
                    + (
                        f" ({record['n_killed']}/{record['universe_size']} killed)"
                        if record.get("universe_size")
                        else ""
                    )
                )

        summary = self.summarise(records, attempted, time.time() - started, projects)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def summarise(
        self,
        records: Dict[str, Dict[str, Any]],
        attempted: Sequence[str],
        elapsed: float,
        projects: Sequence[str],
    ) -> Dict[str, Any]:
        """
        The run's own account of itself, including what it failed to measure.

        `projects_with_labels` is the number G6 is stated in -- external
        validation on at least three projects -- so it counts projects with at
        least one `harvested` record, not projects attempted. A summary that
        counted attempts would let a run of three failures read as success.
        """
        usable = [r for r in records.values() if r.get("status") == "harvested"]
        by_status: Dict[str, int] = {}
        for record in records.values():
            by_status[str(record.get("status"))] = by_status.get(str(record.get("status")), 0) + 1

        projects_with_labels = sorted({str(r.get("project")) for r in usable})
        per_project: Dict[str, Dict[str, Any]] = {}
        for record in records.values():
            project = str(record.get("project"))
            bucket = per_project.setdefault(
                project, {"records": 0, "harvested": 0, "universe_sizes": [], "n_killed": []}
            )
            bucket["records"] += 1
            if record.get("status") == "harvested":
                bucket["harvested"] += 1
                bucket["universe_sizes"].append(record.get("universe_size"))
                bucket["n_killed"].append(record.get("n_killed"))

        for bucket in per_project.values():
            sizes = [s for s in bucket.pop("universe_sizes") if isinstance(s, int)]
            kills = [k for k in bucket.pop("n_killed") if isinstance(k, int)]
            bucket["mean_universe_size"] = round(sum(sizes) / len(sizes), 2) if sizes else None
            bucket["mean_n_killed"] = round(sum(kills) / len(kills), 2) if kills else None

        return {
            "produced_by": PRODUCED_BY,
            "generated_at": utc_now(),
            "elapsed_seconds": round(elapsed, 1),
            "bugsinpy_url": BUGSINPY_URL,
            "bugsinpy_commit": self.metadata_commit(),
            "projects_requested": list(projects),
            "base_python": self.base_python,
            "base_python_version": self.base_python_version(),
            "allow_interpreter_mismatch": self.allow_interpreter_mismatch,
            "install_pinned_requirements": self.install_pinned_requirements,
            "baseline_runs": self.baseline_runs,
            "suite_timeout": self.suite_timeout,
            "n_records": len(records),
            "n_attempted_this_run": len(attempted),
            "n_labelled": len(usable),
            "labels_measured": bool(usable),
            "status_counts": by_status,
            "projects_with_labels": projects_with_labels,
            "n_projects_with_labels": len(projects_with_labels),
            "per_project": per_project,
            "records_path": str(self.records_path),
        }


def survey(bugsinpy_root: str | Path, projects: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Read every bug's metadata and report what could be measured, without running anything.

    This exists because the execution tier is expensive and unevenly available:
    every one of the 501 bugs pins a Python between 3.6.9 and 3.8.3, and a machine
    that has neither cannot honour a single pin. Knowing that from the metadata
    costs seconds; discovering it one failed install at a time costs hours.

    Reports per project: bug count, the pinned interpreters, how many bugs have a
    readable failing-test command, and whether a run profile exists.
    """
    root = Path(bugsinpy_root)
    names = list(projects) if projects else list_projects(root)
    per_project: Dict[str, Any] = {}
    versions: Dict[str, int] = {}
    total = 0
    readable = 0

    for project in sorted(names):
        specs = load_project(root, project)
        total += len(specs)
        project_versions: Dict[str, int] = {}
        project_readable = 0
        for spec in specs:
            project_versions[spec.python_version] = project_versions.get(spec.python_version, 0) + 1
            versions[spec.python_version] = versions.get(spec.python_version, 0) + 1
            if spec.documented_failing_tests:
                project_readable += 1
        readable += project_readable
        per_project[project] = {
            "n_bugs": len(specs),
            "python_versions": dict(sorted(project_versions.items())),
            "n_with_readable_test_command": project_readable,
            "has_run_profile": project in PROJECT_PROFILES,
        }

    return {
        "produced_by": PRODUCED_BY + " --survey",
        "generated_at": utc_now(),
        "bugsinpy_root": str(root),
        "n_projects": len(per_project),
        "n_bugs": total,
        "n_with_readable_test_command": readable,
        "pinned_python_versions": dict(sorted(versions.items())),
        "profiled_projects": sorted(PROJECT_PROFILES),
        "per_project": per_project,
    }
