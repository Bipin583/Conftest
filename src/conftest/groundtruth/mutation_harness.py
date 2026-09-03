"""
ConfTest Mutation Harvesting Harness.

Produces regression-test-selection ground truth by injecting real faults
into a real repository and EXECUTING its real test suite. Every failure
label emitted by this module is the observed outcome of a pytest run.

Protocol
--------
1. BASELINE   Run the full suite N times on the clean checkout. Tests that
              pass every time form the stable universe; anything flaky,
              failing, or skipped is excluded from the dataset.
2. GENERATE   Enumerate single-node AST mutants of the source tree.
3. SAMPLE     Draw a bounded, operator-stratified subset.
4. HARVEST    For each mutant: apply -> run full suite -> parse JUnit XML
              -> restore the original file unconditionally.
5. LABEL      label_failed = 1 iff a stable-passing test FAILED or ERRORED
              under the mutant.

Randomness is used for exactly one purpose: choosing *which* mutants to
sample. It never touches a label.

Why the full suite runs every time
----------------------------------
Ground truth requires the outcome of *every* test, not just the ones a
selector would have picked. A partial label matrix cannot measure recall,
which is the metric the whole project turns on.
"""

import ctypes
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from conftest.groundtruth.mutators import Mutant, generate_mutants
from conftest.logging_config import get_logger
from conftest.tests.executor import SafeTestExecutor

logger = get_logger(__name__)

# Newline used inside f-strings that build multi-line diagnostics.
NEWLINE = chr(10)

# Importing a package is fast; a cap only exists so a broken environment cannot
# hang the harvest before it starts.
PROVENANCE_PROBE_TIMEOUT = 120


# Directories that hold a package rather than being one. A src-layout project
# puts its distribution inside one of these, so the importable name is the child,
# not the directory. `validators` ships a docstring-only src/__init__.py, which
# makes the directory look like a package it is not.
CONTAINER_DIR_NAMES = frozenset({"src", "lib", "sources"})

# The run summary is the only on-disk record that the labels came from an
# interpreter which provably imported the checkout. Downstream consumers refuse
# to build a dataset without it, so it is written by the harness rather than by
# whichever script happens to drive it.
SUMMARY_FILENAME = "harvest_summary.json"

# Directory names never treated as mutable source.
EXCLUDED_DIR_NAMES = frozenset({
    ".git", ".tox", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".eggs", "build", "dist", "docs", "examples", "benchmarks",
    "node_modules", ".idea", ".github",
})

# File names never mutated: test files change the oracle, not the code.
EXCLUDED_FILE_NAMES = frozenset({"setup.py", "conftest.py", "_version.py", "version.py"})

# A mutant that fails more than this fraction of the suite has almost
# certainly broken imports or collection rather than introduced a fault.
BROKE_SUITE_KILL_RATIO = 0.80

# Outcome statuses that count as a killed test.
FAILING_STATUSES = frozenset({"FAILED", "ERROR"})

# Status for a mutant whose suite run outlived the timeout that was supposed to
# bound it. Measured once, expensively: a 180s ceiling returned after 28,270s
# because a grandchild held an inherited pipe, and the mutated tree stayed live
# for 7h51m -- long enough for the suite to truncate one of its own fixtures.
# Such a record is not a slow label, it is an unattributable one.
TIMEOUT_BROKEN_STATUS = "timeout_broken"

# Journal written while a mutation is on disk. Surviving the run means the
# process died between applying a mutant and restoring the file, so the
# checkout can no longer be assumed to hold the revision that was screened.
PENDING_MUTATION_FILENAME = "pending_mutation.json"

# Read-only git queries take milliseconds; the cap only stops a wedged git
# from hanging a multi-hour harvest.
GIT_TIMEOUT_SECONDS = 60

# One harvest per output directory. Two of them sharing a checkout is not a
# slowdown, it is silent corruption: they apply mutants to the same files and
# each one's restore lands inside the other's suite run.
LOCK_FILENAME = "harvest.lock"
STILL_ACTIVE = 259
ERROR_INVALID_PARAMETER = 87
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class CheckoutNotPristine(RuntimeError):
    """
    The subject checkout differs from HEAD before a harvest starts.

    Mutants are enumerated from the files on disk and the baseline screen runs
    against those same files, so a modified checkout silently changes both
    which mutants exist and which tests count as stable. Refusing is the only
    way to keep every label traceable to a known revision.
    """


class HarvestAlreadyRunning(RuntimeError):
    """Another live harvest already owns this output directory."""


class SampleMismatch(RuntimeError):
    """
    The checkpoint records mutants the current sample does not contain.

    Enumerating candidates over a modified tree produces different mutant IDs,
    so resuming would append labels drawn from two different populations into
    one file. Raising keeps the two apart.
    """


def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Return stdout of a git command, or None when git cannot answer."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"git {args[0]} could not run in {repo_root}: {exc}")
        return None

    if proc.returncode != 0:
        logger.warning(
            f"git {args[0]} exited {proc.returncode} in {repo_root}: "
            f"{proc.stderr.strip()}"
        )
        return None

    return proc.stdout


def _git_paths(output: Optional[str]) -> List[str]:
    """Split newline-separated git output into a sorted, de-duplicated list."""
    if not output:
        return []
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def tracked_modifications(repo_root: Path) -> Optional[List[str]]:
    """
    Repo-relative paths of tracked files that differ from HEAD.

    Returns None to mean "unknown" -- the directory is not a git checkout, or
    git could not be run. A caller must not read None as "clean"; it is the
    absence of evidence, and the harness records it as such.
    """
    output = _git(repo_root, "diff", "--name-only", "HEAD")
    if output is None:
        return None
    return _git_paths(output)


def untracked_files(repo_root: Path) -> List[str]:
    """
    Repo-relative paths present on disk but unknown to git, ignores excluded.

    Untracked debris cannot change a mutant or a label -- nothing imports it --
    so it is reported rather than treated as a fault.
    """
    return _git_paths(_git(repo_root, "ls-files", "--others", "--exclude-standard"))


def restore_tracked(repo_root: Path) -> List[str]:
    """
    Discard modifications to tracked files, returning the paths restored.

    `git checkout HEAD -- .` rewrites the index and working tree for tracked
    paths only. Untracked files are deliberately left alone: `git clean` would
    delete work this harness never created and cannot judge.
    """
    modified = tracked_modifications(repo_root)
    if not modified:
        return []

    if _git(repo_root, "checkout", "HEAD", "--", ".") is None:
        raise CheckoutNotPristine(
            f"Cannot restore {repo_root}: git checkout failed. "
            f"Modified: {', '.join(modified)}"
        )

    still_dirty = tracked_modifications(repo_root)
    if still_dirty:
        raise CheckoutNotPristine(
            f"{repo_root} is still modified after restore: {', '.join(still_dirty)}"
        )

    logger.info(f"Restored {len(modified)} tracked file(s) in {repo_root}.")
    return modified


def process_is_alive(pid: Optional[int]) -> Optional[bool]:
    """
    Is `pid` a live process? None when that cannot be determined.

    The POSIX idiom `os.kill(pid, 0)` must not be used unguarded here: on
    Windows `os.kill` calls TerminateProcess, so the "signal 0 is only a
    liveness probe" assumption would kill the process it is asking about --
    and on this project that process is a running harvest.

    A pid can also be recycled, so a True answer means "something with that pid
    is running", not "the harvest is running". Callers therefore treat True as
    a reason to stop and ask rather than as proof, which is why --break-lock
    exists.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Only "no such process" is evidence of death; anything else
            # (access denied, for instance) leaves the question open.
            return False if ctypes.get_last_error() == ERROR_INVALID_PARAMETER else None
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, and it exists.
        return True
    except OSError:
        return None
    return True


def sample_digest(mutants: Sequence[Mutant]) -> str:
    """
    Fingerprint of a sample's membership, recorded alongside the labels.

    Two runs that agree on this string enumerated the same candidates from the
    same source, which is what makes their records poolable.
    """
    joined = NEWLINE.join(sorted(m.mutant_id for m in mutants))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class BaselineProfile:
    """Deterministic pass/fail profile of a clean checkout."""

    stable_passing: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)
    mean_durations: Dict[str, float] = field(default_factory=dict)
    suite_duration: float = 0.0
    n_runs: int = 0
    # Tracked files the suite modified while being screened. A suite that edits
    # its own fixtures makes its own outcomes depend on run order, which is why
    # the affected tests show up as flaky rather than as stable.
    self_damage: List[str] = field(default_factory=list)

    @property
    def universe_size(self) -> int:
        return len(self.stable_passing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "suite_duration": round(self.suite_duration, 3),
            "universe_size": self.universe_size,
            "n_excluded": len(self.excluded),
            "stable_passing": self.stable_passing,
            "excluded": self.excluded,
            "mean_durations": {k: round(v, 4) for k, v in self.mean_durations.items()},
            "self_damage": self.self_damage,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaselineProfile":
        return cls(
            stable_passing=list(payload.get("stable_passing", [])),
            excluded=dict(payload.get("excluded", {})),
            mean_durations=dict(payload.get("mean_durations", {})),
            suite_duration=float(payload.get("suite_duration", 0.0)),
            n_runs=int(payload.get("n_runs", 0)),
            self_damage=list(payload.get("self_damage", [])),
        )


def read_source(path: Path) -> str:
    """
    Read a source file preserving its exact line endings.

    ``newline=""`` disables universal-newline translation, so a CRLF file
    round-trips byte-identically through mutation and restore. Without this,
    a mutant would rewrite every line ending and stop being a one-line diff.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_source(path: Path, content: str) -> None:
    """Write source text without translating line endings."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def discover_source_files(repo_root: Path, source_dirs: Sequence[str]) -> List[Path]:
    """
    Collect mutable Python source files.

    Test files are excluded deliberately: mutating a test changes the oracle
    rather than the code under test, which would produce meaningless labels.
    """
    found: List[Path] = []

    for rel_dir in source_dirs:
        base = repo_root / rel_dir
        if not base.exists():
            logger.warning(f"Source directory not found, skipping: {base}")
            continue

        for path in sorted(base.rglob("*.py")):
            parts = set(path.parts)
            if parts & EXCLUDED_DIR_NAMES:
                continue
            if path.name in EXCLUDED_FILE_NAMES:
                continue
            name = path.name
            if name.startswith("test_") or name.endswith("_test.py"):
                continue
            if "tests" in path.relative_to(repo_root).parts:
                continue
            found.append(path)

    return found


def import_roots(repo_root: Path, source_dirs: Sequence[str]) -> List[str]:
    """
    Top-level module names the subject suite must import to reach mutated code.

    Needed because the harness has to prove that the interpreter running the
    suite imports the *checkout* and not some other copy of the same package.

    Derived from the files `discover_source_files` will actually mutate, not
    from the directory names, so the answer always describes the code under
    test. The three layouts among the screened repositories:

    * package at the root   -- ``sqlparse/sql.py``        -> ``sqlparse``
    * src layout            -- ``src/validators/uri.py``  -> ``validators``
    * single module         -- ``parse.py``               -> ``parse``

    A leading container directory is stripped. `validators` is why this is not
    optional: it ships a docstring-only ``src/__init__.py``, so reading the
    directory as a package would demand that ``import src`` succeed, which it
    never does -- the installed distribution exposes ``validators``.
    """
    roots: List[str] = []

    for path in discover_source_files(repo_root, source_dirs):
        try:
            parts = list(path.resolve().relative_to(repo_root).parts)
        except ValueError:
            continue
        if parts and parts[0] in CONTAINER_DIR_NAMES:
            parts = parts[1:]
        if not parts:
            continue
        if len(parts) == 1:
            stem = Path(parts[0]).stem
            if stem != "__init__":
                roots.append(stem)
        else:
            roots.append(parts[0])

    # Order-stable dedupe: keeps the manifest readable and the logs comparable.
    seen: Set[str] = set()
    unique: List[str] = []
    for name in roots:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


class MutationHarness:
    """Applies mutants to a repository and records measured test outcomes."""

    def __init__(
        self,
        repo_root: str,
        repo_name: str,
        source_dirs: Sequence[str],
        output_dir: str,
        suite_timeout: int = 300,
        baseline_runs: int = 3,
        seed: int = 42,
        python_executable: Optional[str] = None,
    ):
        """
        Args:
            repo_root: Path to the target repository checkout.
            repo_name: Short identifier recorded on every emitted record.
            source_dirs: Repo-relative directories holding mutable source.
            output_dir: Directory for mutants.jsonl, baseline.json, checkpoint.json.
            suite_timeout: Per-run wall-clock cap. Mutations can cause hangs.
            baseline_runs: Repeat count for the flakiness screen.
            seed: Seed for mutant sampling only; never affects labels.
            python_executable: Interpreter that runs the subject suite. Must be
                the environment the repository is installed into -- see
                `verify_import_provenance`. Defaults to the current interpreter,
                which is only correct when repo_root is this project.
        """
        self.repo_root = Path(repo_root).resolve()
        self.repo_name = repo_name
        self.source_dirs = list(source_dirs)
        self.output_dir = Path(output_dir).resolve()
        self.suite_timeout = suite_timeout
        self.baseline_runs = baseline_runs
        self.seed = seed
        self.python_executable = python_executable or sys.executable

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mutants_path = self.output_dir / "mutants.jsonl"
        self.baseline_path = self.output_dir / "baseline.json"
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.summary_path = self.output_dir / SUMMARY_FILENAME
        self.pending_mutation_path = self.output_dir / PENDING_MUTATION_FILENAME
        self.lock_path = self.output_dir / LOCK_FILENAME
        self.holds_lock = False

        self.executor = SafeTestExecutor(
            repo_root=str(self.repo_root),
            default_timeout=suite_timeout,
            python_executable=python_executable,
        )

    # ------------------------------------------------------------------
    # Step 0a: prove the tree on disk is the revision that was screened
    # ------------------------------------------------------------------

    def read_lock(self) -> Optional[Dict[str, Any]]:
        """Return the lock record, or None when the directory is unlocked."""
        if not self.lock_path.exists():
            return None
        try:
            return json.loads(self.lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            # A lock we cannot read is still a lock. Reporting it as absent
            # would be the one interpretation that permits a second harvest.
            return {"pid": None, "host": None, "unreadable": True}

    def acquire_lock(self, break_lock: bool = False) -> Dict[str, Any]:
        """
        Claim this output directory for the current process.

        Concurrent harvests over one checkout corrupt each other in a way that
        leaves no trace in the labels: both processes pick the same next mutant
        from the same seeded sample, and whichever restores first pulls the
        fault out from under the other's suite run. That was observed here on
        2026-09-03 -- a second run treated the first run's in-flight mutation
        journal as the residue of a dead run and discarded it. The journal
        cannot distinguish the two cases; a lock naming a live pid can.

        `break_lock` (--break-lock) forces the takeover for the case the lock
        outlives its process in a way the probe cannot confirm.
        """
        existing = self.read_lock()
        if existing is not None:
            pid = existing.get("pid")
            host = existing.get("host")
            same_host = host == socket.gethostname()
            alive = process_is_alive(pid) if same_host else None

            if break_lock:
                logger.warning(
                    f"--break-lock: taking over the lock held by pid {pid} on {host} "
                    f"(started {existing.get('started_at')}). If that harvest is still "
                    f"running, both runs are now writing to {self.output_dir}."
                )
            elif alive is False:
                logger.warning(
                    f"Taking over a stale lock from pid {pid} (started "
                    f"{existing.get('started_at')}): that process is gone."
                )
            else:
                detail = (
                    "it is running" if alive
                    else f"it cannot be checked from here (lock host {host!r})"
                )
                raise HarvestAlreadyRunning(
                    f"{self.repo_name}: another harvest holds {self.lock_path}."
                    + NEWLINE
                    + f"  - pid {pid} on {host}, started {existing.get('started_at')}, "
                    f"argv: {existing.get('argv')}"
                    + NEWLINE
                    + f"  - {detail}"
                    + NEWLINE
                    + "Two harvests over one checkout overwrite each other's mutations "
                    "and mislabel both runs."
                    + NEWLINE
                    + "Wait for it to finish, or re-run with --break-lock once it is "
                    "confirmed dead."
                )

        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": " ".join(sys.argv[1:]),
        }
        self.lock_path.write_text(json.dumps(payload, indent=2))
        self.holds_lock = True
        return payload

    def release_lock(self) -> None:
        """Drop the lock, but only if this process is the one holding it."""
        if self.holds_lock:
            self.lock_path.unlink(missing_ok=True)
            self.holds_lock = False

    def pending_mutation(self) -> Optional[Dict[str, Any]]:
        """Return the unfinished mutation recorded by a dead run, if any."""
        if not self.pending_mutation_path.exists():
            return None
        try:
            return json.loads(self.pending_mutation_path.read_text())
        except (OSError, json.JSONDecodeError):
            # The file exists, so a run died mid-mutant; unreadable contents
            # make that worse, not better.
            return {"mutant_id": "unreadable", "file_path": "unknown"}

    def assert_pristine_checkout(self, restore: bool = False) -> Dict[str, Any]:
        """
        Refuse to harvest a checkout that no longer matches HEAD.

        Two failure modes are covered, both observed in practice:

        * a previous run was hard-killed while a mutant was applied, leaving
          the fault in the tree. Later runs then enumerate candidates over
          mutated source -- a different sample -- and screen a baseline that
          silently contains an extra fault.
        * the subject suite modifies its own tracked fixtures while running.
          The next baseline screen blames the damage on the checkout and
          excludes the test as `failing_on_clean_checkout`, which is false.

        `restore` discards tracked modifications first (`--restore-checkout`).
        It is off by default: throwing away someone else's edits is not a
        decision this harness should make on its own.

        Returns the observation so the run summary can record what was seen,
        including the case where git is unavailable and nothing could be proven.
        """
        journal = self.pending_mutation()
        modified = tracked_modifications(self.repo_root)
        restored: List[str] = []

        if restore and modified:
            restored = restore_tracked(self.repo_root)
            modified = tracked_modifications(self.repo_root)

        untracked = untracked_files(self.repo_root)
        observation: Dict[str, Any] = {
            "verified": modified is not None,
            "tracked_modifications": modified if modified is not None else [],
            "untracked_files": untracked,
            # What was actually discarded, not merely whether the switch was
            # given: the summary is the audit trail for how a label was made.
            "restored": restored,
        }

        if modified is None:
            observation["reason"] = "not_a_git_checkout_or_git_unavailable"
            logger.warning(
                f"Cannot verify that {self.repo_root} is pristine: no usable git "
                f"checkout. Labels from this run are unverifiable at the revision level."
            )
        if untracked:
            logger.warning(
                f"{len(untracked)} untracked file(s) in {self.repo_root}: "
                f"{', '.join(untracked[:5])}. Untracked files are not imported, "
                f"so they are recorded rather than treated as a fault."
            )

        # A journal names the damage rather than being damage of its own: it
        # records a mutation that was applied and never reverted. So once
        # --restore-checkout has verifiably returned the tree to HEAD, the
        # journal is resolved, not an independent reason to refuse. Treating it
        # as one would mean the single flag documented as the remedy could never
        # repair the state it exists for -- every hard-killed run would have to
        # be cleared by deleting the file by hand, which is the undocumented
        # manual step this guard was written to remove.
        verified_clean = modified is not None and not modified
        problems: List[str] = []
        if journal is not None:
            if restore and verified_clean:
                observation["resolved_pending_mutation"] = journal
                logger.warning(
                    f"Discarded the mutation a dead run left applied: "
                    f"{journal.get('mutant_id')} in {journal.get('file_path')} "
                    f"(operator {journal.get('operator')}, applied "
                    f"{journal.get('applied_at')})."
                )
            else:
                unprovable = (
                    "" if modified is not None
                    else ", and git cannot confirm whether it is still applied"
                )
                observation["pending_mutation"] = journal
                problems.append(
                    f"a previous run died with mutant {journal.get('mutant_id')} applied "
                    f"to {journal.get('file_path')}{unprovable}"
                )
        if modified:
            problems.append(f"tracked files differ from HEAD: {', '.join(modified)}")

        if problems:
            raise CheckoutNotPristine(
                f"{self.repo_name}: refusing to harvest an unclean checkout at "
                f"{self.repo_root}."
                + NEWLINE
                + NEWLINE.join(f"  - {p}" for p in problems)
                + NEWLINE
                + "Mutants are enumerated from these files, so the sample and the "
                "baseline both depend on them."
                + NEWLINE
                + "Re-run with --restore-checkout to discard the modifications, or "
                "restore the checkout by hand."
            )

        self.pending_mutation_path.unlink(missing_ok=True)
        return observation

    # ------------------------------------------------------------------
    # Step 0b: prove the suite imports the code we are about to mutate
    # ------------------------------------------------------------------

    def verify_import_provenance(self) -> Dict[str, Any]:
        """
        Check that `self.python_executable` imports the checkout, not a copy.

        A mutation harvest is only measuring something if the package the tests
        import is the one on disk being mutated. When it is not, every mutant
        looks harmless and every label comes out 0 -- a clean-looking dataset
        that is entirely wrong, produced without drawing a single random number.

        This is not hypothetical. `sqlparse` is installed in this project's own
        ambient environment, so harvesting it under the default interpreter
        risked importing the released package while mutating the clone.

        The probe runs from a directory outside the repository, because
        `python -m pytest` puts the working directory on `sys.path` and that
        alone can make a root-layout checkout importable by accident. The
        property we require is stronger and does not depend on where pytest is
        invoked from: the environment itself resolves the package into
        `repo_root`, which is what `pip install -e .` guarantees and what a
        stale site-packages copy does not.

        Returns a per-module record for the harvest manifest. Raises
        RuntimeError if any module is missing or resolves outside the checkout.
        """
        roots = import_roots(self.repo_root, self.source_dirs)
        if not roots:
            logger.warning(
                f"{self.repo_name}: could not derive an import name from "
                f"source_dirs={self.source_dirs}; import provenance NOT verified."
            )
            return {"verified": False, "reason": "no_import_root_derived", "modules": {}}

        # "|" separates name from path: it is illegal in a Windows filename and
        # never appears in a module name, so it cannot occur inside either field.
        probe = textwrap.dedent(
            """
            import importlib, sys
            for name in sys.argv[1:]:
                try:
                    mod = importlib.import_module(name)
                    print(name + "|" + (getattr(mod, "__file__", None) or "<namespace>"))
                except Exception as exc:
                    print(name + "|ERROR: " + type(exc).__name__ + ": " + str(exc)[:200])
            """
        )

        # Neutral working directory: see the docstring. A temporary directory is
        # used rather than a path inside the repo so nothing is written into the
        # checkout that G3's clean-tree check would then see.
        with tempfile.TemporaryDirectory() as neutral_cwd:
            proc = subprocess.run(
                [self.python_executable, "-c", probe, *roots],
                cwd=neutral_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=PROVENANCE_PROBE_TIMEOUT,
            )

        modules: Dict[str, str] = {}
        for line in proc.stdout.splitlines():
            name, sep, where = line.partition("|")
            if sep:
                modules[name.strip()] = where.strip()

        problems: List[str] = []
        for name in roots:
            where = modules.get(name)
            if where is None:
                problems.append(
                    f"{name}: the interpreter printed no result for it"
                    + (f" (stderr: {proc.stderr.strip()[:200]})" if proc.stderr.strip() else "")
                )
                continue
            if where.startswith("ERROR: "):
                problems.append(f"{name}: not importable -- {where[len('ERROR: '):]}")
                continue
            if where == "<namespace>":
                problems.append(f"{name}: resolved to a namespace package with no file")
                continue
            try:
                Path(where).resolve().relative_to(self.repo_root)
            except ValueError:
                problems.append(
                    f"{name}: imports {where}, which is OUTSIDE the checkout at "
                    f"{self.repo_root} -- mutations to the checkout would be invisible"
                )

        if problems:
            detail = "".join(f"{NEWLINE}  - {problem}" for problem in problems)
            raise RuntimeError(
                f"{self.repo_name}: the suite would not exercise the mutated code."
                f"{detail}{NEWLINE}"
                f"Interpreter: {self.python_executable}{NEWLINE}"
                f"Install the checkout into the environment that runs the suite "
                f"(`pip install -e .`), then re-run. Harvesting under an interpreter "
                f"that imports a different copy of the package records every test as "
                f"passing for every mutant."
            )

        logger.info(
            f"{self.repo_name}: import provenance OK -- "
            + ", ".join(f"{name} -> {where}" for name, where in modules.items())
        )
        return {
            "verified": True,
            "interpreter": self.python_executable,
            "modules": modules,
        }

    # ------------------------------------------------------------------
    # Step 1: baseline and flakiness screen
    # ------------------------------------------------------------------

    def profile_baseline(self, force: bool = False) -> BaselineProfile:
        """
        Run the clean suite repeatedly to establish a deterministic universe.

        Only tests that PASS on every run are eligible for labelling. A flaky
        test would otherwise be recorded as mutation-killed when it simply
        failed on its own, injecting exactly the kind of fake label this
        pipeline exists to eliminate.

        The tree is restored between runs. One subject suite truncates a tracked
        fixture of its own when it runs; without a restore, run 1 damages the
        tree and runs 2 and 3 measure the damage, so the test looks like it fails
        on a clean checkout. Restoring makes each run an independent trial of the
        same revision, which is what repeating it is for -- the test then shows
        up as flaky, which is the truth about it.
        """
        if self.baseline_path.exists() and not force:
            logger.info(f"Reusing cached baseline: {self.baseline_path}")
            return BaselineProfile.from_dict(json.loads(self.baseline_path.read_text()))

        logger.info(f"Profiling baseline for '{self.repo_name}' ({self.baseline_runs} runs)...")

        per_run_status: List[Dict[str, str]] = []
        per_run_duration: List[Dict[str, float]] = []
        suite_durations: List[float] = []
        drift_checkable = tracked_modifications(self.repo_root) is not None
        self_damage: Set[str] = set()

        for run_index in range(self.baseline_runs):
            result = self.executor.run_tests(timeout=self.suite_timeout)

            if drift_checkable:
                drifted = tracked_modifications(self.repo_root)
                if drifted:
                    self_damage.update(drifted)
                    logger.warning(
                        f"  run {run_index + 1} modified tracked file(s): "
                        f"{', '.join(drifted)}. The suite edits its own checkout; "
                        f"restoring so the next run screens the same revision."
                    )
                    restore_tracked(self.repo_root)

            if result.timed_out:
                raise RuntimeError(
                    f"Baseline run {run_index + 1} timed out after {self.suite_timeout}s. "
                    "This repository is too slow to harvest; raise suite_timeout or "
                    "screen it out."
                )
            if not result.test_runs:
                raise RuntimeError(
                    f"Baseline run {run_index + 1} collected zero tests. "
                    f"Check the repo installs correctly.\nstderr: {result.stderr[:600]}"
                )

            per_run_status.append({r["test_id"]: r["status"] for r in result.test_runs})
            per_run_duration.append({r["test_id"]: float(r["duration"]) for r in result.test_runs})
            suite_durations.append(result.total_duration)

            logger.info(
                f"  run {run_index + 1}/{self.baseline_runs}: "
                f"{result.passed_count} passed, {result.failed_count} failed, "
                f"{result.skipped_count} skipped ({result.total_duration:.1f}s)"
            )

        # A test must appear, and pass, in every run to enter the universe.
        all_test_ids: Set[str] = set()
        for status_map in per_run_status:
            all_test_ids |= set(status_map)

        stable_passing: List[str] = []
        excluded: Dict[str, str] = {}

        for test_id in sorted(all_test_ids):
            statuses = [status_map.get(test_id) for status_map in per_run_status]

            if any(s is None for s in statuses):
                excluded[test_id] = "not_collected_in_every_run"
            elif all(s == "PASSED" for s in statuses):
                stable_passing.append(test_id)
            elif len(set(statuses)) > 1:
                excluded[test_id] = f"flaky:{'/'.join(str(s) for s in statuses)}"
            elif statuses[0] == "SKIPPED":
                excluded[test_id] = "skipped"
            else:
                excluded[test_id] = f"failing_on_clean_checkout:{statuses[0]}"

        mean_durations = {
            test_id: sum(d.get(test_id, 0.0) for d in per_run_duration) / len(per_run_duration)
            for test_id in stable_passing
        }

        profile = BaselineProfile(
            stable_passing=stable_passing,
            excluded=excluded,
            mean_durations=mean_durations,
            suite_duration=sum(suite_durations) / len(suite_durations),
            n_runs=self.baseline_runs,
            self_damage=sorted(self_damage),
        )

        if not profile.stable_passing:
            raise RuntimeError(
                f"No test passed all {self.baseline_runs} baseline runs for "
                f"'{self.repo_name}'. Nothing can be labelled."
            )

        flaky = sum(1 for reason in excluded.values() if reason.startswith("flaky"))
        logger.info(
            f"Baseline ready: {profile.universe_size} stable tests, "
            f"{len(excluded)} excluded ({flaky} flaky), "
            f"mean suite {profile.suite_duration:.1f}s"
        )

        self.baseline_path.write_text(json.dumps(profile.to_dict(), indent=2))
        return profile

    # ------------------------------------------------------------------
    # Step 2 and 3: generation and stratified sampling
    # ------------------------------------------------------------------

    def collect_candidates(self) -> List[Mutant]:
        """Enumerate every valid mutant across the repository's source tree."""
        source_files = discover_source_files(self.repo_root, self.source_dirs)
        logger.info(f"Scanning {len(source_files)} source files for mutants...")

        candidates: List[Mutant] = []
        for path in source_files:
            try:
                source = read_source(path)
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(f"Cannot read {path}: {exc}")
                continue

            rel_path = path.relative_to(self.repo_root).as_posix()
            candidates.extend(generate_mutants(source, rel_path))

        logger.info(f"Generated {len(candidates)} candidate mutants.")
        return candidates

    def sample_mutants(self, candidates: List[Mutant], n_mutants: int) -> List[Mutant]:
        """
        Draw a bounded subset, balanced across operator families.

        Unstratified sampling is dominated by constant mutations (roughly two
        thirds of all candidates, mostly string literals in log messages),
        which rarely kill a test. Round-robin selection across families keeps
        the dataset's fault distribution varied.

        This is the only randomness in the harness, and it decides which
        mutants to try, never what their outcome is.
        """
        if n_mutants <= 0 or n_mutants >= len(candidates):
            return list(candidates)

        rng = random.Random(self.seed)

        by_family: Dict[str, List[Mutant]] = {}
        for mutant in candidates:
            family = mutant.operator.split("_")[0]
            by_family.setdefault(family, []).append(mutant)

        for family in by_family:
            rng.shuffle(by_family[family])

        selected: List[Mutant] = []
        families = sorted(by_family)
        cursor = {family: 0 for family in families}

        while len(selected) < n_mutants:
            progressed = False
            for family in families:
                if len(selected) >= n_mutants:
                    break
                index = cursor[family]
                if index < len(by_family[family]):
                    selected.append(by_family[family][index])
                    cursor[family] += 1
                    progressed = True
            if not progressed:
                break

        distribution: Dict[str, int] = {}
        for mutant in selected:
            family = mutant.operator.split("_")[0]
            distribution[family] = distribution.get(family, 0) + 1
        logger.info(f"Sampled {len(selected)} mutants, families: {distribution}")

        # Stable order so a resumed run replays the identical sequence.
        return sorted(selected, key=lambda m: m.mutant_id)

    # ------------------------------------------------------------------
    # Step 4: harvest
    # ------------------------------------------------------------------

    @contextmanager
    def _applied_mutation(self, mutant: Mutant) -> Iterator[None]:
        """
        Apply a mutation, guaranteeing restoration of the original bytes.

        The original content is captured as bytes and rewritten in a finally
        block, so an exception or timeout cannot leave the target repository in
        a mutated state.

        A `finally` block cannot survive SIGKILL or a machine losing power, and
        one such death did leave a mutated file behind. So the mutation is also
        journalled to disk before it is applied and the journal removed after
        the restore: if the file outlives the process, the next run finds it and
        refuses to harvest instead of mutating an already-mutated tree.
        """
        target = self.repo_root / mutant.file_path
        original_bytes = target.read_bytes()
        self.pending_mutation_path.write_text(json.dumps({
            "mutant_id": mutant.mutant_id,
            "repo": self.repo_name,
            "file_path": mutant.file_path,
            "line": mutant.line,
            "operator": mutant.operator,
            "mutated_snippet": mutant.mutated_snippet,
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        try:
            write_source(target, mutant.mutated_source)
            yield
        finally:
            target.write_bytes(original_bytes)
            self.pending_mutation_path.unlink(missing_ok=True)

    def _harvest_one(self, mutant: Mutant, baseline: BaselineProfile) -> Dict[str, Any]:
        """Run the full suite under one mutant and label the measured outcome."""
        with self._applied_mutation(mutant):
            result = self.executor.run_tests(timeout=self.suite_timeout)

        observed = {r["test_id"]: r["status"] for r in result.test_runs}
        universe = baseline.stable_passing

        # A stable test is killed if it now fails, errors, or vanished from
        # collection (an import-level break is a real observable failure).
        killed = [
            test_id
            for test_id in universe
            if observed.get(test_id, "MISSING") in FAILING_STATUSES
            or test_id not in observed
        ]

        kill_ratio = len(killed) / len(universe) if universe else 0.0

        # A timeout that could not be enforced is its own status, and it
        # outranks the others. `timed_out` says "this mutant was slow, drop the
        # record"; this says "the tree under test was still alive when these
        # outcomes were read, and the next mutant may inherit what it wrote".
        # The first is a property of one record, the second contaminates the run.
        enforced = getattr(result, "timeout_enforced", True)
        if not enforced:
            status = TIMEOUT_BROKEN_STATUS
            logger.error(
                f"{mutant.mutant_id}: the {self.suite_timeout}s suite timeout did "
                f"not hold -- the run took {result.total_duration:.0f}s and the "
                f"tree was live throughout. Labelling it "
                f"{TIMEOUT_BROKEN_STATUS!r}: its outcomes were read against a "
                f"checkout something else still had open."
            )
        elif result.timed_out:
            status = "timed_out"
        elif kill_ratio >= BROKE_SUITE_KILL_RATIO:
            status = "broke_suite"
        else:
            status = "harvested"

        record: Dict[str, Any] = {
            "mutant_id": mutant.mutant_id,
            "repo": self.repo_name,
            "file_path": mutant.file_path,
            "line": mutant.line,
            "operator": mutant.operator,
            "original_snippet": mutant.original_snippet,
            "mutated_snippet": mutant.mutated_snippet,
            "status": status,
            "timed_out": result.timed_out,
            "timeout_enforced": enforced,
            "exit_code": result.exit_code,
            "suite_duration": round(result.total_duration, 3),
            "universe_size": len(universe),
            "n_killed": len(killed),
            "kill_ratio": round(kill_ratio, 4),
            # Only killed tests are stored. Every other test in the baseline
            # universe carries label 0 by construction, so this is compact
            # and lossless.
            "killed": killed,
        }
        return record

    def _load_checkpoint_payload(self) -> Dict[str, Any]:
        """Return the checkpoint as written, or an empty mapping."""
        if not self.checkpoint_path.exists():
            return {}
        try:
            payload = json.loads(self.checkpoint_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Unreadable checkpoint, starting fresh: {exc}")
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_checkpoint(self) -> Set[str]:
        """Return mutant IDs already harvested, enabling interrupt-safe resume."""
        return set(self._load_checkpoint_payload().get("completed", []))

    def _save_checkpoint(
        self,
        completed: Set[str],
        digest: Optional[str] = None,
        n_sampled: Optional[int] = None,
    ) -> None:
        payload = {
            "repo": self.repo_name,
            "seed": self.seed,
            # Which sample these IDs were drawn from. A later run that
            # enumerates different candidates can be caught instead of
            # appending labels from a second population to the same file.
            "sample_digest": digest,
            "n_sampled": n_sampled,
            "completed": sorted(completed),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2))

    def _verify_resumable(self, completed: Set[str], mutants: Sequence[Mutant]) -> None:
        """
        Refuse to resume when the checkpoint and the sample disagree.

        Growing `--mutants` is legitimate: stratified sampling is a prefix walk
        over seeded per-family orders, so a larger sample contains the smaller
        one. Completed IDs that are *absent* from the current sample are not
        legitimate -- they mean the candidate set itself changed, which happens
        when the tree was mutated or edited between runs.
        """
        orphans = sorted(completed - {m.mutant_id for m in mutants})
        if orphans:
            raise SampleMismatch(
                f"{self.repo_name}: checkpoint holds {len(completed)} completed "
                f"mutants, {len(orphans)} of which are absent from the current "
                f"{len(mutants)}-mutant sample."
                + NEWLINE
                + f"  first missing: {', '.join(orphans[:5])}"
                + NEWLINE
                + "The candidate set changed since those labels were recorded, so "
                "resuming would pool two different populations in one file."
                + NEWLINE
                + "Check the checkout is pristine, then either re-harvest from "
                "scratch (--no-resume) or truncate mutants.jsonl and checkpoint.json "
                "to the records drawn from this sample."
            )

        recorded = self._load_checkpoint_payload().get("sample_digest")
        current = sample_digest(mutants)
        if recorded and recorded != current:
            logger.info(
                f"Sample extended: checkpoint digest {recorded} -> {current} "
                f"({len(mutants)} mutants, {len(completed)} already done)."
            )

    def file_totals(self) -> Dict[str, Any]:
        """
        Count the labels actually on disk, not the ones this process wrote.

        A resumed run only touches the mutants it still owed, so its own tallies
        describe the increment. The summary is read later as a description of
        the dataset, so it has to report the file. The log is append-only and a
        re-harvest can write a mutant twice; the last record for an ID wins,
        which is what a reader of the file would conclude too.
        """
        totals: Dict[str, Any] = {
            "harvested": 0, "broke_suite": 0, "timed_out": 0, "error": 0,
            TIMEOUT_BROKEN_STATUS: 0,
            "killed_none": 0, "total_kills": 0, "checkout_drifted": 0,
        }
        if not self.mutants_path.exists():
            totals["records"] = 0
            totals["unique_mutants"] = 0
            return totals

        latest: Dict[str, Dict[str, Any]] = {}
        n_lines = 0
        with open(self.mutants_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                n_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping unreadable record in {self.mutants_path}")
                    continue
                latest[record.get("mutant_id", f"unnamed_{n_lines}")] = record

        for record in latest.values():
            status = record.get("status", "error")
            totals[status] = totals.get(status, 0) + 1
            totals["total_kills"] += record.get("n_killed", 0)
            if record.get("n_killed", 0) == 0:
                totals["killed_none"] += 1
            if record.get("checkout_drifted"):
                totals["checkout_drifted"] += 1

        totals["records"] = n_lines
        totals["unique_mutants"] = len(latest)
        return totals

    def harvest(
        self,
        mutants: Sequence[Mutant],
        baseline: BaselineProfile,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the suite once per mutant, appending one JSONL record each.

        Records are flushed immediately and the checkpoint is updated after
        every mutant, so an interrupted multi-hour run resumes without
        repeating work.

        After every mutant the checkout is re-checked against HEAD. A suite that
        edits its own tracked fixtures leaves damage the next mutant would
        inherit -- and that a later baseline screen would misread as a test
        failing on a clean checkout -- so the drift is recorded on the record and
        undone before the loop moves on.
        """
        completed = self._load_checkpoint() if resume else set()
        digest = sample_digest(mutants)

        if completed:
            self._verify_resumable(completed, mutants)

        pending = [m for m in mutants if m.mutant_id not in completed]

        if completed:
            logger.info(f"Resuming: {len(completed)} done, {len(pending)} remaining.")

        # One probe decides whether per-mutant drift detection is available at
        # all: a synthetic fixture repo is not a git checkout and cannot answer.
        drift_checkable = tracked_modifications(self.repo_root) is not None

        this_run = {
            "harvested": 0, "broke_suite": 0, "timed_out": 0,
            TIMEOUT_BROKEN_STATUS: 0,
            "killed_none": 0, "total_kills": 0, "checkout_drifted": 0,
        }
        started = time.time()

        with open(self.mutants_path, "a", encoding="utf-8") as sink:
            for index, mutant in enumerate(pending, start=1):
                try:
                    record = self._harvest_one(mutant, baseline)
                except Exception as exc:
                    logger.error(f"Harvest failed for {mutant.mutant_id}: {exc}")
                    record = {
                        "mutant_id": mutant.mutant_id,
                        "repo": self.repo_name,
                        "file_path": mutant.file_path,
                        "line": mutant.line,
                        "operator": mutant.operator,
                        "status": "error",
                        "error": str(exc),
                        "killed": [],
                        "n_killed": 0,
                    }

                if drift_checkable:
                    drifted = tracked_modifications(self.repo_root)
                    if drifted:
                        record["checkout_drifted"] = drifted
                        this_run["checkout_drifted"] += 1
                        logger.error(
                            f"{mutant.mutant_id} left the checkout modified: "
                            f"{', '.join(drifted)}. The suite edited its own tracked "
                            f"files, so this record may hold inherited damage rather "
                            f"than kills. Restoring before the next mutant."
                        )
                        restore_tracked(self.repo_root)

                sink.write(json.dumps(record) + "\n")
                sink.flush()

                completed.add(mutant.mutant_id)
                self._save_checkpoint(completed, digest=digest, n_sampled=len(mutants))

                this_run[record["status"]] = this_run.get(record["status"], 0) + 1
                this_run["total_kills"] += record.get("n_killed", 0)
                if record.get("n_killed", 0) == 0:
                    this_run["killed_none"] += 1

                elapsed = time.time() - started
                rate = elapsed / index
                remaining = rate * (len(pending) - index)
                logger.info(
                    f"[{index}/{len(pending)}] {mutant.operator} "
                    f"{mutant.file_path}:{mutant.line} -> "
                    f"{record['status']}, killed {record.get('n_killed', 0)} "
                    f"(eta {remaining / 60:.0f}m)"
                )

        this_run["elapsed_seconds"] = round(time.time() - started, 1)
        this_run["n_mutants"] = len(pending)

        # Top-level counts describe the label file; `this_run` describes what
        # this process contributed to it.
        summary: Dict[str, Any] = dict(self.file_totals())
        summary["this_run"] = this_run
        summary["n_completed"] = len(completed)
        summary["elapsed_seconds"] = this_run["elapsed_seconds"]
        summary["sample_digest"] = digest
        summary["drift_detection"] = "git" if drift_checkable else "unavailable"
        return summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        n_mutants: int = 250,
        resume: bool = True,
        verify_provenance: bool = True,
        restore_checkout: bool = False,
        reprofile_baseline: bool = False,
        break_lock: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full protocol end to end for this repository.

        `verify_provenance` is only turned off by unit tests, which run against
        synthetic repositories that are deliberately never installed anywhere.

        The checkout is checked before anything else: the baseline screen and
        the candidate enumeration both read the files on disk, so a tree that
        does not match HEAD changes the sample and the label universe silently.
        `restore_checkout` discards tracked modifications first, and
        `reprofile_baseline` re-runs the flakiness screen instead of trusting a
        cached one -- which is required after the checkout has been repaired,
        because the cached profile was screened against the damaged tree.

        The lock is taken before the checkout is even inspected, because
        `restore_checkout` is destructive to a concurrent run: it would discard
        the mutation another live harvest has applied. Refusing first is what
        makes the repair switch safe to hand out.
        """
        harvester = self.acquire_lock(break_lock=break_lock)
        try:
            checkout = self.assert_pristine_checkout(restore=restore_checkout)

            provenance: Dict[str, Any] = {
                "verified": False, "reason": "not_requested", "modules": {},
            }
            if verify_provenance:
                provenance = self.verify_import_provenance()

            baseline = self.profile_baseline(force=reprofile_baseline)
            candidates = self.collect_candidates()

            if not candidates:
                raise RuntimeError(f"No mutants generated for '{self.repo_name}'.")

            selected = self.sample_mutants(candidates, n_mutants)
            summary = self.harvest(selected, baseline, resume=resume)

            summary.update({
                "repo": self.repo_name,
                "interpreter": self.python_executable,
                "import_provenance": provenance,
                "checkout": checkout,
                "harvester": harvester,
                "universe_size": baseline.universe_size,
                "baseline_self_damage": baseline.self_damage,
                "n_candidates": len(candidates),
                "n_sampled": len(selected),
                "mutants_path": str(self.mutants_path),
            })

            self.write_summary(summary)
        finally:
            self.release_lock()

        logger.info(f"Harvest complete for '{self.repo_name}': {summary}")
        return summary

    def write_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Persist the run summary beside the labels.

        The dataset builder reads this back to prove that the interpreter which
        produced the labels resolved the package into the checkout. Without the
        file there is nothing to audit, so it is written even when provenance
        verification was skipped -- an unverified record is a fact the builder
        is entitled to refuse, not something to hide by omission.
        """
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return self.summary_path
