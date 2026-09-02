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

import json
import random
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


@dataclass
class BaselineProfile:
    """Deterministic pass/fail profile of a clean checkout."""

    stable_passing: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)
    mean_durations: Dict[str, float] = field(default_factory=dict)
    suite_duration: float = 0.0
    n_runs: int = 0

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
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaselineProfile":
        return cls(
            stable_passing=list(payload.get("stable_passing", [])),
            excluded=dict(payload.get("excluded", {})),
            mean_durations=dict(payload.get("mean_durations", {})),
            suite_duration=float(payload.get("suite_duration", 0.0)),
            n_runs=int(payload.get("n_runs", 0)),
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

        self.executor = SafeTestExecutor(
            repo_root=str(self.repo_root),
            default_timeout=suite_timeout,
            python_executable=python_executable,
        )

    # ------------------------------------------------------------------
    # Step 0: prove the suite imports the code we are about to mutate
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
        """
        if self.baseline_path.exists() and not force:
            logger.info(f"Reusing cached baseline: {self.baseline_path}")
            return BaselineProfile.from_dict(json.loads(self.baseline_path.read_text()))

        logger.info(f"Profiling baseline for '{self.repo_name}' ({self.baseline_runs} runs)...")

        per_run_status: List[Dict[str, str]] = []
        per_run_duration: List[Dict[str, float]] = []
        suite_durations: List[float] = []

        for run_index in range(self.baseline_runs):
            result = self.executor.run_tests(timeout=self.suite_timeout)

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
        block, so an exception, timeout, or interrupt cannot leave the target
        repository in a mutated state.
        """
        target = self.repo_root / mutant.file_path
        original_bytes = target.read_bytes()
        try:
            write_source(target, mutant.mutated_source)
            yield
        finally:
            target.write_bytes(original_bytes)

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

        if result.timed_out:
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

    def _load_checkpoint(self) -> Set[str]:
        """Return mutant IDs already harvested, enabling interrupt-safe resume."""
        if not self.checkpoint_path.exists():
            return set()
        try:
            payload = json.loads(self.checkpoint_path.read_text())
            return set(payload.get("completed", []))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Unreadable checkpoint, starting fresh: {exc}")
            return set()

    def _save_checkpoint(self, completed: Set[str]) -> None:
        payload = {
            "repo": self.repo_name,
            "seed": self.seed,
            "completed": sorted(completed),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2))

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
        """
        completed = self._load_checkpoint() if resume else set()
        pending = [m for m in mutants if m.mutant_id not in completed]

        if completed:
            logger.info(f"Resuming: {len(completed)} done, {len(pending)} remaining.")

        summary = {
            "harvested": 0, "broke_suite": 0, "timed_out": 0,
            "killed_none": 0, "total_kills": 0,
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

                sink.write(json.dumps(record) + "\n")
                sink.flush()

                completed.add(mutant.mutant_id)
                self._save_checkpoint(completed)

                summary[record["status"]] = summary.get(record["status"], 0) + 1
                summary["total_kills"] += record.get("n_killed", 0)
                if record.get("n_killed", 0) == 0:
                    summary["killed_none"] += 1

                elapsed = time.time() - started
                rate = elapsed / index
                remaining = rate * (len(pending) - index)
                logger.info(
                    f"[{index}/{len(pending)}] {mutant.operator} "
                    f"{mutant.file_path}:{mutant.line} -> "
                    f"{record['status']}, killed {record.get('n_killed', 0)} "
                    f"(eta {remaining / 60:.0f}m)"
                )

        summary["elapsed_seconds"] = round(time.time() - started, 1)
        return summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        n_mutants: int = 250,
        resume: bool = True,
        verify_provenance: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full protocol end to end for this repository.

        `verify_provenance` is only turned off by unit tests, which run against
        synthetic repositories that are deliberately never installed anywhere.
        """
        provenance: Dict[str, Any] = {"verified": False, "reason": "not_requested", "modules": {}}
        if verify_provenance:
            provenance = self.verify_import_provenance()

        baseline = self.profile_baseline()
        candidates = self.collect_candidates()

        if not candidates:
            raise RuntimeError(f"No mutants generated for '{self.repo_name}'.")

        selected = self.sample_mutants(candidates, n_mutants)
        summary = self.harvest(selected, baseline, resume=resume)

        summary.update({
            "repo": self.repo_name,
            "interpreter": self.python_executable,
            "import_provenance": provenance,
            "universe_size": baseline.universe_size,
            "n_candidates": len(candidates),
            "n_sampled": len(selected),
            "mutants_path": str(self.mutants_path),
        })

        self.write_summary(summary)

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
