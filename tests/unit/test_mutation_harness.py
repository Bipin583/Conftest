"""
Unit tests for the ConfTest mutation harvesting harness.

Focus is on the safety-critical invariants rather than on pytest itself:

* a mutated file is ALWAYS restored, including on exception
* flaky and pre-failing tests never enter the labelled universe
* import-breaking mutants are flagged rather than silently labelled
* an interrupted harvest resumes without repeating or losing work
* test files are never mutated
* CRLF line endings survive a mutate/restore cycle
* the interpreter that runs the suite imports the checkout being mutated
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from conftest.groundtruth.mutation_harness import (
    BROKE_SUITE_KILL_RATIO,
    SUMMARY_FILENAME,
    BaselineProfile,
    MutationHarness,
    discover_source_files,
    import_roots,
    read_source,
    write_source,
)
from conftest.groundtruth.mutators import Mutant


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeResult:
    """Stand-in for PytestExecutionResult."""

    def __init__(
        self,
        statuses: Dict[str, str],
        timed_out: bool = False,
        duration: float = 1.0,
        exit_code: int = 0,
    ):
        self.test_runs = [
            {"test_id": tid, "status": st, "duration": 0.01, "failure_message": None}
            for tid, st in statuses.items()
        ]
        self.timed_out = timed_out
        self.total_duration = duration
        self.exit_code = exit_code
        self.stdout = ""
        self.stderr = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] == "PASSED")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] in ("FAILED", "ERROR"))

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.test_runs if r["status"] == "SKIPPED")


class FakeExecutor:
    """Returns a scripted sequence of results and counts invocations."""

    def __init__(self, sequence: List[FakeResult]):
        self.sequence = sequence
        self.calls = 0

    def run_tests(self, *args: Any, **kwargs: Any) -> FakeResult:
        result = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        return result


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal repository with a source package and a test package."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )
    (repo / "tests" / "test_core.py").write_text(
        "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    return repo


def _make_harness(repo: Path, tmp_path: Path, **kwargs: Any) -> MutationHarness:
    return MutationHarness(
        repo_root=str(repo),
        repo_name="fake_repo",
        source_dirs=["pkg"],
        output_dir=str(tmp_path / "out"),
        baseline_runs=kwargs.pop("baseline_runs", 3),
        seed=kwargs.pop("seed", 42),
        **kwargs,
    )


def _mutant_for(repo: Path, rel_path: str = "pkg/core.py") -> Mutant:
    original = (repo / rel_path).read_text()
    return Mutant(
        mutant_id="mut_test01",
        file_path=rel_path,
        line=2,
        operator="arith_+_to_-",
        original_snippet="a + b",
        mutated_snippet="(a - b)",
        mutated_source=original.replace("a + b", "(a - b)"),
    )


# --------------------------------------------------------------------------
# Restoration -- the single most important invariant
# --------------------------------------------------------------------------

def test_mutation_is_applied_then_restored(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    target = repo / "pkg" / "core.py"
    original = target.read_bytes()

    mutant = _mutant_for(repo)
    with harness._applied_mutation(mutant):
        assert "(a - b)" in target.read_text(), "mutation was not applied"

    assert target.read_bytes() == original, "original bytes were not restored"


def test_file_is_restored_even_when_body_raises(tmp_path):
    """A crash mid-harvest must never leave the target repository mutated."""
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    target = repo / "pkg" / "core.py"
    original = target.read_bytes()

    with pytest.raises(RuntimeError):
        with harness._applied_mutation(_mutant_for(repo)):
            raise RuntimeError("simulated crash during test execution")

    assert target.read_bytes() == original


def test_crlf_line_endings_survive_round_trip(tmp_path):
    """
    Universal-newline translation would rewrite every line ending, turning a
    one-line mutation into a whole-file diff.
    """
    repo = _make_repo(tmp_path)
    target = repo / "pkg" / "core.py"
    target.write_bytes(b"def add(a, b):\r\n    return a + b\r\n")
    original = target.read_bytes()

    harness = _make_harness(repo, tmp_path)
    source = read_source(target)
    assert "\r\n" in source, "read_source must preserve CRLF"

    mutant = Mutant(
        mutant_id="mut_crlf",
        file_path="pkg/core.py",
        line=2,
        operator="arith_+_to_-",
        original_snippet="a + b",
        mutated_snippet="(a - b)",
        mutated_source=source.replace("a + b", "(a - b)"),
    )

    with harness._applied_mutation(mutant):
        mutated = target.read_bytes()
        assert b"\r\r\n" not in mutated, "line endings were doubled"
        assert mutated.count(b"\r\n") == original.count(b"\r\n")

    assert target.read_bytes() == original


def test_write_source_does_not_translate_newlines(tmp_path):
    path = tmp_path / "sample.py"
    write_source(path, "a = 1\nb = 2\n")
    assert path.read_bytes() == b"a = 1\nb = 2\n"


# --------------------------------------------------------------------------
# Baseline flakiness screening
# --------------------------------------------------------------------------

def test_flaky_test_is_excluded_from_universe(tmp_path):
    """A test that flips between runs would be mislabelled as mutation-killed."""
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([
        FakeResult({"t::stable": "PASSED", "t::flaky": "PASSED"}),
        FakeResult({"t::stable": "PASSED", "t::flaky": "FAILED"}),
        FakeResult({"t::stable": "PASSED", "t::flaky": "PASSED"}),
    ])

    profile = harness.profile_baseline()

    assert profile.stable_passing == ["t::stable"]
    assert "t::flaky" in profile.excluded
    assert profile.excluded["t::flaky"].startswith("flaky")


def test_test_failing_on_clean_checkout_is_excluded(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([
        FakeResult({"t::ok": "PASSED", "t::broken": "FAILED"}),
    ] * 3)

    profile = harness.profile_baseline()

    assert profile.stable_passing == ["t::ok"]
    assert "failing_on_clean_checkout" in profile.excluded["t::broken"]


def test_skipped_test_is_excluded(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([
        FakeResult({"t::ok": "PASSED", "t::skip": "SKIPPED"}),
    ] * 3)

    profile = harness.profile_baseline()

    assert profile.stable_passing == ["t::ok"]
    assert profile.excluded["t::skip"] == "skipped"


def test_baseline_runs_the_suite_once_per_configured_repeat(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path, baseline_runs=3)
    harness.executor = FakeExecutor([FakeResult({"t::ok": "PASSED"})] * 3)

    harness.profile_baseline()

    assert harness.executor.calls == 3


def test_baseline_rejects_zero_collected_tests(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([FakeResult({})])

    with pytest.raises(RuntimeError, match="zero tests"):
        harness.profile_baseline()


def test_baseline_rejects_timeout(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([
        FakeResult({"t::ok": "PASSED"}, timed_out=True),
    ])

    with pytest.raises(RuntimeError, match="timed out"):
        harness.profile_baseline()


def test_baseline_rejects_fully_unstable_suite(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([
        FakeResult({"t::a": "PASSED"}),
        FakeResult({"t::a": "FAILED"}),
        FakeResult({"t::a": "PASSED"}),
    ])

    with pytest.raises(RuntimeError, match="No test passed all"):
        harness.profile_baseline()


def test_baseline_is_cached_and_reused(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([FakeResult({"t::ok": "PASSED"})] * 3)

    first = harness.profile_baseline()
    calls_after_first = harness.executor.calls
    second = harness.profile_baseline()

    assert harness.executor.calls == calls_after_first, "cached baseline re-ran the suite"
    assert second.stable_passing == first.stable_passing


def test_baseline_records_mean_durations(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.executor = FakeExecutor([FakeResult({"t::ok": "PASSED"})] * 3)

    profile = harness.profile_baseline()

    assert "t::ok" in profile.mean_durations
    assert profile.mean_durations["t::ok"] == pytest.approx(0.01)


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------

def test_killed_tests_are_labelled_from_observed_failures(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a", "t::b", "t::c"], n_runs=3)

    harness.executor = FakeExecutor([
        FakeResult({"t::a": "FAILED", "t::b": "PASSED", "t::c": "ERROR"}),
    ])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert sorted(record["killed"]) == ["t::a", "t::c"]
    assert record["n_killed"] == 2
    assert record["status"] == "harvested"


def test_test_vanishing_from_collection_counts_as_killed(tmp_path):
    """An import-level break is a real, observable failure."""
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a", "t::b"], n_runs=3)

    harness.executor = FakeExecutor([FakeResult({"t::a": "PASSED"})])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert record["killed"] == ["t::b"]


def test_mutant_killing_nothing_is_recorded_with_zero(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a", "t::b"], n_runs=3)

    harness.executor = FakeExecutor([
        FakeResult({"t::a": "PASSED", "t::b": "PASSED"}),
    ])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert record["n_killed"] == 0
    assert record["killed"] == []
    assert record["status"] == "harvested"


def test_suite_breaking_mutant_is_flagged(tmp_path):
    """Killing nearly everything means broken imports, not a realistic fault."""
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    universe = [f"t::{i}" for i in range(10)]
    baseline = BaselineProfile(stable_passing=universe, n_runs=3)

    harness.executor = FakeExecutor([
        FakeResult({tid: "ERROR" for tid in universe}),
    ])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert record["status"] == "broke_suite"
    assert record["kill_ratio"] >= BROKE_SUITE_KILL_RATIO


def test_timeout_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)

    harness.executor = FakeExecutor([
        FakeResult({"t::a": "PASSED"}, timed_out=True),
    ])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert record["status"] == "timed_out"
    assert record["timed_out"] is True


def test_record_omits_passing_tests_for_compactness(tmp_path):
    """Only killed tests are stored; label 0 is implied by the baseline universe."""
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a", "t::b", "t::c"], n_runs=3)

    harness.executor = FakeExecutor([
        FakeResult({"t::a": "FAILED", "t::b": "PASSED", "t::c": "PASSED"}),
    ])

    record = harness._harvest_one(_mutant_for(repo), baseline)

    assert record["killed"] == ["t::a"]
    assert record["universe_size"] == 3
    assert "t::b" not in json.dumps(record)


# --------------------------------------------------------------------------
# Harvest loop, checkpointing, resume
# --------------------------------------------------------------------------

def _fixed_mutants(repo: Path, count: int) -> List[Mutant]:
    original = (repo / "pkg" / "core.py").read_text()
    return [
        Mutant(
            mutant_id=f"mut_{i:03d}",
            file_path="pkg/core.py",
            line=2,
            operator="arith_+_to_-",
            original_snippet="a + b",
            mutated_snippet="(a - b)",
            mutated_source=original.replace("a + b", "(a - b)"),
        )
        for i in range(count)
    ]


def test_harvest_writes_one_record_per_mutant(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)
    harness.executor = FakeExecutor([FakeResult({"t::a": "FAILED"})])

    harness.harvest(_fixed_mutants(repo, 4), baseline)

    lines = harness.mutants_path.read_text().strip().split("\n")
    assert len(lines) == 4
    assert {json.loads(line)["mutant_id"] for line in lines} == {
        "mut_000", "mut_001", "mut_002", "mut_003"
    }


def test_checkpoint_records_completed_mutants(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)
    harness.executor = FakeExecutor([FakeResult({"t::a": "PASSED"})])

    harness.harvest(_fixed_mutants(repo, 3), baseline)
    payload = json.loads(harness.checkpoint_path.read_text())

    assert set(payload["completed"]) == {"mut_000", "mut_001", "mut_002"}


def test_resume_skips_already_harvested_mutants(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)
    mutants = _fixed_mutants(repo, 5)

    harness.executor = FakeExecutor([FakeResult({"t::a": "PASSED"})])
    harness.harvest(mutants[:2], baseline)
    calls_after_first = harness.executor.calls

    harness.harvest(mutants, baseline, resume=True)

    # Only the three outstanding mutants should have triggered a run.
    assert harness.executor.calls == calls_after_first + 3
    lines = harness.mutants_path.read_text().strip().split("\n")
    assert len(lines) == 5


def test_resume_disabled_reharvests_everything(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)
    mutants = _fixed_mutants(repo, 2)

    harness.executor = FakeExecutor([FakeResult({"t::a": "PASSED"})])
    harness.harvest(mutants, baseline)
    harness.harvest(mutants, baseline, resume=False)

    assert harness.executor.calls == 4


def test_corrupt_checkpoint_does_not_abort_harvest(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    harness.checkpoint_path.write_text("{not valid json")

    assert harness._load_checkpoint() == set()


def test_harvest_error_is_recorded_and_loop_continues(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    baseline = BaselineProfile(stable_passing=["t::a"], n_runs=3)

    calls = {"n": 0}

    def exploding_run(*args: Any, **kwargs: Any) -> FakeResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk gremlin")
        return FakeResult({"t::a": "PASSED"})

    harness.executor = type("E", (), {"run_tests": staticmethod(exploding_run)})()

    summary = harness.harvest(_fixed_mutants(repo, 2), baseline)

    lines = harness.mutants_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "error"
    assert summary["harvested"] == 1


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------

def test_discovery_finds_source_and_skips_tests(tmp_path):
    repo = _make_repo(tmp_path)
    found = discover_source_files(repo, ["pkg", "tests"])
    names = {p.name for p in found}

    assert "core.py" in names
    assert "test_core.py" not in names, "test files must never be mutated"


def test_discovery_skips_excluded_filenames(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "pkg" / "conftest.py").write_text("x = 1\n")
    (repo / "pkg" / "_version.py").write_text("__version__ = '1.0'\n")

    names = {p.name for p in discover_source_files(repo, ["pkg"])}

    assert "conftest.py" not in names
    assert "_version.py" not in names


def test_discovery_skips_cache_and_venv_directories(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "stale.py").write_text("x = 1\n")
    (repo / "pkg" / ".venv").mkdir()
    (repo / "pkg" / ".venv" / "lib.py").write_text("y = 2\n")

    names = {p.name for p in discover_source_files(repo, ["pkg"])}

    assert "stale.py" not in names
    assert "lib.py" not in names


def test_discovery_tolerates_missing_directory(tmp_path):
    repo = _make_repo(tmp_path)
    assert discover_source_files(repo, ["does_not_exist"]) == []


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def test_sampling_is_balanced_across_operator_families(tmp_path):
    """
    Unstratified sampling is swamped by constant mutations, which rarely kill
    a test. Families must be drawn round-robin.
    """
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)

    candidates = (
        [Mutant(f"c{i}", "pkg/core.py", 1, "const_str_empty", "a", "b", "x") for i in range(100)]
        + [Mutant(f"a{i}", "pkg/core.py", 1, "arith_+_to_-", "a", "b", "x") for i in range(10)]
        + [Mutant(f"r{i}", "pkg/core.py", 1, "return_to_None", "a", "b", "x") for i in range(10)]
    )

    selected = harness.sample_mutants(candidates, 30)
    families: Dict[str, int] = {}
    for mutant in selected:
        family = mutant.operator.split("_")[0]
        families[family] = families.get(family, 0) + 1

    assert len(selected) == 30
    # Constants are 83% of candidates but must not dominate the sample.
    assert families.get("const", 0) <= 15
    assert families.get("arith", 0) >= 5
    assert families.get("return", 0) >= 5


def test_sampling_is_reproducible_under_seed(tmp_path):
    repo = _make_repo(tmp_path)
    candidates = [
        Mutant(f"m{i}", "pkg/core.py", 1, "arith_+_to_-", "a", "b", "x")
        for i in range(50)
    ]

    first = _make_harness(repo, tmp_path, seed=7).sample_mutants(candidates, 10)
    second = _make_harness(repo, tmp_path, seed=7).sample_mutants(candidates, 10)

    assert [m.mutant_id for m in first] == [m.mutant_id for m in second]


def test_sampling_returns_all_when_request_exceeds_supply(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)
    candidates = [
        Mutant(f"m{i}", "pkg/core.py", 1, "arith_+_to_-", "a", "b", "x")
        for i in range(5)
    ]

    assert len(harness.sample_mutants(candidates, 100)) == 5


# --------------------------------------------------------------------------
# Profile serialization
# --------------------------------------------------------------------------

def test_baseline_profile_round_trips():
    profile = BaselineProfile(
        stable_passing=["t::a", "t::b"],
        excluded={"t::c": "skipped"},
        mean_durations={"t::a": 0.5, "t::b": 1.5},
        suite_duration=12.0,
        n_runs=3,
    )

    restored = BaselineProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert restored.stable_passing == profile.stable_passing
    assert restored.excluded == profile.excluded
    assert restored.n_runs == 3
    assert restored.universe_size == 2


# --------------------------------------------------------------------------
# Import roots
# --------------------------------------------------------------------------

def _make_package(root: Path, package: str, container: str = "") -> Path:
    """Create <root>/[container/]<package>/ with two mutable modules."""
    base = root / container if container else root
    (base / package).mkdir(parents=True)
    (base / package / "__init__.py").write_text("VALUE = 1")
    (base / package / "core.py").write_text(
        """
def add(a, b):
    return a + b
"""
    )
    return root


def test_import_roots_for_a_root_layout_package(tmp_path):
    repo = _make_repo(tmp_path)
    assert import_roots(repo, ["pkg"]) == ["pkg"]


def test_import_roots_strips_the_container_directory(tmp_path):
    """
    A src-layout checkout is importable as its inner package, never as `src`.

    This is the validators case. That project ships a tracked, docstring-only
    ``src/__init__.py``, so `src` looks like a package on disk while the
    installed distribution exposes only ``validators``. Deriving the root from
    the directory name demanded that ``import src`` succeed, which it never
    does, and the provenance guard rejected a perfectly good environment.
    """
    repo = _make_package(tmp_path / "repo", "widget", container="src")
    (repo / "src" / "__init__.py").write_text('"""Docstring only, as upstream ships it."""')

    assert import_roots(repo, ["src"]) == ["widget"]


def test_import_roots_for_a_single_module_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "parse.py").write_text("VALUE = 1")

    assert import_roots(repo, ["."]) == ["parse"]


def test_import_roots_are_deduplicated(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "pkg" / "extra.py").write_text("VALUE = 2")

    assert import_roots(repo, ["pkg"]) == ["pkg"]


def test_import_roots_ignores_a_bare_package_marker(tmp_path):
    """An __init__.py alone names no importable module of its own."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "__init__.py").write_text("")

    assert import_roots(repo, ["src"]) == []


# --------------------------------------------------------------------------
# Import provenance -- the guard against labelling an uninstalled checkout
# --------------------------------------------------------------------------

def test_provenance_accepts_an_interpreter_that_imports_the_checkout(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(repo))

    record = _make_harness(repo, tmp_path).verify_import_provenance()

    assert record["verified"] is True
    assert str(repo) in record["modules"]["pkg"]


def test_provenance_rejects_an_interpreter_that_cannot_import_the_package(tmp_path, monkeypatch):
    repo = _make_package(tmp_path / "repo", "conftest_probe_absent")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    harness = MutationHarness(
        repo_root=str(repo),
        repo_name="absent",
        source_dirs=["conftest_probe_absent"],
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(RuntimeError, match="not importable"):
        harness.verify_import_provenance()


def test_provenance_rejects_a_package_resolved_outside_the_checkout(tmp_path, monkeypatch):
    """
    The failure this guard exists for, and the one that nearly happened.

    `sqlparse` is installed in this project's own ambient environment, so
    harvesting it under the default interpreter would have imported the
    released package while mutating the clone: every mutant harmless, every
    label 0, a clean-looking dataset that measures nothing. No exception is
    raised by that arrangement on its own -- which is exactly why it has to be
    checked before compute is spent.
    """
    name = "conftest_probe_shadowed"
    repo = _make_package(tmp_path / "repo", name)
    _make_package(tmp_path / "site", name)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "site"))

    harness = MutationHarness(
        repo_root=str(repo),
        repo_name="shadowed",
        source_dirs=[name],
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(RuntimeError, match="OUTSIDE the checkout"):
        harness.verify_import_provenance()


def test_provenance_reports_unverified_when_no_root_can_be_derived(tmp_path):
    """
    An unverifiable configuration must say so rather than claim a pass.

    Nothing is mutable in this case either, so the harvest fails on its own
    later; the record exists so the manifest never carries a silent True.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    harness = MutationHarness(
        repo_root=str(repo),
        repo_name="empty",
        source_dirs=["does_not_exist"],
        output_dir=str(tmp_path / "out"),
    )
    record = harness.verify_import_provenance()

    assert record["verified"] is False
    assert record["reason"] == "no_import_root_derived"


def test_harness_passes_its_interpreter_to_the_executor(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path, python_executable=sys.executable)

    assert harness.python_executable == sys.executable
    assert harness.executor.python_executable == str(Path(sys.executable).resolve())


# --------------------------------------------------------------------------
# The run summary is the only auditable record of how labels were produced
# --------------------------------------------------------------------------

def test_write_summary_lands_beside_the_labels(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)

    written = harness.write_summary({'repo': 'fake_repo', 'harvested': 3})

    assert written == harness.summary_path
    assert written.parent == harness.output_dir
    assert written.name == SUMMARY_FILENAME
    assert json.loads(written.read_text()) == {'repo': 'fake_repo', 'harvested': 3}


def test_write_summary_serialises_paths_rather_than_failing(tmp_path):
    # Summaries carry interpreter and mutant paths; a TypeError here would lose
    # the whole record of a harvest that has already been paid for.
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)

    written = harness.write_summary({'interpreter': Path(sys.executable)})

    assert json.loads(written.read_text())['interpreter'] == sys.executable


def test_write_summary_overwrites_so_a_resumed_run_is_not_ambiguous(tmp_path):
    repo = _make_repo(tmp_path)
    harness = _make_harness(repo, tmp_path)

    harness.write_summary({'harvested': 1})
    harness.write_summary({'harvested': 2})

    assert json.loads(harness.summary_path.read_text()) == {'harvested': 2}
