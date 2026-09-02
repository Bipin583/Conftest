"""
Unit tests for the real-dataset builder (Component 2).

This script is the seam where measured mutation outcomes become model inputs.
The first iteration of this project fabricated its labels, so the tests here
concentrate on the two ways a rebuilt dataset could still lie:

* a test ID resolving to the wrong file, silently mislabelling a whole column;
* a history feature seeing an outcome from its own or a later mutant, which
  leaks the label into the features and inflates every downstream metric.

Anything absent must raise rather than default, so the "missing input" tests
are as load-bearing as the arithmetic ones.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_real_dataset import (  # noqa: E402
    G1_MAX_FAILURE_RATE,
    G1_MIN_FAILURE_RATE,
    MESSAGE_DERIVED_FEATURES,
    ORDER_EPOCH,
    RECENT_WINDOW,
    USABLE_STATUS,
    Harvest,
    HistoryAccumulator,
    ResolvedTest,
    accepted_repos,
    build_dataset,
    build_rows,
    load_harvest,
    resolve_test_id,
)
from conftest.features.pipeline import FEATURE_NAMES  # noqa: E402


# --------------------------------------------------------------------------
# Test ID resolution
# --------------------------------------------------------------------------

def _write(path: Path, text: str = "def test_a(): pass\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolves_a_plain_module_level_test(tmp_path):
    _write(tmp_path / "tests" / "test_cli.py")
    got = resolve_test_id("tests/test_cli::test_main_help", tmp_path)
    assert got == ResolvedTest(
        test_id="tests/test_cli::test_main_help",
        test_path="tests/test_cli.py",
        test_function="test_main_help",
        test_class="",
    )


def test_preserves_the_class_so_same_named_tests_do_not_collide(tmp_path):
    """
    The JUnit writer emits no `file` attribute, so IDs are built from
    `classname` and the class survives. That is what keeps LRUCacheTest and
    LFUCacheTest from merging into one label.
    """
    _write(tmp_path / "tests" / "test_cache.py")
    a = resolve_test_id("tests/test_cache/LRUCacheTest::test_getsizeof", tmp_path)
    b = resolve_test_id("tests/test_cache/LFUCacheTest::test_getsizeof", tmp_path)
    assert a.test_class == "LRUCacheTest"
    assert b.test_class == "LFUCacheTest"
    assert a.test_path == b.test_path == "tests/test_cache.py"
    assert a != b


def test_strips_the_checkout_path_prefix(tmp_path):
    """
    classname is relative to the pytest rootdir, which sits above the checkout,
    so a real ID arrives as `data/repos/cachetools/tests/test_cache/...`.
    """
    _write(tmp_path / "tests" / "test_cache.py")
    got = resolve_test_id(
        "data/repos/cachetools/tests/test_cache/CacheTest::test_clear", tmp_path
    )
    assert got.test_path == "tests/test_cache.py"
    assert got.test_class == "CacheTest"


def test_parameter_repr_containing_a_double_colon_still_resolves(tmp_path):
    """
    Measured on sqlparse: 45 of 509 IDs carry SQL casts in the parameter repr,
    e.g. `test_grouping::test_group_identifier_list[sum(a)::integer, b]`.
    Splitting on the LAST separator landed inside the brackets and lost them.
    """
    _write(tmp_path / "tests" / "test_grouping.py")
    got = resolve_test_id(
        "tests/test_grouping::test_group_identifier_list[sum(a)::integer, b]", tmp_path
    )
    assert got is not None
    assert got.test_path == "tests/test_grouping.py"
    assert got.test_class == ""
    assert got.test_function == "test_group_identifier_list[sum(a)::integer, b]"


def test_nested_test_package_resolves_to_the_deepest_real_module(tmp_path):
    """A shallower same-named module must not shadow the real one."""
    _write(tmp_path / "tests" / "unit" / "test_core.py")
    got = resolve_test_id("proj/tests/unit/test_core::test_x", tmp_path)
    assert got.test_path == "tests/unit/test_core.py"


def test_unresolvable_id_returns_none_rather_than_guessing(tmp_path):
    """
    Returning a plausible path here is exactly the failure mode that put the
    literal string "src/module.py" into the first dataset.
    """
    assert resolve_test_id("tests/test_ghost::test_x", tmp_path) is None


def test_id_without_a_separator_returns_none(tmp_path):
    _write(tmp_path / "tests" / "test_cli.py")
    assert resolve_test_id("tests/test_cli", tmp_path) is None


def test_id_without_a_function_returns_none(tmp_path):
    _write(tmp_path / "tests" / "test_cli.py")
    assert resolve_test_id("tests/test_cli::", tmp_path) is None


# --------------------------------------------------------------------------
# Causal history
# --------------------------------------------------------------------------

def test_history_starts_empty_rather_than_smoothed():
    hist = HistoryAccumulator({"t1": 0.5})
    feats = hist.features_for("t1", "src/a.py")
    assert feats["hist_total_prior_runs"] == 0.0
    assert feats["hist_prior_failures"] == 0.0
    assert feats["hist_lifetime_failure_rate"] == 0.0
    assert feats["hist_recent_10_failure_rate"] == 0.0
    assert feats["hist_has_ever_failed"] == 0.0
    # The duration is a real baseline measurement and is available immediately.
    assert feats["hist_avg_duration"] == 0.5


def test_a_recorded_failure_is_invisible_until_the_next_read():
    """The causality guarantee: features are read strictly before recording."""
    hist = HistoryAccumulator({})
    before = hist.features_for("t1", "src/a.py")
    hist.record(killed=["t1"], universe=["t1", "t2"], changed_file="src/a.py")
    after = hist.features_for("t1", "src/a.py")

    assert before["hist_prior_failures"] == 0.0
    assert after["hist_prior_failures"] == 1.0
    assert after["hist_total_prior_runs"] == 1.0
    assert after["hist_lifetime_failure_rate"] == 1.0
    assert after["hist_has_ever_failed"] == 1.0


def test_a_passing_test_accumulates_runs_but_not_failures():
    hist = HistoryAccumulator({})
    hist.record(killed=["t1"], universe=["t1", "t2"], changed_file="src/a.py")
    feats = hist.features_for("t2", "src/a.py")
    assert feats["hist_total_prior_runs"] == 1.0
    assert feats["hist_prior_failures"] == 0.0
    assert feats["hist_has_ever_failed"] == 0.0


def test_recent_window_forgets_beyond_its_length():
    hist = HistoryAccumulator({})
    hist.record(killed=["t1"], universe=["t1"], changed_file="src/a.py")
    for _ in range(RECENT_WINDOW):
        hist.record(killed=[], universe=["t1"], changed_file="src/a.py")

    feats = hist.features_for("t1", "src/a.py")
    # The single failure has aged out of the window but not out of the lifetime.
    assert feats["hist_recent_10_failure_rate"] == 0.0
    assert feats["hist_prior_failures"] == 1.0
    assert feats["hist_lifetime_failure_rate"] == pytest.approx(1 / (RECENT_WINDOW + 1))


def test_changed_file_modification_count_is_per_file():
    hist = HistoryAccumulator({})
    hist.record(killed=[], universe=["t1"], changed_file="src/a.py")
    hist.record(killed=[], universe=["t1"], changed_file="src/a.py")
    hist.record(killed=[], universe=["t1"], changed_file="src/b.py")

    assert hist.features_for("t1", "src/a.py")["hist_changed_files_prior_mod_count"] == 2.0
    assert hist.features_for("t1", "src/b.py")["hist_changed_files_prior_mod_count"] == 1.0
    assert hist.features_for("t1", "src/c.py")["hist_changed_files_prior_mod_count"] == 0.0


def test_flaky_score_is_zero_by_construction_not_by_default():
    """
    The baseline universe admits only tests that passed identically on every
    screening run, so there is no flakiness left to measure. Zero here is a
    fact about the universe; the manifest reports it as inert.
    """
    hist = HistoryAccumulator({"t1": 1.0})
    hist.record(killed=["t1"], universe=["t1"], changed_file="src/a.py")
    assert hist.features_for("t1", "src/a.py")["hist_flaky_score"] == 0.0


def test_history_emits_exactly_the_declared_hist_features():
    hist = HistoryAccumulator({})
    declared = {n for n in FEATURE_NAMES if n.startswith("hist_")}
    assert set(hist.features_for("t1", "src/a.py")) == declared


# --------------------------------------------------------------------------
# Harvest loading: absent input must raise, never default
# --------------------------------------------------------------------------

def _harvest_dir(tmp_path: Path, baseline: dict, mutant_lines: list) -> Path:
    hdir = tmp_path / "harvest" / "demo"
    hdir.mkdir(parents=True)
    (hdir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (hdir / "mutants.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in mutant_lines), encoding="utf-8"
    )
    return hdir


def _baseline(tests=("tests/test_a::test_x",)) -> dict:
    return {
        "n_runs": 3,
        "stable_passing": list(tests),
        "excluded": {},
        "mean_durations": {t: 0.01 for t in tests},
        "suite_duration": 1.0,
    }


def _mutant(mutant_id: str, killed=(), status=USABLE_STATUS, **over) -> dict:
    rec = {
        "mutant_id": mutant_id,
        "repo": "demo",
        "file_path": "demo/core.py",
        "line": 7,
        "operator": "comparison_swap",
        "status": status,
        "killed": list(killed),
        "n_killed": len(killed),
    }
    rec.update(over)
    return rec


def test_missing_baseline_raises(tmp_path):
    hdir = tmp_path / "harvest" / "demo"
    hdir.mkdir(parents=True)
    (hdir / "mutants.jsonl").write_text(json.dumps(_mutant("m1")) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no baseline"):
        load_harvest("demo", tmp_path, hdir, "abc123")


def test_missing_mutants_raises(tmp_path):
    hdir = tmp_path / "harvest" / "demo"
    hdir.mkdir(parents=True)
    (hdir / "baseline.json").write_text(json.dumps(_baseline()), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no mutants"):
        load_harvest("demo", tmp_path, hdir, "abc123")


def test_empty_universe_raises(tmp_path):
    hdir = _harvest_dir(tmp_path, _baseline(tests=()), [_mutant("m1")])
    with pytest.raises(ValueError, match="universe is empty"):
        load_harvest("demo", tmp_path, hdir, "abc123")


def test_corrupt_harvest_line_raises_with_its_line_number(tmp_path):
    hdir = _harvest_dir(tmp_path, _baseline(), [_mutant("m1")])
    with (hdir / "mutants.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        load_harvest("demo", tmp_path, hdir, "abc123")


def test_unusable_statuses_are_excluded_and_counted(tmp_path):
    hdir = _harvest_dir(tmp_path, _baseline(), [
        _mutant("m1"),
        _mutant("m2", status="broke_suite"),
        _mutant("m3", status="timed_out"),
        _mutant("m4", status="broke_suite"),
    ])
    harvest = load_harvest("demo", tmp_path, hdir, "abc123")
    assert [m["mutant_id"] for m in harvest.mutants] == ["m1"]
    assert harvest.excluded_status == {"broke_suite": 2, "timed_out": 1}


def test_a_harvest_of_only_unusable_mutants_raises(tmp_path):
    hdir = _harvest_dir(tmp_path, _baseline(), [_mutant("m1", status="broke_suite")])
    with pytest.raises(ValueError, match="no mutants with status"):
        load_harvest("demo", tmp_path, hdir, "abc123")


def test_mutants_are_ordered_deterministically(tmp_path):
    hdir = _harvest_dir(tmp_path, _baseline(), [
        _mutant("m_c"), _mutant("m_a"), _mutant("m_b"),
    ])
    harvest = load_harvest("demo", tmp_path, hdir, "abc123")
    assert [m["mutant_id"] for m in harvest.mutants] == ["m_a", "m_b", "m_c"]


# --------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------

def _tiny_repo(tmp_path: Path) -> Path:
    """A minimal but real repository the extractors can actually parse."""
    root = tmp_path / "demo"
    _write(root / "demo" / "__init__.py", "")
    _write(root / "demo" / "core.py", "def add(a, b):\n    return a + b\n")
    _write(
        root / "tests" / "test_core.py",
        "from demo.core import add\n\n\n"
        "def test_x():\n    assert add(1, 2) == 3\n\n\n"
        "def test_y():\n    assert add(0, 0) == 0\n",
    )
    return root


def _rows_for(tmp_path, mutants):
    root = _tiny_repo(tmp_path)
    universe = ["tests/test_core::test_x", "tests/test_core::test_y"]
    resolved = {t: resolve_test_id(t, root) for t in universe}
    assert all(resolved.values())
    harvest = Harvest(
        repo="demo",
        repo_root=root,
        commit_sha="deadbeef",
        universe=universe,
        mean_durations={t: 0.02 for t in universe},
        mutants=mutants,
    )
    return list(build_rows(harvest, resolved))


def test_every_universe_test_gets_a_row_per_mutant(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1"), _mutant("m2")])
    assert len(rows) == 4


def test_label_comes_from_the_recorded_kill_list(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1", killed=["tests/test_core::test_x"])])
    labels = {r["test_id"]: r["label_failed"] for r in rows}
    assert labels == {"tests/test_core::test_x": 1, "tests/test_core::test_y": 0}


def test_a_row_never_sees_its_own_label(tmp_path):
    """
    The leak that contaminated historical_failure_rate in iteration one. test_x
    is killed by both mutants; its own outcome must be absent from its own row.
    """
    killed = ["tests/test_core::test_x"]
    rows = _rows_for(tmp_path, [_mutant("m1", killed=killed), _mutant("m2", killed=killed)])
    by_mutant = {}
    for row in rows:
        if row["test_id"] == "tests/test_core::test_x":
            by_mutant[row["commit_sha"]] = row

    assert by_mutant["m1"]["hist_prior_failures"] == 0.0
    assert by_mutant["m1"]["hist_total_prior_runs"] == 0.0
    assert by_mutant["m2"]["hist_prior_failures"] == 1.0
    assert by_mutant["m2"]["hist_total_prior_runs"] == 1.0


def test_ordering_surrogate_is_monotonic_and_paired_with_an_explicit_index(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1"), _mutant("m2"), _mutant("m3")])
    seen = {r["commit_sha"]: (r["mutant_index"], r["commit_timestamp"]) for r in rows}
    assert seen["m1"][0] == 0 and seen["m2"][0] == 1 and seen["m3"][0] == 2
    assert seen["m1"][1] == ORDER_EPOCH
    assert seen["m3"][1] == ORDER_EPOCH + timedelta(seconds=2)
    assert seen["m1"][1] < seen["m2"][1] < seen["m3"][1]


def test_rows_carry_full_provenance(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1", operator="comparison_swap", line=2)])
    row = rows[0]
    assert row["repo"] == "demo"
    assert row["repo_commit_sha"] == "deadbeef"
    assert row["mutant_operator"] == "comparison_swap"
    assert row["changed_file_path"] == "demo/core.py"
    assert row["changed_line"] == 2
    assert row["test_path"] == "tests/test_core.py"
    assert row["test_function"] in {"test_x", "test_y"}


def test_rows_contain_every_declared_feature(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1")])
    missing = [name for name in FEATURE_NAMES if name not in rows[0]]
    assert missing == []


def test_message_derived_features_are_constant_not_invented(tmp_path):
    rows = _rows_for(tmp_path, [_mutant("m1"), _mutant("m2")])
    for name in MESSAGE_DERIVED_FEATURES:
        assert len({r[name] for r in rows}) == 1, f"{name} varies without a message"


def test_a_mutant_killing_nothing_still_produces_rows(tmp_path):
    """Zero-kill mutants are the majority and carry the negative class."""
    rows = _rows_for(tmp_path, [_mutant("m1", killed=[])])
    assert len(rows) == 2
    assert all(r["label_failed"] == 0 for r in rows)


def test_unresolved_tests_are_dropped_not_defaulted(tmp_path):
    root = _tiny_repo(tmp_path)
    universe = ["tests/test_core::test_x", "tests/test_ghost::test_z"]
    resolved = {"tests/test_core::test_x": resolve_test_id(universe[0], root)}
    harvest = Harvest(
        repo="demo", repo_root=root, commit_sha="deadbeef",
        universe=universe, mean_durations={}, mutants=[_mutant("m1")],
    )
    rows = list(build_rows(harvest, resolved))
    assert [r["test_id"] for r in rows] == ["tests/test_core::test_x"]


# --------------------------------------------------------------------------
# Screening report
# --------------------------------------------------------------------------

def _report(tmp_path: Path, candidates: dict) -> Path:
    path = tmp_path / "screening_report.json"
    path.write_text(
        json.dumps({"screened_at": "2026-09-02T00:00:00", "candidates": candidates}),
        encoding="utf-8",
    )
    return path


def test_only_stage3_survivors_are_eligible(tmp_path):
    """
    Stage 3 is the harvest gate, not stage 2. A repo can install cleanly and run
    a fast green suite (stage 2) and still be useless: parse and inflection are
    single-module libraries whose dependency features never vary, so every row
    carries the same relationship and there is nothing to select on.
    """
    path = _report(tmp_path, {
        "good": {"name": "good", "stage1_passed": True, "stage2_passed": True,
                 "stage3_passed": True},
        "flat": {"name": "flat", "stage1_passed": True, "stage2_passed": True,
                 "stage3_passed": False},
        "slow": {"name": "slow", "stage1_passed": True, "stage2_passed": False,
                 "stage3_passed": False},
        "nope": {"name": "nope", "stage1_passed": False, "stage2_passed": False,
                 "stage3_passed": False},
    })
    assert [r["name"] for r in accepted_repos(path)] == ["good"]


def test_a_stage2_pass_alone_is_not_enough(tmp_path):
    """Guards the gate against a silent regression to the weaker predicate."""
    path = _report(tmp_path, {"flat": {"name": "flat", "stage2_passed": True}})
    with pytest.raises(ValueError, match="stage 3"):
        accepted_repos(path)


def test_missing_report_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No screening report"):
        accepted_repos(tmp_path / "absent.json")


def test_report_with_no_survivors_raises(tmp_path):
    path = _report(tmp_path, {"slow": {"name": "slow", "stage2_passed": False}})
    with pytest.raises(ValueError, match="No repository passed stage 3"):
        accepted_repos(path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_build_dataset_produces_a_labelled_frame_and_an_honest_manifest(tmp_path):
    workspace = tmp_path / "repos"
    workspace.mkdir()
    root = _tiny_repo(workspace)
    assert root.name == "demo"

    universe = ["tests/test_core::test_x", "tests/test_core::test_y"]
    harvest_root = tmp_path / "harvest"
    hdir = harvest_root / "demo"
    hdir.mkdir(parents=True)
    (hdir / "baseline.json").write_text(json.dumps(_baseline(universe)), encoding="utf-8")
    (hdir / "mutants.jsonl").write_text(
        json.dumps(_mutant("m1", killed=[universe[0]])) + "\n"
        + json.dumps(_mutant("m2")) + "\n"
        + json.dumps(_mutant("m3", status="broke_suite")) + "\n",
        encoding="utf-8",
    )
    report = _report(workspace, {
        "demo": {"name": "demo", "stage2_passed": True, "stage3_passed": True, "commit_sha": "deadbeef"}
    })

    dataset, manifest = build_dataset(["demo"], workspace, harvest_root, report)

    assert len(dataset) == 4                      # 2 usable mutants x 2 tests
    assert int(dataset["label_failed"].sum()) == 1
    assert manifest["n_rows"] == 4
    assert manifest["n_failures"] == 1
    assert manifest["failure_rate"] == 0.25
    assert manifest["g1_failure_rate_in_band"] is False   # 25% > band ceiling
    assert manifest["g1_failure_rate_band"] == [G1_MIN_FAILURE_RATE, G1_MAX_FAILURE_RATE]
    assert "measured pytest outcomes" in manifest["label_source"]

    repo_entry = manifest["repos"][0]
    assert repo_entry["n_mutants_used"] == 2
    assert repo_entry["n_mutants_excluded_by_status"] == {"broke_suite": 1}
    assert repo_entry["n_tests_resolved"] == 2
    assert repo_entry["n_tests_unresolved"] == 0

    # The inert list must name the message features rather than hide them.
    assert set(MESSAGE_DERIVED_FEATURES).issubset(set(manifest["inert_features"]))


def test_requesting_an_unscreened_repo_raises(tmp_path):
    workspace = tmp_path / "repos"
    workspace.mkdir()
    report = _report(workspace, {"demo": {"name": "demo", "stage2_passed": True, "stage3_passed": True}})
    with pytest.raises(ValueError, match="Not accepted by screening"):
        build_dataset(["other"], workspace, tmp_path / "harvest", report)


def test_a_missing_checkout_raises_and_names_the_recovery(tmp_path):
    workspace = tmp_path / "repos"
    workspace.mkdir()
    report = _report(workspace, {"demo": {"name": "demo", "stage2_passed": True, "stage3_passed": True}})
    with pytest.raises(FileNotFoundError, match="Re-run scripts/screen_repos.py"):
        build_dataset(["demo"], workspace, tmp_path / "harvest", report)


# --------------------------------------------------------------------------
# Guard: the constants that encode the methodology
# --------------------------------------------------------------------------

def test_only_fully_harvested_mutants_are_usable():
    assert USABLE_STATUS == "harvested"


def test_g1_band_is_a_realistic_failure_rate():
    assert 0.0 < G1_MIN_FAILURE_RATE < G1_MAX_FAILURE_RATE < 0.5
