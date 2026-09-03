#!/usr/bin/env python
"""
Component 2 — build the real feature dataset from harvested mutation labels.

Turns the harness output (`baseline.json` + `mutants.jsonl` per repo) into the
tabular `(change, test) -> label` dataset the model trains on.

Every label in the output is the recorded outcome of an actual pytest run.
Nothing here draws a random number, and nothing here substitutes a plausible
value for a missing measurement: if an input is absent, this script raises.

    python scripts/build_real_dataset.py --all
    python scripts/build_real_dataset.py --repos cachetools parse

Three honest limitations are recorded in the manifest rather than papered over:

1. A mutant has no commit message, so the four message-derived diff features
   are constant. They are listed in the manifest as inert.
2. A mutant has no wall-clock time. `commit_timestamp` carries a monotonic
   ordering surrogate derived from harvest order, which is what the temporal
   splitter actually needs; `mutant_index` is emitted alongside it so the
   ordering is explicit rather than reverse-engineered from a fake date.
3. The baseline universe contains only tests that passed identically on every
   screening run, so `hist_flaky_score` is legitimately zero throughout.

History features are accumulated causally: the row for mutant *i* sees only
outcomes from mutants *0..i-1*. That ordering is the same one the timestamp
surrogate encodes, so a temporal split cannot leak a future outcome into a
past feature. This is the discipline whose absence let fabricated labels
contaminate `historical_failure_rate` in the first iteration of the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conftest.logging_config import configure_logging, get_logger  # noqa: E402

from conftest.features.ast_features import extract_ast_metrics_from_file  # noqa: E402
from conftest.features.dependency_graph import DependencyGraphBuilder  # noqa: E402
from conftest.features.diff_features import extract_diff_features  # noqa: E402
from conftest.features.pipeline import FEATURE_NAMES  # noqa: E402
from conftest.groundtruth.mutation_harness import SUMMARY_FILENAME  # noqa: E402

logger = get_logger(__name__)

# A mutation rewrites exactly one line: one line removed, one added.
MUTATION_LINES_ADDED = 1
MUTATION_LINES_DELETED = 1

# Only these mutants become training rows. `broke_suite` mutants kill nearly the
# whole suite and describe a compilation-level break rather than a localized
# fault; `timed_out` and `timeout_broken` mutants have no trustworthy outcome at
# all. All of them are kept in the harvest file and counted in the manifest,
# never silently dropped.
USABLE_STATUS = "harvested"

# Two ways a record can claim `harvested` and still not be a label:
#
#   * `checkout_drifted` -- the suite edited its own tracked files during that
#     run, so the kills it reports may be inherited damage rather than effects
#     of the mutation. Measured once on sqlparse: mut_42bfc7db58b0 left
#     tests/files/function.sql truncated.
#   * `timeout_enforced: false` -- the suite outlived the timeout meant to bound
#     it, so its outcomes were read while a tree was still writing to the
#     checkout.
#
# The harness records both and repairs what it can. Refusing them here is what
# keeps them out of the training data.
DRIFT_FIELD = "checkout_drifted"
ENFORCED_FIELD = "timeout_enforced"

# One contaminated record is an incident and is excluded. A repo where they are
# common is a broken harvest, and quietly training on the other 98% while
# publishing a headline would be exactly the fabrication this pipeline exists to
# prevent -- so past this fraction the build stops and asks for a re-harvest.
MAX_CONTAMINATED_FRACTION = 0.02

# Recent-window size for hist_recent_10_failure_rate.
RECENT_WINDOW = 10

# Ordering surrogate. Mutants have no wall-clock time; the temporal splitter
# needs only a monotonic key, so one synthetic second per mutant is enough.
# Recorded in the manifest as a surrogate so no reader mistakes it for a date.
ORDER_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# Message-derived features. A mutant has no commit message, so these four are
# constant across every row. Reported, not hidden.
MESSAGE_DERIVED_FEATURES = [
    "diff_msg_length",
    "diff_msg_word_count",
    "diff_is_fix_commit",
    "diff_is_refactor_commit",
]

# Gate G1 from the master plan: a realistic failure rate, not a degenerate one.
# The plan states the band *per repository*, and the pooled rate can sit inside
# it while a repo does not -- a large low-density repo pulls the pool towards
# itself. Both are reported, and the pooled figure never stands in for the
# per-repo one.
G1_MIN_FAILURE_RATE = 0.01
G1_MAX_FAILURE_RATE = 0.15
G1_MIN_PAIRS_PER_REPO = 5_000


# ---------------------------------------------------------------------------
# Test ID resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTest:
    """A JUnit test ID mapped back onto a real file on disk."""

    test_id: str
    test_path: str          # repo-relative, e.g. "tests/test_cache.py"
    test_function: str      # e.g. "test_clear"
    test_class: str = ""    # e.g. "CacheTest", empty for module-level tests


def resolve_test_id(test_id: str, repo_root: Path) -> Optional[ResolvedTest]:
    """
    Map a harness test ID back to (file, class, function).

    pytest's JUnit writer never emits a `file` attribute -- verified across
    1138 testcases in three repositories, 0 of which had one -- so the harness
    builds IDs from `classname`, which is the rootdir-relative module path with
    dots for separators. Two consequences:

    * The class name is preserved, so `TestA::test_x` and `TestB::test_x` in
      one module do not collide. Had `file` been present, both would have
      become `module.py::test_x` and silently merged into one label.
    * The path is prefixed by wherever the checkout sits under pytest's
      rootdir, e.g. `cachetools.tests.test_cache.CacheTest`.

    So the module cannot be read off the ID directly; it has to be resolved
    against the filesystem. We take the longest trailing segment run that names
    a real `.py` file, and treat whatever follows it as class nesting. Returns
    None when nothing resolves, so the caller can count and report the misses
    instead of guessing a path.
    """
    if "::" not in test_id:
        return None

    # Split on the FIRST separator, not the last. A parametrized ID carries the
    # parameter repr verbatim, and SQL parameters contain '::' themselves --
    # `test_grouping::test_group_identifier_list[sum(a)::integer, b]`. rpartition
    # split inside the brackets and lost 45 of sqlparse's 509 tests. The module
    # path is a filesystem path with dots for separators, so it can never
    # contain a colon; the first '::' is always the real boundary.
    module_part, _, function = test_id.partition("::")
    segments = [s for s in module_part.replace("\\", "/").split("/") if s]
    if not segments or not function:
        return None

    # Prefer the deepest module (largest k), then the longest surviving path.
    for k in range(len(segments), 0, -1):
        for start in range(0, k):
            candidate = "/".join(segments[start:k]) + ".py"
            if (repo_root / candidate).is_file():
                classes = segments[k:]
                return ResolvedTest(
                    test_id=test_id,
                    test_path=candidate,
                    test_function=function,
                    test_class=".".join(classes),
                )
    return None


# ---------------------------------------------------------------------------
# Harvest inputs
# ---------------------------------------------------------------------------

@dataclass
class Harvest:
    """One repository's harvest, validated."""

    repo: str
    repo_root: Path
    commit_sha: str
    universe: List[str]
    mean_durations: Dict[str, float]
    mutants: List[Dict[str, Any]]
    # Proved at load time by load_provenance, which raises rather than returning a
    # Harvest whose labels cannot be tied to the mutated checkout. The default
    # exists so unit tests can build a Harvest to exercise row construction alone.
    provenance: Dict[str, Any] = field(default_factory=dict)
    excluded_status: Dict[str, int] = field(default_factory=dict)
    # Harvested records refused by contamination_reason. Listed, not counted:
    # which mutants were dropped is the part a reader needs to check.
    excluded_contaminated: List[Dict[str, str]] = field(default_factory=list)


def load_provenance(repo: str, repo_root: Path, harvest_dir: Path) -> Dict[str, Any]:
    """
    Read the harvest's own proof that its labels came from the checkout.

    If the interpreter that ran a subject suite imported the *installed* copy of
    the package instead of the mutated checkout, every mutant survives and every
    label is 0. Nothing about the resulting file looks wrong: it has the right
    shape, the right test IDs and a plausible class balance. So the proof is
    required here, not assumed, and a harvest that cannot produce one is refused.
    """
    summary_path = harvest_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{repo}: no {SUMMARY_FILENAME} at {summary_path}, so there is no record of "
            f"which interpreter produced the labels. Re-harvest with "
            f"scripts/harvest_mutations.py --repos {repo}, which writes it."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("repo") not in (None, repo):
        raise ValueError(
            f"{repo}: {summary_path} records repo {summary.get('repo')!r}. "
            f"The summary belongs to another harvest."
        )

    provenance = summary.get("import_provenance") or {}
    if not provenance.get("verified"):
        raise ValueError(
            f"{repo}: import provenance was not verified for this harvest "
            f"(reason: {provenance.get('reason', 'unrecorded')}). The labels cannot be "
            f"shown to describe the mutated checkout, so the dataset would be fiction. "
            f"Re-harvest with scripts/harvest_mutations.py, which verifies before running."
        )

    modules = provenance.get("modules") or {}
    if not modules:
        raise ValueError(
            f"{repo}: provenance claims verified but names no module. Re-harvest."
        )

    root = str(repo_root).replace(chr(92), "/").rstrip("/").lower()
    outside = {
        name: path for name, path in modules.items()
        if not str(path).replace(chr(92), "/").lower().startswith(root + "/")
    }
    if outside:
        raise ValueError(
            f"{repo}: provenance resolves {sorted(outside)} outside the checkout at "
            f"{repo_root}: {outside}. The summary does not match this checkout."
        )

    return provenance


def contamination_reason(record: Dict[str, Any]) -> Optional[str]:
    """
    Why this harvested record is not a label, or None if it is one.

    Absence of either field means the harvest predates the check, not that the
    run was clean. That is a real limitation of the three harvests taken before
    the guards existed and it is recorded as such in the manifest; what this
    function can do is refuse the cases a harness did observe and report.
    """
    drifted = record.get(DRIFT_FIELD)
    if drifted:
        files = ", ".join(drifted) if isinstance(drifted, list) else str(drifted)
        return f"suite modified tracked files during the run: {files}"
    if record.get(ENFORCED_FIELD) is False:
        return "the suite outlived the timeout meant to bound it"
    return None


def load_harvest(repo: str, repo_root: Path, harvest_dir: Path, commit_sha: str) -> Harvest:
    """Load and validate one repository's harvest. Missing inputs raise."""
    baseline_path = harvest_dir / "baseline.json"
    mutants_path = harvest_dir / "mutants.jsonl"

    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"{repo}: no baseline at {baseline_path}. Run the mutation harness first."
        )
    if not mutants_path.is_file():
        raise FileNotFoundError(
            f"{repo}: no mutants at {mutants_path}. Run the mutation harness first."
        )

    # Cheap and decisive: refuse a harvest that cannot show its labels came from
    # the mutated checkout, before spending anything on parsing it.
    provenance = load_provenance(repo, repo_root, harvest_dir)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    universe = list(baseline.get("stable_passing", []))
    if not universe:
        raise ValueError(f"{repo}: baseline universe is empty; nothing to label against.")

    durations = {k: float(v) for k, v in baseline.get("mean_durations", {}).items()}

    usable: List[Dict[str, Any]] = []
    excluded: Dict[str, int] = {}
    contaminated: List[Tuple[str, str]] = []
    with mutants_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{repo}: corrupt harvest record at line {line_no}: {exc}") from exc
            status = record.get("status", "unknown")
            if status != USABLE_STATUS:
                excluded[status] = excluded.get(status, 0) + 1
                continue
            reason = contamination_reason(record)
            if reason:
                contaminated.append((str(record.get("mutant_id", f"line_{line_no}")), reason))
                continue
            usable.append(record)

    if contaminated:
        listed = ", ".join(f"{mid} ({why})" for mid, why in contaminated[:5])
        logger.error(
            f"{repo}: excluding {len(contaminated)} harvested record(s) measured "
            f"against a checkout that was not clean: {listed}"
            f"{' ...' if len(contaminated) > 5 else ''}"
        )
        share = len(contaminated) / (len(contaminated) + len(usable))
        if share > MAX_CONTAMINATED_FRACTION:
            raise ValueError(
                f"{repo}: {share:.1%} of harvested records ({len(contaminated)}) were "
                f"measured against a contaminated checkout, over the "
                f"{MAX_CONTAMINATED_FRACTION:.0%} ceiling. That is a broken harvest, "
                f"not isolated incidents; re-harvest with scripts/harvest_mutations.py "
                f"--repos {repo} --restore-checkout --reprofile-baseline rather than "
                f"training on what survived."
            )

    if not usable:
        raise ValueError(
            f"{repo}: no usable mutants with status={USABLE_STATUS!r}. "
            f"Statuses present: {excluded or 'none'}; "
            f"harvested but contaminated: {len(contaminated)}"
        )

    # Deterministic order. This is the order history accumulates in and the
    # order the timestamp surrogate encodes, so it must not depend on dict
    # iteration or filesystem order.
    usable.sort(key=lambda r: str(r.get("mutant_id", "")))

    return Harvest(
        repo=repo,
        repo_root=repo_root,
        commit_sha=commit_sha,
        universe=universe,
        mean_durations=durations,
        mutants=usable,
        provenance=provenance,
        excluded_status=excluded,
        excluded_contaminated=[{"mutant_id": mid, "reason": why} for mid, why in contaminated],
    )


# ---------------------------------------------------------------------------
# Causal history accumulator
# ---------------------------------------------------------------------------

class HistoryAccumulator:
    """
    Per-test outcome history, advanced one mutant at a time.

    Features for mutant *i* are read before mutant *i*'s own outcomes are
    recorded, so a row never sees its own label or any later one. The first
    mutant therefore has an all-zero history for every test, which is correct
    rather than something to smooth over.
    """

    def __init__(self, mean_durations: Dict[str, float]) -> None:
        self.mean_durations = mean_durations
        self.prior_runs = 0
        self.failures: Dict[str, int] = {}
        self.recent: Dict[str, deque] = {}
        self.file_mod_counts: Dict[str, int] = {}

    def features_for(self, test_id: str, changed_file: str) -> Dict[str, float]:
        """Read the history visible *before* the current mutant is recorded."""
        failures = self.failures.get(test_id, 0)
        window = self.recent.get(test_id)
        recent_rate = (sum(window) / len(window)) if window else 0.0
        lifetime_rate = (failures / self.prior_runs) if self.prior_runs else 0.0

        return {
            "hist_total_prior_runs": float(self.prior_runs),
            "hist_prior_failures": float(failures),
            "hist_lifetime_failure_rate": float(lifetime_rate),
            "hist_recent_10_failure_rate": float(recent_rate),
            # A real per-test measurement from the baseline profile.
            "hist_avg_duration": float(self.mean_durations.get(test_id, 0.0)),
            # The baseline universe admits only tests that passed identically on
            # every screening run, so this is zero by construction, not by
            # default. Listed as inert in the manifest.
            "hist_flaky_score": 0.0,
            "hist_has_ever_failed": 1.0 if failures else 0.0,
            "hist_changed_files_prior_mod_count": float(
                self.file_mod_counts.get(changed_file, 0)
            ),
        }

    def record(self, killed: Sequence[str], universe: Sequence[str], changed_file: str) -> None:
        """Fold one mutant's measured outcomes into the history."""
        killed_set = set(killed)
        for test_id in universe:
            failed = test_id in killed_set
            if failed:
                self.failures[test_id] = self.failures.get(test_id, 0) + 1
            window = self.recent.setdefault(test_id, deque(maxlen=RECENT_WINDOW))
            window.append(1 if failed else 0)
        self.prior_runs += 1
        self.file_mod_counts[changed_file] = self.file_mod_counts.get(changed_file, 0) + 1


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Memoises the two expensive extractors.

    Without this, a 250-mutant harvest over a 330-test universe would re-parse
    every test file and rebuild every dependency lookup 82,500 times.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.dep_builder = DependencyGraphBuilder(str(repo_root))
        self._ast: Dict[str, Dict[str, float]] = {}
        self._dep: Dict[Tuple[str, str], Dict[str, float]] = {}

    def ast_features(self, test_path: str, test_function: str) -> Dict[str, float]:
        cached = self._ast.get(test_path)
        if cached is None:
            info = extract_ast_metrics_from_file(str(self.repo_root / test_path))
            cached = {
                "ast_test_file_functions_count": float(info.get("functions_count", 0)),
                "ast_test_file_classes_count": float(info.get("classes_count", 0)),
                "ast_test_file_imports_count": float(info.get("imports_count", 0)),
                "ast_test_file_complexity": float(info.get("complexity", 0.0)),
            }
            self._ast[test_path] = cached
        # These two vary per test, not per file.
        return {
            **cached,
            "ast_test_is_parameterized": 1.0 if ("[" in test_function and "]" in test_function) else 0.0,
            "ast_test_func_name_length": float(len(test_function)),
        }

    def dep_features(self, test_path: str, changed_file: str) -> Dict[str, float]:
        key = (test_path, changed_file)
        cached = self._dep.get(key)
        if cached is None:
            cached = self.dep_builder.compute_dependency_features(test_path, [changed_file])
            self._dep[key] = cached
        return cached


def build_rows(harvest: Harvest, resolved: Dict[str, ResolvedTest]) -> Iterator[Dict[str, Any]]:
    """
    Yield one row per (mutant, test) pair, labelled from the measured outcome.

    Only killed tests are stored per mutant, so every other test in the
    baseline universe carries label 0 by construction.
    """
    cache = FeatureCache(harvest.repo_root)
    history = HistoryAccumulator(harvest.mean_durations)
    universe = [t for t in harvest.universe if t in resolved]

    for index, mutant in enumerate(harvest.mutants):
        changed_file = str(mutant["file_path"]).replace("\\", "/")
        killed = set(mutant.get("killed", []))

        changed_files = [{
            "file_path": changed_file,
            "lines_added": MUTATION_LINES_ADDED,
            "lines_deleted": MUTATION_LINES_DELETED,
            "change_type": "modified",
            "is_test": False,
        }]
        # A mutant has no commit message. An empty string keeps the four
        # message-derived features honestly constant; inventing prose would
        # manufacture signal the real change event never carried.
        diff_feats = extract_diff_features(changed_files, commit_message="")

        for test_id in universe:
            test = resolved[test_id]
            row: Dict[str, Any] = {
                # Identity and provenance.
                "commit_sha": mutant["mutant_id"],
                "commit_timestamp": ORDER_EPOCH + timedelta(seconds=index),
                "mutant_index": index,
                "repo": harvest.repo,
                "repo_commit_sha": harvest.commit_sha,
                "mutant_operator": mutant.get("operator", ""),
                "changed_file_path": changed_file,
                "changed_line": mutant.get("line", 0),
                "test_id": test_id,
                "test_path": test.test_path,
                "test_class": test.test_class,
                "test_function": test.test_function,
                # The label: measured, not modelled.
                "label_failed": 1 if test_id in killed else 0,
            }
            row.update(diff_feats)
            row.update(cache.ast_features(test.test_path, test.test_function))
            row.update(cache.dep_features(test.test_path, changed_file))
            row.update(history.features_for(test_id, changed_file))
            yield row

        # Advance history only after the whole mutant has been emitted.
        history.record(killed, universe, changed_file)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def accepted_repos(report_path: Path) -> List[Dict[str, Any]]:
    """Read the screener's verdict. Only stage-2 survivors are eligible."""
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No screening report at {report_path}. Run scripts/screen_repos.py --all first."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    # The screener keys candidates by repository name. Stage 3 is the last gate:
    # it rejects repositories where the dependency features do not vary, so test
    # selection would be vacuous. A stage-2 pass alone is not enough.
    candidates = payload.get("candidates", {})
    accepted = [
        entry for entry in candidates.values()
        if isinstance(entry, dict) and entry.get("stage3_passed")
    ]
    if not accepted:
        raise ValueError(
            f"No repository passed stage 3 in {report_path}. "
            f"Run scripts/screen_repos.py --stage3 to apply the structural gate."
        )
    return accepted


def build_dataset(
    repos: Sequence[str],
    workspace: Path,
    harvest_root: Path,
    report_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the combined dataset and a provenance manifest."""
    accepted = {r["name"]: r for r in accepted_repos(report_path)}

    unknown = [r for r in repos if r not in accepted]
    if unknown:
        raise ValueError(
            f"Not accepted by screening: {unknown}. Eligible: {sorted(accepted)}"
        )

    frames: List[pd.DataFrame] = []
    manifest_repos: List[Dict[str, Any]] = []

    for repo in repos:
        entry = accepted[repo]
        repo_root = (workspace / repo).resolve()
        if not repo_root.is_dir():
            raise FileNotFoundError(
                f"{repo}: checkout missing at {repo_root}. "
                f"Re-run scripts/screen_repos.py to restore it from the pinned SHA."
            )
        harvest = load_harvest(
            repo=repo,
            repo_root=repo_root,
            harvest_dir=harvest_root / repo,
            commit_sha=str(entry.get("commit_sha", "")),
        )

        resolved: Dict[str, ResolvedTest] = {}
        unresolved: List[str] = []
        for test_id in harvest.universe:
            hit = resolve_test_id(test_id, harvest.repo_root)
            if hit is None:
                unresolved.append(test_id)
            else:
                resolved[test_id] = hit

        if not resolved:
            raise ValueError(
                f"{repo}: not one of {len(harvest.universe)} test IDs resolved to a file "
                f"under {harvest.repo_root}. The checkout is missing or the ID format changed."
            )
        if unresolved:
            logger.warning(
                f"  {repo}: {len(unresolved)}/{len(harvest.universe)} test IDs did not "
                f"resolve to a file and were dropped, e.g. {unresolved[:3]}"
            )

        rows = list(build_rows(harvest, resolved))
        frame = pd.DataFrame(rows)
        # A row lives as a Python dict at ~2.7 KiB before pandas packs it into
        # ~360 bytes of typed columns. Releasing the dicts here bounds peak memory
        # at one repository's worth instead of two consecutive ones, which matters
        # because the largest subject contributes ~224k rows.
        del rows
        frames.append(frame)

        n_fail = int(frame["label_failed"].sum())
        rate = n_fail / len(frame) if len(frame) else 0.0
        logger.info(
            f"  {repo}: {len(harvest.mutants)} mutants x {len(resolved)} tests "
            f"= {len(frame):,} rows, {n_fail:,} failures ({rate:.2%})"
        )

        manifest_repos.append({
            "repo": repo,
            "repo_commit_sha": harvest.commit_sha,
            "repo_root": str(harvest.repo_root),
            "n_mutants_used": len(harvest.mutants),
            "n_mutants_excluded_by_status": harvest.excluded_status,
            "mutants_excluded_as_contaminated": harvest.excluded_contaminated,
            "universe_size": len(harvest.universe),
            "n_tests_resolved": len(resolved),
            "n_tests_unresolved": len(unresolved),
            "unresolved_examples": unresolved[:5],
            "n_rows": len(frame),
            "n_failures": n_fail,
            "failure_rate": round(rate, 6),
            "g1_pairs_met": bool(len(frame) >= G1_MIN_PAIRS_PER_REPO),
            "g1_failure_rate_in_band": bool(
                G1_MIN_FAILURE_RATE <= rate <= G1_MAX_FAILURE_RATE
            ),
            "import_provenance": harvest.provenance,
        })
        if not (G1_MIN_FAILURE_RATE <= rate <= G1_MAX_FAILURE_RATE):
            logger.warning(
                f"  {repo}: failure rate {rate:.2%} is outside the G1 band of "
                f"{G1_MIN_FAILURE_RATE:.0%}-{G1_MAX_FAILURE_RATE:.0%}. Its rows "
                f"stay in the dataset and this stays in the manifest; the pooled "
                f"rate does not excuse it."
            )

    dataset = pd.concat(frames, ignore_index=True)

    missing = [name for name in FEATURE_NAMES if name not in dataset.columns]
    if missing:
        raise ValueError(f"Feature columns missing from the built dataset: {missing}")

    inert = [
        name for name in FEATURE_NAMES
        if dataset[name].nunique(dropna=False) <= 1
    ]

    total_rate = float(dataset["label_failed"].mean())
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_source": "measured pytest outcomes under AST mutation; no value is drawn or imputed",
        "n_rows": int(len(dataset)),
        "n_failures": int(dataset["label_failed"].sum()),
        "failure_rate": round(total_rate, 6),
        "g1_failure_rate_band": [G1_MIN_FAILURE_RATE, G1_MAX_FAILURE_RATE],
        "g1_failure_rate_in_band": bool(
            G1_MIN_FAILURE_RATE <= total_rate <= G1_MAX_FAILURE_RATE
        ),
        "g1_min_pairs_per_repo": G1_MIN_PAIRS_PER_REPO,
        "g1_pairs_met_by_every_repo": all(r["g1_pairs_met"] for r in manifest_repos),
        "g1_failure_rate_in_band_for_every_repo": all(
            r["g1_failure_rate_in_band"] for r in manifest_repos
        ),
        "g1_repos_outside_the_band": [
            {"repo": r["repo"], "failure_rate": r["failure_rate"]}
            for r in manifest_repos if not r["g1_failure_rate_in_band"]
        ],
        "n_repos": len(manifest_repos),
        "repos": manifest_repos,
        "inert_features": inert,
        "message_derived_features_are_inert_because":
            "a mutant carries no commit message; an empty string is used rather "
            "than invented prose",
        "commit_timestamp_is_a_surrogate":
            "mutants have no wall-clock time; commit_timestamp encodes harvest "
            "order (one second per mutant from ORDER_EPOCH) so temporal splits "
            "respect the order in which history features accumulated. Use "
            "mutant_index for the ordering itself",
        "history_is_causal":
            "features for mutant i are read before mutant i's outcomes are "
            "recorded, so no row sees its own or any later label",
        "import_provenance_verified_for_every_repo": all(
            bool(r["import_provenance"].get("verified")) for r in manifest_repos
        ),
        "import_provenance_means":
            "the interpreter that ran each subject suite was checked, before any "
            "mutant, to import the package from the checkout being mutated. A "
            "harvest without that proof is refused rather than included",
    }
    return dataset, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the real feature dataset from harvested mutation labels."
    )
    parser.add_argument("--repos", nargs="*", default=None, help="repository names to include")
    parser.add_argument("--all", action="store_true", help="include every screened-accepted repo")
    parser.add_argument("--workspace", default="data/repos", help="where the checkouts live")
    parser.add_argument("--harvest-root", default="data/harvest", help="where the harness wrote its output")
    parser.add_argument("--out", default="data/processed/real_features.csv", help="dataset output path")
    args = parser.parse_args()

    configure_logging()

    workspace = (REPO_ROOT / args.workspace).resolve()
    harvest_root = (REPO_ROOT / args.harvest_root).resolve()
    report_path = workspace / "screening_report.json"

    if args.all:
        repos = [r["name"] for r in accepted_repos(report_path)]
    elif args.repos:
        repos = list(args.repos)
    else:
        parser.error("choose --all or --repos NAME [NAME ...]")

    logger.info(f"Building dataset from {len(repos)} repository/ies: {', '.join(repos)}")
    dataset, manifest = build_dataset(repos, workspace, harvest_root, report_path)

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)

    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 74)
    logger.info(f"Rows            : {manifest['n_rows']:,}")
    logger.info(f"Failures        : {manifest['n_failures']:,} ({manifest['failure_rate']:.2%})")
    band = manifest["g1_failure_rate_band"]
    verdict = "in band" if manifest["g1_failure_rate_in_band"] else "OUT OF BAND"
    logger.info(f"G1 pooled rate  : {band[0]:.0%}-{band[1]:.0%} -> {verdict}")
    outside = manifest["g1_repos_outside_the_band"]
    if outside:
        listed = ", ".join(f"{o['repo']} {o['failure_rate']:.2%}" for o in outside)
        logger.warning(f"G1 per repo     : OUT OF BAND for {listed}")
    else:
        logger.info("G1 per repo     : every repo in band")
    pairs = "met by every repo" if manifest["g1_pairs_met_by_every_repo"] else "NOT met"
    logger.info(f"G1 min pairs    : {manifest['g1_min_pairs_per_repo']:,} -> {pairs}")
    logger.info(f"Inert features  : {len(manifest['inert_features'])} "
                f"({', '.join(manifest['inert_features']) or 'none'})")
    logger.info(f"Dataset         : {out_path}")
    logger.info(f"Manifest        : {manifest_path}")
    logger.info("=" * 74)

    if not manifest["g1_failure_rate_in_band"]:
        logger.warning(
            "Failure rate is outside the G1 band. This is reported, not corrected: "
            "investigate the harvest rather than reweighting the dataset."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
