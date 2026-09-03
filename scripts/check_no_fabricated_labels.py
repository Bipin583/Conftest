"""
CI guard: fail the build if any label-producing code path draws a random number.

This exists because the project previously shipped an evaluation built on
fabricated ground truth. `real_repo_miner.py` -- the module whose entire purpose
was to mine REAL data -- decided whether a test failed with:

    label = 1 if np.random.rand() < 0.70 else 0

and stamped the output "REAL_GIT_MINED". Six structurally different baselines
then all scored exactly 20% recall, because a p=0.70 Bernoulli draw is ~30%
irreducibly unpredictable and no feature can beat that ceiling.

The rules this script enforces
------------------------------
Rule 1 -- the label path. A test outcome must be MEASURED by executing the
suite, never SAMPLED. Randomness is legitimate for: choosing which mutants to
sample, train/test shuffling, model seeds, and bootstrap resampling. It is never
legitimate for deciding whether a test passed.

Rule 2 -- the reporting path. Rule 1 passed on 108 files while three scripts in
reports/ were publishing results computed over inputs they had generated
themselves: run_statistical_tests.py drew 100 commits of recall from a Beta,
run_cross_repo_eval.py invented four repositories and named them after real
GitHub projects, and run_continuous_learning.py injected the concept drift it
then reported detecting. None of them assigned a label, so Rule 1 had nothing to
catch. So: a script that writes a report must read an input artifact, and a
feature matrix on the reporting path may not be drawn from a random number
generator.

Usage
-----
    python scripts/check_no_fabricated_labels.py          # exit 1 on violation
    python scripts/check_no_fabricated_labels.py --list   # show what is scanned
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Names that mean "a test outcome" in this codebase.
LABEL_NAMES = {
    "label", "label_failed", "labels", "is_fail", "is_failed", "failed",
    "outcome", "test_outcome", "fail_prob", "failure_rate_draw", "y", "y_true",
    # y_init and y_comm are what the drift simulation called its Bernoulli draws, and
    # a bare "y" was the only spelling this set knew.
    "y_init", "y_comm", "y_commit", "y_train", "y_val", "y_test", "y_target",
    "y_mock", "y_drift", "y_source", "y_noisy",
}

# Attribute/function names that produce randomness.
RANDOM_CALLS = {
    "rand", "random", "randint", "random_sample", "uniform", "choice",
    "binomial", "bernoulli", "exponential", "normal", "poisson", "sample",
    "randrange", "shuffle", "permutation", "random_integers", "gauss", "betavariate",
    # randn and standard_normal were missing until 2026-09-03. Every one of the three
    # reporting-path fabricators built its feature matrix with rng.randn, so the omission
    # was the whole reason this file could pass while reports/ was fiction.
    "randn", "standard_normal", "beta", "gamma", "lognormal", "laplace", "logistic",
    "chisquare", "geometric", "negative_binomial", "rayleigh", "triangular", "weibull",
    "pareto", "power", "wald", "hypergeometric", "multivariate_normal", "dirichlet",
    "default_rng", "RandomState",
}

# Files exempt because randomness there is legitimate and reviewed.
EXEMPT_FILES = {
    # Explicitly synthetic: name says so, construction requires
    # acknowledge_synthetic=True, and every commit and test run is stamped
    # data_origin=SYNTHETIC_FABRICATED_LABELS. Smoke tests and fixtures only.
    "src/conftest/repository/synthetic_generator.py",
    # This script quotes the forbidden pattern in its own docstring.
    "scripts/check_no_fabricated_labels.py",
    # Explicitly synthetic: name says so, construction requires
    # acknowledge_synthetic=True, and every row is stamped
    # data_origin=SYNTHETIC_FABRICATED_LABELS. Smoke tests only.
    "src/benchmark/dataset_generator.py",
}

# Directories scanned. Tests are excluded: fixtures may fabricate freely.
SCAN_DIRS = ["src", "scripts"]

# Rule 2. Names that mean "the feature matrix the model will be scored on".
FEATURE_MATRIX_NAMES = {
    "x", "x_train", "x_val", "x_test", "x_init", "x_batch", "x_mock", "x_comm",
    "x_commit", "x_target", "x_source", "x_drift", "features", "feature_matrix",
    "feature_rows",
}

# Calls that mean "this code read something off disk or out of an artifact".
READ_CALLS = {
    "read_csv", "read_json", "read_parquet", "read_text", "read_bytes", "load",
    "loads", "load_ensemble", "open", "read", "iterdir", "rglob", "glob",
    # A database query is a read. extract_features.py takes its rows from SQLite via
    # SQLAlchemy and writes a CSV, which is a measurement pipeline, not a fabricator.
    "select", "execute", "query", "scalars", "scalar_one_or_none", "first", "all",
    # Some scripts reach their input through a service object rather than a file handle:
    # collect_repository_data.py mines git through CollectorService, select_tests.py reads a
    # checkout through GitRepositoryMiner. These are named explicitly rather than matched by
    # a pattern, so that a new script publishing a report without a visible input source
    # fails this check until somebody answers "where do these rows come from?".
    "SessionLocal", "CollectorService", "GitRepositoryMiner", "ConfTestEngine",
    "collect", "extract_commit_diff", "get_or_create_repository", "analyze_and_select",
}

# Calls that mean "this code published something".
WRITE_CALLS = {"dump", "to_csv", "to_json", "write_text", "write_bytes", "savefig"}

# Reporting-path files where generated features are the measurement, not the input.
GENERATOR_EXEMPT = {
    # Rule 2 does not apply to a module whose declared job is fabrication; both are
    # gated behind acknowledge_synthetic=True and stamp every row SYNTHETIC.
    "src/conftest/repository/synthetic_generator.py",
    "src/benchmark/dataset_generator.py",
}


class Violation(NamedTuple):
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _is_random_call(node: ast.AST) -> str:
    """Return a description if the node is a randomness-producing call."""
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in RANDOM_CALLS:
        base = func.value
        prefix = ""
        if isinstance(base, ast.Name):
            prefix = base.id
        elif isinstance(base, ast.Attribute):
            prefix = base.attr
        return f"{prefix}.{func.attr}()" if prefix else f"{func.attr}()"
    if isinstance(func, ast.Name) and func.id in RANDOM_CALLS:
        return f"{func.id}()"
    return ""


def _opens_for_reading(node: ast.Call) -> bool:
    """
    True when an open() call is a read.

    open(path, "w") is how a report gets written, so treating every open() as evidence
    that the script read an input is how the publishes-without-reading check ends up
    unable to fire at all.
    """
    mode = None
    if len(node.args) >= 2:
        mode = node.args[1]
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return True  # default mode is "r"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return mode.value.startswith("r")
    return True  # a computed mode is not worth guessing about


def _targets(node: ast.AST) -> List[str]:
    """Names assigned to by an assignment-like node."""
    names: List[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Subscript):
                # df["label"] = ...
                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    names.append(target.slice.value)
            elif isinstance(target, ast.Attribute):
                names.append(target.attr)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        target = node.target
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def scan_file(path: Path) -> List[Violation]:
    """Flag assignments that bind a label-ish name to a random draw."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=rel)
    except SyntaxError as exc:
        return [Violation(rel, exc.lineno or 0, f"could not parse: {exc.msg}")]

    violations: List[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue

        assigned = [n.lower() for n in _targets(node)]
        if not any(name in LABEL_NAMES for name in assigned):
            continue

        # Does the right-hand side involve randomness anywhere?
        value = node.value
        if value is None:
            continue
        for inner in ast.walk(value):
            detail = _is_random_call(inner)
            if detail:
                violations.append(
                    Violation(
                        rel,
                        getattr(node, "lineno", 0),
                        f"'{assigned[0]}' is assigned from {detail} -- a test "
                        f"outcome must be measured, not sampled",
                    )
                )
                break

    return violations


def scan_reporting_path(path: Path) -> List[Violation]:
    """
    Flag a report that was computed over an input the script made up.

    Two checks, both narrow enough to name the offending line:

    * a feature matrix assigned from a random draw, anywhere on the reporting path;
    * a script that publishes a report while never reading an artifact, which is the
      shape all three fabricators had -- no dataset in, a results file out.
    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel in GENERATOR_EXEMPT or rel in EXEMPT_FILES:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=rel)
    except SyntaxError:
        return []  # Rule 1 already reports the parse failure.

    violations: List[Violation] = []
    reads: List[str] = []
    writes: List[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "open":
                if _opens_for_reading(node):
                    reads.append(name)
            elif name in READ_CALLS or name.startswith("load_"):
                reads.append(name)
            elif name in WRITE_CALLS:
                writes.append(getattr(node, "lineno", 0))

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assigned = [n.lower() for n in _targets(node)]
            if not any(name in FEATURE_MATRIX_NAMES for name in assigned):
                continue
            value = node.value
            if value is None:
                continue
            for inner in ast.walk(value):
                detail = _is_random_call(inner)
                if detail:
                    violations.append(
                        Violation(
                            rel,
                            getattr(node, "lineno", 0),
                            f"'{assigned[0]}' is a feature matrix assigned from {detail} "
                            f"-- a reported result must be computed over rows that were "
                            f"harvested, not drawn",
                        )
                    )
                    break

    if writes and not reads and rel.startswith("scripts/"):
        violations.append(
            Violation(
                rel,
                writes[0],
                "publishes a report but never reads an input artifact -- if the input is "
                "generated in-process, the output is not a measurement",
            )
        )

    return violations


def iter_scanned_files() -> List[Path]:
    """Python files under the scanned directories, minus exemptions."""
    files: List[Path] = []
    for rel_dir in SCAN_DIRS:
        base = REPO_ROOT / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.relative_to(REPO_ROOT).as_posix() in EXEMPT_FILES:
                continue
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list scanned files and exit")
    args = parser.parse_args()

    files = iter_scanned_files()

    if args.list:
        print(f"Scanning {len(files)} files under {SCAN_DIRS}:")
        for path in files:
            print(f"  {path.relative_to(REPO_ROOT).as_posix()}")
        print(f"\nExempt ({len(EXEMPT_FILES)}):")
        for rel in sorted(EXEMPT_FILES):
            print(f"  {rel}")
        return 0

    violations: List[Violation] = []
    for path in files:
        violations.extend(scan_file(path))
        violations.extend(scan_reporting_path(path))

    if violations:
        print("=" * 74)
        print("FABRICATED LABEL CHECK: FAILED")
        print("=" * 74)
        for violation in violations:
            print(f"  {violation}")
        print()
        print("A test outcome must be observed by running the suite.")
        print("Use conftest.groundtruth.mutation_harness, which injects a fault,")
        print("executes pytest, and records which tests actually failed.")
        print("A reported result must be computed over harvested rows: read the")
        print("dataset, or raise MissingArtifact naming the command that builds it.")
        print("=" * 74)
        return 1

    print(f"FABRICATED LABEL CHECK: PASSED ({len(files)} files scanned, both rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
