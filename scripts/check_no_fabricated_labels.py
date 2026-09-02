"""
CI guard: fail the build if any label-producing code path draws a random number.

This exists because the project previously shipped an evaluation built on
fabricated ground truth. `real_repo_miner.py` -- the module whose entire purpose
was to mine REAL data -- decided whether a test failed with:

    label = 1 if np.random.rand() < 0.70 else 0

and stamped the output "REAL_GIT_MINED". Six structurally different baselines
then all scored exactly 20% recall, because a p=0.70 Bernoulli draw is ~30%
irreducibly unpredictable and no feature can beat that ceiling.

The rule this script enforces
-----------------------------
A test outcome must be MEASURED by executing the suite, never SAMPLED.
Randomness is legitimate for: choosing which mutants to sample, train/test
shuffling, model seeds, and bootstrap resampling. It is never legitimate for
deciding whether a test passed.

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
}

# Attribute/function names that produce randomness.
RANDOM_CALLS = {
    "rand", "random", "randint", "random_sample", "uniform", "choice",
    "binomial", "bernoulli", "exponential", "normal", "poisson", "sample",
    "randrange", "shuffle", "permutation", "random_integers", "gauss", "betavariate",
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
        print("=" * 74)
        return 1

    print(f"FABRICATED LABEL CHECK: PASSED ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
