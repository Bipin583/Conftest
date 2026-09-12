"""
Every "produced by" string must name a script that exists.

Several modules refuse to guess: when an artifact is absent they raise
MissingArtifact naming the command that would produce it, so a reader is told
how to get the number rather than handed a fabricated one. That contract is
only as good as the name. Three of these strings named scripts that were never
in the repository -- ``scripts/analyze_uncertainty.py`` (the real producer is
``uncertainty_eval.py``), ``scripts/explain_predictions.py``
(``generate_explanations.py``), and ``scripts/run_calibration.py``
(``calibrate_model.py``) -- so the honest-refusal path sent anyone who hit it
to a file that does not exist. Nothing checked, because the tests asserted the
literal strings back, which is how the wrong ones stayed wrong.

The documentation is swept for the same reason. Restating the README against
the measured artifacts meant writing "produced by ``scripts/run_ablation.py``"
and "``scripts/evaluate_g5.py``" for files actually named
``run_ablation_study.py`` and ``check_g5_recall_floor.py``. A reproduction
command that does not run is the same defect as a refusal that points nowhere:
the number stops being checkable. So prose counts as a reference site.

This sweeps the trees instead of enumerating call sites, so a new reference to
a script that does not exist fails here rather than in front of a user.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Trees that ship: library code, the Streamlit dashboard, and the scripts
# themselves (they reference each other). `tests/` is excluded because test
# fixtures legitimately name synthetic paths like `scripts/fake.py`.
SEARCH_TREES = ("src", "dashboard", "scripts")

# Documentation that tells a reader how to reproduce a number. Directories are
# swept recursively; individual files are taken as-is.
DOC_TREES = ("docs", "slides", "paper", "ktu_report")
DOC_FILES = ("README.md", "DOCUMENTATION.md", "RELEASE_NOTES.md")
DOC_SUFFIXES = (".md", ".tex", ".html", ".rst")

SCRIPT_REFERENCE = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")


def _python_sources():
    for tree in SEARCH_TREES:
        for path in sorted((PROJECT_ROOT / tree).rglob("*.py")):
            if "__pycache__" in path.parts or ".egg-info" in str(path):
                continue
            yield path


def _doc_sources():
    for name in DOC_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            yield path
    for tree in DOC_TREES:
        root = PROJECT_ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in DOC_SUFFIXES:
                yield path


def _references_in(paths):
    """(source file, referenced script name) for every scripts/*.py mention."""
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in SCRIPT_REFERENCE.finditer(text):
            found.append((path.relative_to(PROJECT_ROOT).as_posix(), match.group(1)))
    return found


def _code_references():
    return _references_in(_python_sources())


def _doc_references():
    return _references_in(_doc_sources())


def test_the_code_sweep_actually_finds_references():
    """Guard the guard: a broken regex would make every assertion below vacuous."""
    refs = _code_references()
    assert len(refs) > 20, f"expected the source trees to name many scripts, found {refs}"
    assert any(name == "tune_policy.py" for _, name in refs)


def test_the_doc_sweep_actually_finds_references():
    """Same guard for the documentation half."""
    refs = _doc_references()
    assert len(refs) > 5, f"expected the docs to name reproduction commands, found {refs}"
    assert any(source == "README.md" for source, _ in refs)


@pytest.mark.parametrize("source, script", sorted(set(_code_references())))
def test_referenced_script_exists(source, script):
    assert (SCRIPTS_DIR / script).is_file(), (
        f"{source} names 'scripts/{script}', which does not exist. "
        f"A produced-by string that points at a missing file turns an honest "
        f"refusal into a dead end -- name the script that really writes the "
        f"artifact, or add it."
    )


@pytest.mark.parametrize("source, script", sorted(set(_doc_references())))
def test_documented_script_exists(source, script):
    assert (SCRIPTS_DIR / script).is_file(), (
        f"{source} tells a reader to run 'scripts/{script}', which does not exist. "
        f"Every published number has to be reproducible by the command printed "
        f"next to it -- fix the name or add the script."
    )
