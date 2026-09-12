"""
The superseded duplicate trees must say so, and nothing shipping may import them.

`src/` carries two implementations of the same responsibilities: the maintained
package `src/conftest/`, and older flat modules (`src/features/`, `src/models/`,
`src/engine/`, `src/dashboard/`) that predate it. The duplicates are not dead --
`tests/test_ast_parser.py`, `tests/test_calibration.py` and
`tests/test_selective_engine.py` import them -- so deleting them would drop
those tests, but leaving them unlabelled is worse: `src/models/lightgbm_model.py`
and `conftest/models/lightgbm_model.py` are the same filename with different
behaviour, and only one of them received the row-bagging fix. A reader who
opens the wrong one, or a `pythonpath = ["src"]` import that resolves
`models.lightgbm_model` instead of `conftest.models.lightgbm_model`, gets the
unmaintained implementation without being told.

These tests keep the labelling honest: every duplicate declares its
replacement, every declared replacement exists, and no shipping tree imports a
duplicate.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
CONFTEST_PKG = SRC / "conftest"

# Trees whose code is served, trained from, or rendered to users.
SHIPPING_TREES = (SRC / "conftest", PROJECT_ROOT / "scripts", PROJECT_ROOT / "dashboard")

# Top-level names that `pythonpath = ["src"]` makes importable and that resolve
# to a superseded flat module rather than to the conftest package.
LEGACY_TOP_LEVEL = ("features", "models", "engine", "benchmark")

MARKER = "__superseded_by__"


def _legacy_modules():
    """Every .py under src/ that is not part of the conftest package."""
    for path in sorted(SRC.rglob("*.py")):
        if CONFTEST_PKG in path.parents or "__pycache__" in path.parts:
            continue
        if ".egg-info" in str(path):
            continue
        yield path


def _declared_replacement(path: Path):
    match = re.search(
        rf'^{MARKER}\s*=\s*"([^"]+)"',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _duplicate_modules():
    """
    Legacy modules whose filename collides with a conftest module, plus the
    hand-listed pairs whose names differ but whose responsibility is the same.
    """
    explicit = {
        "src/features/ast_parser.py",
        "src/features/history_miner.py",
        "src/engine/selective_engine.py",
        "src/dashboard/app.py",
        "src/models/colab_trainer.py",
    }
    found = set()
    for path in _legacy_modules():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in explicit:
            found.add(rel)
            continue
        # e.g. src/models/calibration.py vs src/conftest/models/calibration.py
        twin = CONFTEST_PKG / path.parent.name / path.name
        if twin.exists():
            found.add(rel)
    return sorted(found)


def test_the_sweep_finds_the_known_duplicates():
    """Guard the guard: an over-narrow sweep would make the rest vacuous."""
    duplicates = _duplicate_modules()

    assert "src/models/lightgbm_model.py" in duplicates
    assert "src/engine/selective_engine.py" in duplicates
    assert len(duplicates) >= 8


@pytest.mark.parametrize("rel", _duplicate_modules())
def test_every_duplicate_declares_its_replacement(rel):
    replacement = _declared_replacement(PROJECT_ROOT / rel)

    assert replacement, (
        f"{rel} duplicates a maintained module but does not declare "
        f"{MARKER}. An unlabelled duplicate is indistinguishable from the "
        f"implementation that is actually shipped."
    )


@pytest.mark.parametrize("rel", _duplicate_modules())
def test_every_declared_replacement_exists(rel):
    replacement = _declared_replacement(PROJECT_ROOT / rel)
    if not replacement:
        pytest.skip("covered by test_every_duplicate_declares_its_replacement")

    if replacement.endswith(".py"):
        assert (PROJECT_ROOT / replacement).is_file(), (
            f"{rel} points at {replacement}, which does not exist"
        )
    else:
        assert importlib.util.find_spec(replacement) is not None, (
            f"{rel} points at module {replacement}, which cannot be imported"
        )


def _imported_top_level_names(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _shipping_sources():
    for tree in SHIPPING_TREES:
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_shipping_module_imports_a_superseded_tree():
    """
    `pythonpath = ["src"]` makes `import models` resolve to the legacy tree.

    So this is not a style rule: a single `from models.lightgbm_model import
    ...` inside src/conftest would serve the unmaintained predictor while every
    report still said conftest.
    """
    offenders = []
    for path in _shipping_sources():
        hits = _imported_top_level_names(path) & set(LEGACY_TOP_LEVEL)
        if hits:
            offenders.append((path.relative_to(PROJECT_ROOT).as_posix(), sorted(hits)))

    assert not offenders, f"shipping code imports superseded trees: {offenders}"
