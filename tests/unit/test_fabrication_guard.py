"""
Tests for the fabrication guard itself.

The guard is the only thing standing between this repository and another round of
published numbers computed over invented data, so it needs its own regression test.
Every fixture below is a reduction of a script that actually shipped and actually
passed the guard at the time -- the omissions those scripts walked through (randn
absent from the random-call list, open(path, "w") counted as a read) are the reason
these cases are pinned.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "check_no_fabricated_labels.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("fabrication_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, source: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _scan(guard, monkeypatch, root: Path, path: Path):
    monkeypatch.setattr(guard, "REPO_ROOT", root)
    return guard.scan_file(path) + guard.scan_reporting_path(path)


def test_sampled_label_is_flagged(guard, monkeypatch, tmp_path):
    """A Bernoulli draw is not a test outcome, whatever the variable is called."""
    path = _write(
        tmp_path,
        "scripts/fake.py",
        "import numpy as np\n"
        "rng = np.random.RandomState(0)\n"
        "df = read_csv('x.csv')\n"
        "y_init = (rng.rand(150) < 0.08).astype(int)\n",
    )
    found = _scan(guard, monkeypatch, tmp_path, path)
    assert any("must be measured, not sampled" in v.detail for v in found)


def test_feature_matrix_from_randn_is_flagged(guard, monkeypatch, tmp_path):
    """randn was missing from the random-call list; that omission hid three fabricators."""
    path = _write(
        tmp_path,
        "scripts/fake.py",
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "df = read_csv('x.csv')\n"
        "X_init = rng.randn(150, 40)\n",
    )
    found = _scan(guard, monkeypatch, tmp_path, path)
    assert any("feature matrix assigned from" in v.detail for v in found)


def test_report_without_any_input_read_is_flagged(guard, monkeypatch, tmp_path):
    """
    The shape all three fabricators had: no dataset in, a results file out.

    open(out_path, "w") used to satisfy the read check, which made this rule
    unreachable for every script that writes a report.
    """
    path = _write(
        tmp_path,
        "scripts/fake.py",
        "import json\n"
        "results = {'auc': 0.99}\n"
        "with open('reports/fake.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(results, f)\n",
    )
    found = _scan(guard, monkeypatch, tmp_path, path)
    assert any("never reads an input artifact" in v.detail for v in found)


def test_reading_a_dataset_then_writing_a_report_passes(guard, monkeypatch, tmp_path):
    """The legitimate shape must stay quiet, or the guard gets switched off."""
    path = _write(
        tmp_path,
        "scripts/real.py",
        "import json\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data/processed/real_features.csv')\n"
        "results = {'rows': len(df)}\n"
        "with open('reports/real.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(results, f)\n",
    )
    assert _scan(guard, monkeypatch, tmp_path, path) == []


def test_open_for_reading_still_counts_as_a_read(guard, monkeypatch, tmp_path):
    """Mode-awareness must not go so far as to stop recognising a plain read."""
    path = _write(
        tmp_path,
        "scripts/real.py",
        "import json\n"
        "with open('data/splits/test.json') as f:\n"
        "    rows = json.load(f)\n"
        "with open('reports/real.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump({'n': len(rows)}, f)\n",
    )
    assert _scan(guard, monkeypatch, tmp_path, path) == []


def test_working_tree_passes_both_rules(guard):
    """The repository itself has to be clean, which is the point of the guard."""
    violations = []
    for path in guard.iter_scanned_files():
        violations.extend(guard.scan_file(path))
        violations.extend(guard.scan_reporting_path(path))
    assert violations == [], "\n".join(str(v) for v in violations)
