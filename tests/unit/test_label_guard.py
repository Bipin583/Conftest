"""
Tests for the fabricated-label CI guard.

The guard exists because the project shipped an evaluation whose ground truth
was `np.random.rand() < 0.70`. These tests verify it actually catches that
pattern and its variants, and does not fire on legitimate randomness.
"""

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_no_fabricated_labels import (  # noqa: E402
    iter_scanned_files,
    scan_file,
)


def _scan_source(tmp_path: Path, code: str, name: str = "probe.py"):
    """Write a snippet into the repo tree and scan it."""
    target = REPO_ROOT / "src" / "_guard_probe.py"
    target.write_text(textwrap.dedent(code), encoding="utf-8")
    try:
        return scan_file(target)
    finally:
        target.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Must catch
# --------------------------------------------------------------------------

def test_catches_the_original_defect(tmp_path):
    """The exact line that invalidated the original evaluation."""
    violations = _scan_source(tmp_path, """
        import numpy as np
        def build():
            label = 1 if np.random.rand() < 0.70 else 0
            return label
    """)
    assert len(violations) == 1
    assert "label" in violations[0].detail
    assert "measured, not sampled" in violations[0].detail


def test_catches_stdlib_random(tmp_path):
    violations = _scan_source(tmp_path, """
        import random
        def build(rate):
            is_fail = random.random() < rate
            return is_fail
    """)
    assert len(violations) == 1


def test_catches_rng_attribute_style(tmp_path):
    """The synthetic_generator pattern: self.rng.random() < fail_prob."""
    violations = _scan_source(tmp_path, """
        class Gen:
            def build(self, fail_prob):
                is_fail = self.rng.random() < fail_prob
                return is_fail
    """)
    assert len(violations) == 1


def test_catches_dataframe_column_assignment(tmp_path):
    violations = _scan_source(tmp_path, """
        import numpy as np
        def build(df, n):
            df["label_failed"] = np.random.binomial(1, 0.3, n)
            return df
    """)
    assert len(violations) == 1


def test_catches_choice_and_uniform(tmp_path):
    violations = _scan_source(tmp_path, """
        import numpy as np
        def build():
            outcome = np.random.choice(["PASS", "FAIL"])
            return outcome
    """)
    assert len(violations) == 1


def test_catches_randomness_nested_in_expression(tmp_path):
    """The draw may be buried inside a larger expression."""
    violations = _scan_source(tmp_path, """
        import numpy as np
        def build(hist):
            label = int(bool(np.random.rand() < (hist * 0.1)))
            return label
    """)
    assert len(violations) == 1


# --------------------------------------------------------------------------
# Must NOT catch -- legitimate randomness
# --------------------------------------------------------------------------

def test_allows_mutant_sampling(tmp_path):
    """Choosing WHICH mutants to run is the one legitimate use of randomness."""
    violations = _scan_source(tmp_path, """
        import random
        def sample(candidates, n):
            rng = random.Random(42)
            chosen = rng.sample(candidates, n)
            return chosen
    """)
    assert violations == []


def test_allows_train_test_shuffling(tmp_path):
    violations = _scan_source(tmp_path, """
        import numpy as np
        def split(x):
            order = np.random.permutation(len(x))
            return x[order]
    """)
    assert violations == []


def test_allows_bootstrap_resampling(tmp_path):
    violations = _scan_source(tmp_path, """
        import numpy as np
        def bootstrap(values, n):
            resampled = np.random.choice(values, size=n, replace=True)
            return resampled.mean()
    """)
    assert violations == []


def test_allows_label_assigned_from_measurement(tmp_path):
    """A label derived from an observed test status is exactly what we want."""
    violations = _scan_source(tmp_path, """
        def label_tests(killed, universe):
            labels = {t: (1 if t in killed else 0) for t in universe}
            return labels
    """)
    assert violations == []


def test_allows_model_seed(tmp_path):
    violations = _scan_source(tmp_path, """
        import numpy as np
        def seed_everything():
            seed = np.random.randint(0, 1000)
            return seed
    """)
    assert violations == []


# --------------------------------------------------------------------------
# The live repository must stay clean
# --------------------------------------------------------------------------

def test_repository_has_no_fabricated_labels():
    """Regression lock: the real tree must pass its own guard."""
    violations = []
    for path in iter_scanned_files():
        violations.extend(scan_file(path))

    assert violations == [], "\n".join(str(v) for v in violations)


def test_guard_scans_a_meaningful_number_of_files():
    """A guard that silently scans nothing would always pass."""
    assert len(iter_scanned_files()) > 50


def test_unparseable_file_is_reported_not_skipped(tmp_path):
    """A syntax error must surface, not be silently treated as clean."""
    violations = _scan_source(tmp_path, """
        def broken(:
    """)
    assert len(violations) == 1
    assert "could not parse" in violations[0].detail
