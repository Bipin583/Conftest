"""Fixtures shared across the suite.

Two kinds of input appear here, and the distinction matters for what a failing
test proves:

* **Synthetic** fixtures build probabilities and labels from a *known*
  generative process. When a conformal guarantee is checked against these, the
  ground truth is genuinely known, so the test measures the estimator rather
  than the corpus.
* **Artefact** fixtures load the committed ``.pkl`` files. They verify the
  shipped model, and they skip -- rather than fail -- when the artefacts are
  absent, so a fresh clone that has not trained yet still gets a green suite
  for everything that does not depend on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from common import load_config, resolve_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    """The project configuration as every module sees it."""
    return load_config()


@pytest.fixture(scope="session")
def targets(config: Dict[str, Any]) -> Dict[str, float]:
    """The acceptance gates from ``config.yaml``."""
    return config["targets"]


# --------------------------------------------------------------------------
# Synthetic scores
# --------------------------------------------------------------------------


def make_scores(
    n: int = 4000,
    failure_rate: float = 0.05,
    separation: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw calibrated-ish probabilities with a known label process.

    Failing rows are drawn from a Beta distribution shifted towards 1 and
    passing rows towards 0, with ``separation`` controlling the overlap. The
    result behaves like a usable-but-imperfect ranker, which is the regime the
    conformal layer is designed for.

    Args:
        n: Number of rows.
        failure_rate: Fraction of rows labelled 1.
        separation: Higher values push the two score distributions apart.
        seed: RNG seed.

    Returns:
        A ``(probabilities, labels)`` pair.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < failure_rate).astype(int)
    probabilities = np.where(
        labels == 1,
        rng.beta(separation, 1.0, size=n),
        rng.beta(1.0, separation * 4, size=n),
    )
    return np.clip(probabilities, 1e-6, 1 - 1e-6), labels


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray]:
    """A default synthetic ``(probabilities, labels)`` pair."""
    return make_scores()


@pytest.fixture
def selector(scores):
    """A PAC selector fitted to the synthetic calibration scores."""
    from models.conformal import build_selector

    probabilities, labels = scores
    return build_selector(probabilities, labels, coverage=0.95, confidence=0.90)["selector"]


# --------------------------------------------------------------------------
# Corpus-shaped frames
# --------------------------------------------------------------------------


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A tiny frame shaped like the harvested corpus.

    Four commits, two tests each, in chronological order. ``tests/test_api.py``
    fails on the last two commits, which gives the history-window features
    something real to pick up and lets a leakage test check that the first
    occurrence of a test has no knowledge of its own future.
    """
    rows: List[Dict[str, Any]] = []
    commits = [
        ("aaa1", "2024-01-01T09:00:00", "src/api/views.py", 40, 5, "fix: handle empty payload"),
        ("bbb2", "2024-01-05T14:30:00", "src/api/views.py", 12, 3, "refactor: extract helper"),
        ("ccc3", "2024-01-09T21:15:00", "src/core/engine.py", 200, 80, "feat: new engine"),
        ("ddd4", "2024-01-12T11:45:00", "src/core/engine.py", 8, 1, "chore: tidy imports"),
    ]
    for index, (sha, timestamp, changed, added, removed, message) in enumerate(commits):
        for test_path, test_name in [
            ("tests/test_api.py", "test_views_ok"),
            ("tests/test_engine.py", "test_engine_runs"),
        ]:
            failed = int(test_path == "tests/test_api.py" and index >= 2)
            rows.append(
                {
                    "commit_sha": sha,
                    "timestamp": timestamp,
                    "changed_file_path": changed,
                    "lines_added": added,
                    "lines_removed": removed,
                    "commit_message": message,
                    "test_id": f"{test_path}::{test_name}",
                    "test_path": test_path,
                    "duration": 1.5 + index * 0.1,
                    "label_failed": failed,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fitted artefacts
# --------------------------------------------------------------------------


def _artifacts_present(config: Dict[str, Any]) -> bool:
    """Whether every fitted artefact named in the config exists on disk."""
    return all(
        resolve_path(config["artifacts"][key]).is_file()
        for key in ("preprocessor_path", "model_path", "calibrator_path", "conformal_path")
    )


@pytest.fixture(scope="session")
def pipeline(config):
    """The shipped pipeline, or a skip when it has not been fitted yet."""
    if not _artifacts_present(config):
        pytest.skip("fitted artefacts absent; run the training pipeline first")
    from models.pipeline import SelectionPipeline

    return SelectionPipeline.load(config)


@pytest.fixture
def api_client(config):
    """A Flask test client for the serving layer."""
    from api.server import create_app

    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def candidate_tests() -> List[Dict[str, Any]]:
    """A realistic ``/predict`` payload: one change, several candidate tests."""
    return [
        {
            "test_id": f"tests/test_module_{index}.py::test_case",
            "test_path": f"tests/test_module_{index}.py",
            "changed_file_path": "src/module_0.py",
            "lines_added": 30,
            "lines_removed": 4,
            "execution_count": 120 + index,
            "failure_rate_lifetime": 0.02 * index,
            "duration_mean": 0.8,
        }
        for index in range(12)
    ]
