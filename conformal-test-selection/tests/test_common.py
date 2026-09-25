"""Configuration, path resolution and artefact I/O.

These are the seams every other module depends on, and the failure modes are
the quiet ones: a config read from the wrong directory, or an artefact that
silently reads back as ``None``. Each test here pins behaviour that a stack
trace elsewhere would only hint at.
"""

from __future__ import annotations

import numpy as np
import pytest

from common import (
    PROJECT_ROOT,
    ArtifactError,
    ConfigError,
    load_artifact,
    load_config,
    resolve_path,
    save_artifact,
    set_seed,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_config_exposes_every_section_the_pipeline_reads(config):
    """A missing section surfaces here rather than mid-training."""
    assert {
        "project",
        "model",
        "calibration",
        "conformal",
        "selection",
        "data",
        "artifacts",
        "targets",
    } <= set(config)


def test_conformal_settings_are_the_two_numbers_the_claim_quotes(config):
    """95% coverage at 90% confidence, under the PAC guarantee.

    The project's stated requirement has two numbers in it, which is a
    training-conditional statement. Shipping ``marginal`` here would make the
    README's phrasing wrong even though the code still ran.
    """
    conformal = config["conformal"]
    assert conformal["coverage"] == 0.95
    assert conformal["confidence"] == 0.90
    assert conformal["guarantee"] == "pac"


def test_targets_are_the_acceptance_gates_not_measurements(config):
    """``targets`` drives the pass/fail column of the evaluation report."""
    targets = config["targets"]
    assert targets["recall"] == 0.92
    assert targets["selection_rate"] == 0.40
    assert targets["ece"] == 0.08
    assert targets["coverage"] == 0.95
    assert targets["cost_reduction"] == 0.60


def test_missing_config_names_the_file_it_wanted(tmp_path):
    """The error must be actionable, not a bare KeyError later on."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml", refresh=True)


def test_malformed_config_is_rejected(tmp_path):
    """A YAML scalar where a mapping belongs is caught at load time."""
    bad = tmp_path / "scalar.yaml"
    bad.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad, refresh=True)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def test_relative_paths_anchor_to_the_project_not_the_shell():
    """``cli.py`` must behave the same from any working directory.

    Config paths such as ``models/gbdt_model.pkl`` are project-relative. If
    they resolved against the caller's cwd, running the CLI from a parent
    directory would write artefacts into the wrong tree.
    """
    assert resolve_path("models/gbdt_model.pkl") == PROJECT_ROOT / "models" / "gbdt_model.pkl"


def test_absolute_paths_pass_through_untouched(tmp_path):
    """An explicit absolute path is an override, not a suggestion."""
    assert resolve_path(tmp_path / "x.pkl") == tmp_path / "x.pkl"


# --------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------


def test_artifacts_round_trip_with_their_contents_intact(tmp_path):
    """What comes back must equal what went in, arrays included."""
    payload = {"array": np.arange(10), "columns": ["a", "b"], "threshold": 0.42}
    written = save_artifact(payload, tmp_path / "nested" / "artefact.pkl")

    assert written.is_file(), "parent directories are created on demand"
    restored = load_artifact(written)
    assert restored["columns"] == payload["columns"]
    assert restored["threshold"] == payload["threshold"]
    assert np.array_equal(restored["array"], payload["array"])


def test_loading_a_missing_artefact_says_which_stage_produces_it(tmp_path):
    """The message is the user's next instruction, so it must be specific."""
    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path / "never_written.pkl")


def test_loading_a_corrupt_artefact_fails_cleanly(tmp_path):
    """Truncated pickles raise ArtifactError, not an opaque unpickling error."""
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle at all")
    with pytest.raises(ArtifactError):
        load_artifact(corrupt)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_seeding_makes_numpy_draws_repeatable(config):
    """The reported numbers must be reproducible from the committed seed."""
    seed = set_seed()
    assert seed == config["project"]["random_seed"]

    first = np.random.rand(5)
    set_seed()
    assert np.array_equal(first, np.random.rand(5))
