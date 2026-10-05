"""The reproducibility manifest.

The pure helpers (package versions, git revision) are checked directly; the
full :func:`build_manifest` needs the fitted artefacts, so it skips rather than
fails on a fresh clone -- the same policy as the pipeline tests.
"""

from __future__ import annotations

import pytest

from models.manifest import (
    ManifestError,
    _git_revision,
    _package_versions,
    build_manifest,
)


def test_package_versions_cover_every_tracked_package():
    """A tracked package is always reported -- as a version or as 'absent'."""
    versions = _package_versions()
    for name in ("xgboost", "scikit-learn", "numpy", "pandas", "scipy"):
        assert name in versions
        assert isinstance(versions[name], str) and versions[name]


def test_git_revision_is_a_hash_or_none():
    """Inside a checkout this is a 40-char hash; outside one it is None, not a crash."""
    revision = _git_revision()
    assert revision is None or (len(revision) == 40 and all(c in "0123456789abcdef" for c in revision))


def test_manifest_records_the_guarantee_and_the_seed(config):
    """The manifest must tie a result to the guarantee and seed that produced it."""
    try:
        manifest = build_manifest(config)
    except ManifestError:
        pytest.skip("artefacts absent; train/calibrate/conformal first")

    assert manifest["schema_version"]
    assert manifest["random_seed"] == config["project"]["random_seed"]
    assert manifest["conformal"]["guarantee"] == config["conformal"]["guarantee"]
    assert manifest["conformal"]["target_coverage"] == config["conformal"]["coverage"]
    assert manifest["conformal"]["n_calibration"] > 0
    assert manifest["n_features"] > 0
    assert manifest["calibration"]["method"] in {"platt", "isotonic"}
