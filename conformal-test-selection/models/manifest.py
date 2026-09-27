"""A reproducibility manifest for a trained selection pipeline.

A conformal guarantee is only meaningful next to the exact ingredients that
produced it: which code revision, which library versions, which random seed,
which calibration method, and which threshold over how many calibration
failures. This module gathers those into ``models/manifest.json`` so a reviewer
(or a future rerun) can tell whether two results are comparable, and so a claim
in the README can be traced back to the artefacts that back it.

The manifest is *descriptive*, not a gate: it records what was, and never
changes an artefact. It is written by ``python cli.py manifest`` after training,
calibrating and fitting the conformal threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ArtifactError, get_logger, load_artifact, load_config, resolve_path  # noqa: E402

LOGGER = get_logger(__name__)

#: Packages whose versions materially affect a result and belong in the manifest.
_TRACKED_PACKAGES = ("xgboost", "scikit-learn", "numpy", "pandas", "scipy")


class ManifestError(RuntimeError):
    """Raised when the artefacts a manifest describes are missing."""


def _git_revision() -> Optional[str]:
    """Return the current commit hash, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(resolve_path(".")),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - a missing git binary must not fail the manifest
        return None


def _package_versions() -> Dict[str, str]:
    """Resolved versions of the tracked packages, ``"absent"`` if not installed."""
    versions: Dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def build_manifest(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the reproducibility manifest from the fitted artefacts.

    Args:
        config: Parsed configuration. Defaults to ``config.yaml``.

    Returns:
        The manifest dictionary.

    Raises:
        ManifestError: If the model, calibrator or conformal artefact is missing.
    """
    from adapters.schema import SCHEMA_VERSION

    cfg = config or load_config()
    artefacts = cfg["artifacts"]

    try:
        model_bundle = load_artifact(artefacts["model_path"])
        calibrator_bundle = load_artifact(artefacts["calibrator_path"])
        conformal_bundle = load_artifact(artefacts["conformal_path"])
    except ArtifactError as exc:
        raise ManifestError(
            f"{exc} Run 'python cli.py train', 'calibrate' and 'conformal' first."
        ) from exc

    selector = conformal_bundle["selector"]
    feature_columns: List[str] = list(model_bundle.get("feature_columns", []))
    calibrator = calibrator_bundle["calibrator"]

    return {
        "schema_version": SCHEMA_VERSION,
        "git_revision": _git_revision(),
        "random_seed": int(cfg["project"].get("random_seed", 0)),
        "packages": _package_versions(),
        "n_features": len(feature_columns),
        "calibration": {
            "method": getattr(calibrator, "method", None),
            "ece_bins": int(cfg["calibration"].get("ece_bins", 15)),
            "ece_threshold": float(cfg["calibration"].get("ece_threshold", 0.08)),
        },
        "conformal": {
            "guarantee": selector.guarantee,
            "target_coverage": selector.coverage,
            "confidence": selector.confidence,
            "certified_coverage": selector.certified_coverage,
            "probability_floor": round(float(selector.probability_floor), 6),
            "class_conditional": selector.class_conditional,
            "n_calibration": selector.n_calibration,
        },
    }


def write_manifest(config: Optional[Dict[str, Any]] = None, path: Optional[Any] = None) -> Path:
    """Build the manifest and write it as JSON.

    Args:
        config: Parsed configuration. Defaults to ``config.yaml``.
        path: Output path. Defaults to ``artifacts.manifest_path``.

    Returns:
        The resolved path written.

    Raises:
        ManifestError: If the artefacts are missing.
    """
    cfg = config or load_config()
    manifest = build_manifest(cfg)
    target = resolve_path(path or cfg["artifacts"].get("manifest_path", "models/manifest.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", target)
    return target


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m models.manifest``.

    Returns:
        ``0`` on success, ``1`` if the artefacts are missing.
    """
    parser = argparse.ArgumentParser(description="Write the pipeline reproducibility manifest.")
    parser.add_argument("--path", default=None, help="Destination manifest path.")
    args = parser.parse_args(argv)
    try:
        write_manifest(path=args.path)
    except ManifestError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
