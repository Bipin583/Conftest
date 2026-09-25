"""Shared infrastructure for the conformal test-selection pipeline.

This module centralises the three concerns every other module needs: reading
``config.yaml``, configuring logging, and persisting/loading joblib artefacts.
Keeping them here means no pipeline stage hard-codes a path or a
hyper-parameter.

Typical use::

    from common import load_config, get_logger, save_artifact

    cfg = load_config()
    log = get_logger(__name__)
    save_artifact(model, cfg["artifacts"]["model_path"])
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import yaml

#: Repository root - the directory containing ``config.yaml``.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Default configuration file name, overridable via ``$CTS_CONFIG``.
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_LOGGING_CONFIGURED = False


class ConfigError(RuntimeError):
    """Raised when ``config.yaml`` is missing, unreadable, or malformed."""


class ArtifactError(RuntimeError):
    """Raised when a model artefact cannot be written or read back."""


def load_config(path: Optional[str | Path] = None, *, refresh: bool = False) -> Dict[str, Any]:
    """Load and cache the YAML configuration.

    Args:
        path: Explicit path to a configuration file. Defaults to the value of
            the ``CTS_CONFIG`` environment variable, else ``config.yaml`` beside
            this module.
        refresh: When ``True``, bypass the cache and re-read from disk. Used by
            the test suite to load fixture configurations.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        ConfigError: If the file is absent, is not valid YAML, or does not
            parse to a mapping.
    """
    resolved = Path(path or os.environ.get("CTS_CONFIG") or DEFAULT_CONFIG_PATH)
    key = str(resolved)

    if not refresh and key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]

    if not resolved.is_file():
        raise ConfigError(
            f"Configuration file not found: {resolved}. "
            "Run commands from the project root or set $CTS_CONFIG."
        )

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {resolved} as YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {resolved}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"{resolved} must contain a YAML mapping, got {type(parsed).__name__}.")

    _CONFIG_CACHE[key] = parsed
    return parsed


def setup_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Configure root logging once per process.

    Repeated calls are no-ops, so every entry point may call this defensively
    without stacking duplicate handlers onto the root logger.

    Args:
        level: Logging level name. Falls back to ``logging.level`` in the
            configuration, then to ``INFO``.
        fmt: ``logging`` format string. Falls back to ``logging.format``.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        if level:
            logging.getLogger().setLevel(level.upper())
        return

    try:
        log_cfg = load_config().get("logging", {})
    except ConfigError:
        log_cfg = {}

    logging.basicConfig(
        level=(level or log_cfg.get("level", "INFO")).upper(),
        format=fmt or log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        stream=sys.stderr,
    )
    # XGBoost and matplotlib are chatty at DEBUG; keep them at WARNING.
    for noisy in ("matplotlib", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    setup_logging()
    return logging.getLogger(name)


def set_seed(seed: Optional[int] = None) -> int:
    """Seed Python and NumPy RNGs for reproducible runs.

    Args:
        seed: Seed value. Defaults to ``project.random_seed`` from the config.

    Returns:
        The seed that was applied.
    """
    if seed is None:
        seed = int(load_config().get("project", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def resolve_path(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root.

    Absolute paths are returned unchanged; relative ones are anchored to
    :data:`PROJECT_ROOT` so commands behave identically regardless of the
    caller's working directory.

    Args:
        path: Absolute or project-relative path.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def save_artifact(obj: Any, path: str | Path, *, compress: int = 3) -> Path:
    """Persist an object with joblib, creating parent directories as needed.

    Args:
        obj: Any joblib-serialisable object.
        path: Destination path, resolved via :func:`resolve_path`.
        compress: joblib compression level, 0-9.

    Returns:
        The absolute path written.

    Raises:
        ArtifactError: If serialisation fails.
    """
    target = resolve_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, target, compress=compress)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"Failed to write artefact to {target}: {exc}") from exc
    return target


def load_artifact(path: str | Path) -> Any:
    """Load a joblib artefact written by :func:`save_artifact`.

    Args:
        path: Artefact path, resolved via :func:`resolve_path`.

    Returns:
        The deserialised object.

    Raises:
        ArtifactError: If the file is missing or cannot be deserialised.
    """
    source = resolve_path(path)
    if not source.is_file():
        raise ArtifactError(
            f"Artefact not found: {source}. Run the pipeline stage that produces it first."
        )
    try:
        return joblib.load(source)
    except Exception as exc:  # joblib surfaces many unpickling error types
        raise ArtifactError(f"Failed to read artefact {source}: {exc}") from exc
