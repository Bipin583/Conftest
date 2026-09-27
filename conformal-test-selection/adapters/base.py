"""Framework adapter contract and registry.

A test-framework adapter answers three questions about a repository checkout:
*is this my framework?* (:meth:`matches`), *what tests exist?*
(:meth:`discover`), and *what am I?* (the ``framework``/``language`` attributes).
That is the seam a second ecosystem would plug into. Today the registry holds a
single adapter -- pytest -- and :func:`detect_adapter` therefore only ever
returns pytest or ``None``; the contract exists so that stays true by
construction rather than by scattered ``if framework == "pytest"`` checks.

Discovery returns :class:`~adapters.schema.NormalizedTestRecord` objects, so the
rest of the system never sees a framework-specific shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from adapters.schema import NormalizedTestRecord


class AdapterError(RuntimeError):
    """Raised when no adapter matches, or a requested framework is unknown."""


class TestFrameworkAdapter(ABC):
    """Base class every framework adapter implements.

    Attributes:
        framework: Framework name, e.g. ``"pytest"``. Must be in
            :data:`~adapters.schema.KNOWN_FRAMEWORKS`.
        language: Implementation language, e.g. ``"python"``.
    """

    framework: str = ""
    language: str = ""

    @abstractmethod
    def matches(self, repo_root: str) -> bool:
        """Whether this adapter's framework is present in ``repo_root``."""

    @abstractmethod
    def discover(self, repo_root: str) -> List[NormalizedTestRecord]:
        """Enumerate the framework's tests as normalized records."""


_REGISTRY: Dict[str, TestFrameworkAdapter] = {}


def register(adapter: TestFrameworkAdapter) -> TestFrameworkAdapter:
    """Register an adapter under its ``framework`` name.

    Args:
        adapter: The adapter instance to register.

    Returns:
        The same adapter, so this can wrap a construction expression.

    Raises:
        AdapterError: If the adapter declares no framework name.
    """
    if not adapter.framework:
        raise AdapterError(f"{type(adapter).__name__} declares no framework name.")
    _REGISTRY[adapter.framework] = adapter
    return adapter


def available_frameworks() -> List[str]:
    """Return the registered framework names, sorted."""
    return sorted(_REGISTRY)


def get_adapter(framework: str) -> TestFrameworkAdapter:
    """Return the adapter for ``framework``.

    Args:
        framework: A registered framework name.

    Returns:
        The matching adapter.

    Raises:
        AdapterError: If no adapter is registered for that name.
    """
    try:
        return _REGISTRY[framework]
    except KeyError as exc:
        raise AdapterError(
            f"No adapter for framework {framework!r}; known: {available_frameworks()}."
        ) from exc


def detect_adapter(repo_root: str) -> Optional[TestFrameworkAdapter]:
    """Return the first registered adapter whose framework matches ``repo_root``.

    Args:
        repo_root: Path to the checked-out repository.

    Returns:
        The matching adapter, or ``None`` when no registered framework is
        detected -- the signal for the caller to abstain rather than guess.
    """
    for name in available_frameworks():
        adapter = _REGISTRY[name]
        try:
            if adapter.matches(repo_root):
                return adapter
        except Exception:  # noqa: BLE001 - detection must never raise
            continue
    return None
