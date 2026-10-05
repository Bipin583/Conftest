"""Framework adapters: framework-neutral test discovery for the pipeline.

Importing this package registers every shipped adapter (currently only pytest),
so :func:`adapters.base.detect_adapter` and :func:`adapters.base.get_adapter`
work without the caller importing each adapter module by hand.

Public surface::

    from adapters import (
        NormalizedTestRecord, normalize_test_id, validate_record, is_valid,
        SCHEMA_VERSION, detect_adapter, get_adapter, available_frameworks,
    )
"""

from __future__ import annotations

from adapters.base import (
    AdapterError,
    TestFrameworkAdapter,
    available_frameworks,
    detect_adapter,
    get_adapter,
    register,
)
from adapters.schema import (
    KNOWN_FRAMEWORKS,
    KNOWN_LANGUAGES,
    SCHEMA_VERSION,
    NormalizedTestRecord,
    is_valid,
    normalize_test_id,
    validate_record,
)

# Registers the pytest adapter as a side effect of import.
from adapters import pytest_adapter as _pytest_adapter  # noqa: E402,F401

__all__ = [
    "AdapterError",
    "TestFrameworkAdapter",
    "available_frameworks",
    "detect_adapter",
    "get_adapter",
    "register",
    "KNOWN_FRAMEWORKS",
    "KNOWN_LANGUAGES",
    "SCHEMA_VERSION",
    "NormalizedTestRecord",
    "is_valid",
    "normalize_test_id",
    "validate_record",
]
