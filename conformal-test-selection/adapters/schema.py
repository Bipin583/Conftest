"""Versioned, framework-neutral test-record schema.

The offline corpus and the live extractor both emit ad-hoc dictionaries whose
keys grew organically. Before this project can honestly claim a path to more than
one test framework, there has to be a *single, versioned* description of what a
"test record" is, independent of pytest node-id spelling or JUnit XML quirks.
This module is that description. It is deliberately small and additive: nothing
in the existing pipeline is forced through it yet, so adding it changes no
behaviour, but every new adapter (starting with pytest) produces records that
validate against it.

**Scope honesty.** Only Python/pytest is implemented today (see
``adapters.pytest_adapter``). The schema carries ``framework``/``language`` fields
so a second ecosystem can be added without changing the contract, not because a
second ecosystem exists.

The canonical ``test_id`` is framework-namespaced
(``pytest::tests/test_api.py::test_ok``) so identifiers from different frameworks
can never collide once more than one exists. The framework-native id the model
was trained on (``tests/test_api.py::test_ok``) is preserved separately as
``raw_node_id`` so normalisation never silently changes what is scored.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

#: Bumped only on a breaking change to the field set below. Consumers should
#: refuse a record whose ``schema_version`` major component they do not know.
SCHEMA_VERSION = "1.0.0"

#: Frameworks the schema is allowed to describe. A value outside this set is a
#: validation error rather than a silent pass, so an unknown framework routes to
#: abstention downstream instead of being scored as if it were pytest.
KNOWN_FRAMEWORKS = ("pytest",)
KNOWN_LANGUAGES = ("python",)

_ID_SEPARATOR = "::"
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class NormalizedTestRecord:
    """One test case, described independently of its framework.

    Attributes:
        test_id: Canonical, framework-namespaced identifier
            (``framework::path[::class]::function``). Unique across frameworks.
        test_path: Repository-relative POSIX path to the file defining the test.
        test_function: Test function or method name; ``""`` when unknown.
        framework: Producing framework, one of :data:`KNOWN_FRAMEWORKS`.
        language: Implementation language, one of :data:`KNOWN_LANGUAGES`.
        raw_node_id: The framework-native identifier fed to the model, preserved
            verbatim so normalisation never changes what is scored.
        class_name: Enclosing test class, when the framework has one.
        schema_version: The :data:`SCHEMA_VERSION` this record was built against.
    """

    test_id: str
    test_path: str
    test_function: str
    framework: str
    language: str
    raw_node_id: str
    class_name: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view."""
        return asdict(self)


def normalize_test_id(
    framework: str,
    test_path: str,
    test_function: Optional[str] = None,
    class_name: Optional[str] = None,
) -> str:
    """Build the canonical, framework-namespaced identifier for a test.

    Args:
        framework: Producing framework, e.g. ``"pytest"``.
        test_path: Repository-relative path to the test file.
        test_function: Test function or method name, if known.
        class_name: Enclosing class, if the framework groups tests in classes.

    Returns:
        ``framework::path[::class]::function`` with back-slashes normalised to
        forward slashes and surrounding whitespace stripped from each segment.
    """
    path = _WHITESPACE_RE.sub("", str(test_path).replace("\\", "/").strip())
    parts: List[str] = [str(framework).strip(), path]
    if class_name:
        parts.append(str(class_name).strip())
    if test_function:
        parts.append(str(test_function).strip())
    return _ID_SEPARATOR.join(p for p in parts if p)


def validate_record(record: NormalizedTestRecord) -> List[str]:
    """Return a list of human-readable reasons the record is invalid.

    An empty list means the record conforms to :data:`SCHEMA_VERSION`.

    Args:
        record: The record to check.

    Returns:
        Zero or more problem descriptions.
    """
    problems: List[str] = []
    if not record.test_id or _ID_SEPARATOR not in record.test_id:
        problems.append("test_id must be a non-empty, framework-namespaced identifier")
    if not record.test_path:
        problems.append("test_path is required")
    if not record.raw_node_id:
        problems.append("raw_node_id is required (the framework-native id fed to the model)")
    if record.framework not in KNOWN_FRAMEWORKS:
        problems.append(f"framework {record.framework!r} is not one of {KNOWN_FRAMEWORKS}")
    if record.language not in KNOWN_LANGUAGES:
        problems.append(f"language {record.language!r} is not one of {KNOWN_LANGUAGES}")
    major = str(record.schema_version).split(".", 1)[0]
    if major != SCHEMA_VERSION.split(".", 1)[0]:
        problems.append(
            f"schema_version {record.schema_version!r} has a different major than {SCHEMA_VERSION!r}"
        )
    return problems


def is_valid(record: NormalizedTestRecord) -> bool:
    """Whether ``record`` conforms to the current schema.

    Args:
        record: The record to check.

    Returns:
        ``True`` if :func:`validate_record` finds no problems.
    """
    return not validate_record(record)
