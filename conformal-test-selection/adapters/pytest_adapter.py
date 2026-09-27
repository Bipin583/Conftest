"""pytest adapter: the one framework this project actually supports.

Discovery reuses :func:`data.live_extract.discover_tests` -- the AST-based
walker already trusted by the advisory job -- rather than re-implementing test
collection, so there is exactly one definition of "what pytest tests exist here"
and this adapter cannot drift from it. Its job is to wrap that output in the
framework-neutral :class:`~adapters.schema.NormalizedTestRecord`, preserving the
pytest node id (``path::func`` or ``path::Class::method``) as ``raw_node_id`` so
the model is still scored on exactly the identifier it was trained on.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from adapters.base import TestFrameworkAdapter, register
from adapters.schema import NormalizedTestRecord, normalize_test_id


def _split_node_id(node_id: str) -> Tuple[Optional[str], str]:
    """Return ``(class_name, function)`` parsed from a pytest node id.

    ``path::Class::method`` yields ``("Class", "method")``; ``path::func`` yields
    ``(None, "func")``; anything shorter yields ``(None, "")``.
    """
    parts = node_id.split("::")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    if len(parts) == 2:
        return None, parts[1]
    return None, ""


class PytestAdapter(TestFrameworkAdapter):
    """Discovers ``test_*.py`` / ``*_test.py`` tests in a Python repository."""

    framework = "pytest"
    language = "python"

    def matches(self, repo_root: str) -> bool:
        """True when the checkout contains at least one pytest test file.

        Walks the tree (skipping vendored/VCS/cache directories) and stops at the
        first file named by pytest convention, so detection stays cheap on large
        repositories.
        """
        for dirpath, _, filenames in os.walk(repo_root):
            parts = set(dirpath.replace("\\", "/").split("/"))
            if parts & {".git", ".venv", "venv", "node_modules", "__pycache__"}:
                continue
            for name in filenames:
                if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
                    return True
        return False

    def discover(self, repo_root: str) -> List[NormalizedTestRecord]:
        """Enumerate pytest tests as normalized records.

        Args:
            repo_root: Path to the checked-out repository.

        Returns:
            One :class:`NormalizedTestRecord` per discovered test function or
            method, with the pytest node id preserved as ``raw_node_id``.
        """
        # Imported lazily so importing the adapter package does not pull in the
        # feature-engineering stack (pandas et al.) until discovery is actually run.
        from data.live_extract import discover_tests

        records: List[NormalizedTestRecord] = []
        for found in discover_tests(repo_root):
            raw_node_id = found["test_id"]
            class_name, function = _split_node_id(raw_node_id)
            function = function or found.get("test_function", "")
            records.append(
                NormalizedTestRecord(
                    test_id=normalize_test_id(self.framework, found["test_path"], function, class_name),
                    test_path=found["test_path"],
                    test_function=function,
                    framework=self.framework,
                    language=self.language,
                    raw_node_id=raw_node_id,
                    class_name=class_name,
                )
            )
        return records


#: The single adapter this project ships. Registered on import.
PYTEST_ADAPTER = register(PytestAdapter())
