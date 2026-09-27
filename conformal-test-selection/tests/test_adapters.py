"""The normalized schema and the pytest adapter.

These tests need no fitted artefacts: the schema is pure data and the adapter
discovers tests from a throwaway repository built in ``tmp_path``. They pin the
two properties the rest of the pipeline will rely on -- that a normalized id is
framework-namespaced while the model still sees the pytest node id, and that
detection returns pytest (never a wrong guess) for a Python repo and ``None``
for one with no tests.
"""

from __future__ import annotations

import pytest

from adapters import (
    SCHEMA_VERSION,
    NormalizedTestRecord,
    available_frameworks,
    detect_adapter,
    get_adapter,
    is_valid,
    normalize_test_id,
    validate_record,
)
from adapters.base import AdapterError


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_normalize_test_id_is_framework_namespaced():
    """A canonical id leads with the framework so ids cannot collide across them."""
    node = normalize_test_id("pytest", "tests/test_api.py", "test_ok")
    assert node == "pytest::tests/test_api.py::test_ok"


def test_normalize_test_id_includes_the_class_when_present():
    """Class-based tests keep the class segment between path and method."""
    node = normalize_test_id("pytest", "tests/test_api.py", "test_ok", class_name="TestApi")
    assert node == "pytest::tests/test_api.py::TestApi::test_ok"


def test_normalize_test_id_normalises_backslashes():
    """A Windows-style path must produce the same id as a POSIX one."""
    assert normalize_test_id("pytest", "tests\\test_api.py", "t") == "pytest::tests/test_api.py::t"


def test_a_well_formed_record_validates():
    """The happy path: every required field present and framework known."""
    record = NormalizedTestRecord(
        test_id="pytest::tests/test_api.py::test_ok",
        test_path="tests/test_api.py",
        test_function="test_ok",
        framework="pytest",
        language="python",
        raw_node_id="tests/test_api.py::test_ok",
    )
    assert is_valid(record)
    assert record.schema_version == SCHEMA_VERSION


def test_an_unknown_framework_is_a_validation_error():
    """An unknown framework must fail validation so it routes to abstention."""
    record = NormalizedTestRecord(
        test_id="jest::a.test.js::ok",
        test_path="a.test.js",
        test_function="ok",
        framework="jest",
        language="javascript",
        raw_node_id="a.test.js::ok",
    )
    problems = validate_record(record)
    assert any("framework" in p for p in problems)
    assert not is_valid(record)


def test_a_missing_raw_node_id_is_rejected():
    """Without the framework-native id there is nothing to score."""
    record = NormalizedTestRecord(
        test_id="pytest::tests/test_api.py::test_ok",
        test_path="tests/test_api.py",
        test_function="test_ok",
        framework="pytest",
        language="python",
        raw_node_id="",
    )
    assert any("raw_node_id" in p for p in validate_record(record))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_pytest_is_the_only_registered_framework():
    """Honesty check: the system supports exactly one ecosystem today."""
    assert available_frameworks() == ["pytest"]


def test_get_adapter_rejects_an_unknown_framework():
    """Asking for a framework we do not have must fail loudly, not fall back."""
    with pytest.raises(AdapterError):
        get_adapter("junit5")


# --------------------------------------------------------------------------
# pytest adapter discovery
# --------------------------------------------------------------------------


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_detect_adapter_returns_pytest_for_a_python_repo(tmp_path):
    """A repo with a pytest file is detected as pytest."""
    _write(tmp_path / "tests" / "test_sample.py", "def test_a():\n    assert True\n")
    adapter = detect_adapter(str(tmp_path))
    assert adapter is not None and adapter.framework == "pytest"


def test_detect_adapter_returns_none_without_tests(tmp_path):
    """No test files means no guess -- the caller should abstain, not assume pytest."""
    _write(tmp_path / "src" / "app.py", "x = 1\n")
    assert detect_adapter(str(tmp_path)) is None


def test_discover_normalises_function_and_class_tests(tmp_path):
    """Both a bare test function and a Test-class method are discovered and namespaced."""
    _write(
        tmp_path / "tests" / "test_sample.py",
        "def test_free():\n    assert True\n\n"
        "class TestGroup:\n    def test_method(self):\n        assert True\n",
    )
    adapter = get_adapter("pytest")
    records = adapter.discover(str(tmp_path))

    by_function = {r.test_function: r for r in records}
    assert {"test_free", "test_method"} <= set(by_function)
    for record in records:
        assert is_valid(record)
        assert record.test_id.startswith("pytest::")
        assert "::" in record.raw_node_id  # the pytest node id the model was trained on

    assert by_function["test_method"].class_name == "TestGroup"
