"""
Unit tests for ConfTest AST mutation operators.

These tests protect the core correctness guarantees of the ground-truth
pipeline: every mutant must compile, differ from the original by exactly
one line, and never touch semantically inert code such as docstrings or
type annotations.
"""

import ast

import pytest

from conftest.groundtruth.mutators import (
    STRING_MUTANT_SENTINEL,
    Mutant,
    generate_mutants,
)


def _operators(mutants):
    """Collect the set of operator labels produced."""
    return {m.operator for m in mutants}


def _find(mutants, operator_prefix):
    """Return mutants whose operator label starts with the given prefix."""
    return [m for m in mutants if m.operator.startswith(operator_prefix)]


# --------------------------------------------------------------------------
# Core invariants
# --------------------------------------------------------------------------

def test_every_mutant_compiles():
    source = (
        "def compute(a, b):\n"
        "    total = a + b\n"
        "    if total > 10 and a != 0:\n"
        "        return total * 2\n"
        "    return 0\n"
    )
    mutants = generate_mutants(source, "src/compute.py")

    assert mutants, "expected at least one mutant"
    for mutant in mutants:
        # Must not raise.
        ast.parse(mutant.mutated_source, filename=mutant.file_path)


def test_every_mutant_changes_exactly_one_line():
    source = (
        "def compute(a, b):\n"
        "    total = a + b\n"
        "    if total > 10:\n"
        "        return total * 2\n"
        "    return 0\n"
    )
    original_lines = source.split("\n")
    mutants = generate_mutants(source, "src/compute.py")

    assert mutants
    for mutant in mutants:
        mutated_lines = mutant.mutated_source.split("\n")
        assert len(mutated_lines) == len(original_lines), (
            f"{mutant.operator} changed the line count"
        )
        differing = [
            i for i, (a, b) in enumerate(zip(original_lines, mutated_lines)) if a != b
        ]
        assert len(differing) == 1, (
            f"{mutant.operator} changed {len(differing)} lines, expected 1"
        )
        # The recorded line number must match the line that actually changed.
        assert differing[0] + 1 == mutant.line


def test_mutants_actually_differ_from_original():
    source = "def f(a, b):\n    return a + b\n"
    mutants = generate_mutants(source, "src/f.py")

    assert mutants
    for mutant in mutants:
        assert mutant.mutated_source != source


def test_mutant_ids_are_unique_and_deterministic():
    source = (
        "def f(a, b):\n"
        "    if a < b:\n"
        "        return a + b\n"
        "    return a - b\n"
    )
    first = generate_mutants(source, "src/f.py")
    second = generate_mutants(source, "src/f.py")

    ids = [m.mutant_id for m in first]
    assert len(ids) == len(set(ids)), "mutant IDs must be unique"
    assert ids == [m.mutant_id for m in second], "generation must be deterministic"


def test_returns_empty_for_unparseable_source():
    assert generate_mutants("def broken(:\n", "src/bad.py") == []


def test_returns_empty_for_source_with_no_mutable_nodes():
    assert generate_mutants("import os\n", "src/plain.py") == []


# --------------------------------------------------------------------------
# Individual operators
# --------------------------------------------------------------------------

def test_arithmetic_operator_swap():
    mutants = generate_mutants("def f(a, b):\n    return a + b\n", "src/f.py")
    arith = _find(mutants, "arith_+_to_-")

    assert arith, f"expected an Add->Sub mutant, got {_operators(mutants)}"
    assert "(a - b)" in arith[0].mutated_source


def test_comparison_boundary_and_direction_swaps():
    mutants = generate_mutants("def f(a, b):\n    return a < b\n", "src/f.py")
    labels = _operators(mutants)

    # Off-by-one boundary fault and direction-flip fault.
    assert "cmp_<_to_<=" in labels
    assert "cmp_<_to_>" in labels


def test_equality_swap():
    mutants = generate_mutants("def f(a, b):\n    return a == b\n", "src/f.py")
    assert "cmp_===_to_!=" not in _operators(mutants)
    assert "cmp_==_to_!=" in _operators(mutants)


def test_boolean_connective_swap():
    mutants = generate_mutants("def f(a, b):\n    return a and b\n", "src/f.py")
    swapped = _find(mutants, "bool_and_to_or")

    assert swapped
    assert "or" in swapped[0].mutated_snippet


def test_negation_is_stripped():
    mutants = generate_mutants("def f(a):\n    return not a\n", "src/f.py")
    dropped = _find(mutants, "bool_drop_not")

    assert dropped
    assert "not" not in dropped[0].mutated_snippet


def test_boolean_constant_flip():
    mutants = generate_mutants("FLAG = True\n", "src/f.py")
    flipped = _find(mutants, "const_bool_")

    assert flipped
    assert "False" in flipped[0].mutated_source


def test_integer_constant_increment():
    mutants = generate_mutants("LIMIT = 10\n", "src/f.py")
    incremented = _find(mutants, "const_int_increment")

    assert incremented
    assert "LIMIT = 11" in incremented[0].mutated_source


def test_bool_is_not_treated_as_integer():
    """bool subclasses int, so ordering of the isinstance checks matters."""
    mutants = generate_mutants("FLAG = True\n", "src/f.py")
    labels = _operators(mutants)

    assert "const_int_increment" not in labels
    assert any(label.startswith("const_bool_") for label in labels)


def test_non_empty_string_is_emptied():
    mutants = generate_mutants("NAME = 'alpha'\n", "src/f.py")
    emptied = _find(mutants, "const_str_empty")

    assert emptied
    assert "NAME = ''" in emptied[0].mutated_source


def test_empty_string_is_filled():
    mutants = generate_mutants("NAME = ''\n", "src/f.py")
    filled = _find(mutants, "const_str_fill")

    assert filled
    assert STRING_MUTANT_SENTINEL in filled[0].mutated_source


def test_return_value_is_voided():
    mutants = generate_mutants("def f(a):\n    return a * 2\n", "src/f.py")
    voided = _find(mutants, "return_to_None")

    assert voided
    assert "return None" in voided[0].mutated_source


def test_bare_return_is_not_mutated():
    mutants = generate_mutants("def f(a):\n    return\n", "src/f.py")
    assert not _find(mutants, "return_to_None")


def test_explicit_return_none_is_not_mutated():
    mutants = generate_mutants("def f(a):\n    return None\n", "src/f.py")
    assert not _find(mutants, "return_to_None")


def test_conditional_forced_to_constant():
    source = "def f(a):\n    if a > 0:\n        return 1\n    return 2\n"
    labels = _operators(generate_mutants(source, "src/f.py"))

    assert "cond_to_True" in labels
    assert "cond_to_False" in labels


# --------------------------------------------------------------------------
# Exclusions -- the safety-critical cases
# --------------------------------------------------------------------------

def test_while_condition_is_never_forced_true():
    """Forcing a loop condition to True guarantees an infinite loop, not a fault."""
    source = (
        "def f(n):\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        i = i + 1\n"
        "    return i\n"
    )
    mutants = generate_mutants(source, "src/f.py")

    for mutant in mutants:
        assert mutant.operator not in ("cond_to_True", "cond_to_False"), (
            "while-loop conditions must not be replaced by constants"
        )


def test_docstrings_are_not_mutated():
    source = (
        '"""Module docstring."""\n'
        "\n"
        "def f(a):\n"
        '    """Function docstring."""\n'
        "    return a\n"
    )
    mutants = generate_mutants(source, "src/f.py")

    for mutant in mutants:
        assert "docstring" not in mutant.original_snippet.lower(), (
            f"{mutant.operator} mutated a docstring"
        )


def test_type_annotations_are_not_mutated():
    source = "def f(a: int = 5) -> int:\n    return a\n"
    mutants = generate_mutants(source, "src/f.py")

    # The default value 5 lives in `defaults`, not the annotation, so it stays
    # mutable; the annotation itself must never be touched.
    for mutant in mutants:
        line = mutant.mutated_source.split("\n")[mutant.line - 1]
        assert "a: int" in line or "return" in line


def test_annotated_assignment_annotation_is_not_mutated():
    source = "from typing import List\nvalues: List[int] = []\n"
    mutants = generate_mutants(source, "src/f.py")

    for mutant in mutants:
        assert "List[int]" in mutant.mutated_source, (
            f"{mutant.operator} corrupted a type annotation"
        )


def test_fstring_internals_are_not_mutated():
    source = "def f(a, b):\n    return f'{a + b}'\n"
    mutants = generate_mutants(source, "src/f.py")

    for mutant in mutants:
        assert mutant.operator != "arith_+_to_-", (
            "f-string internals must be excluded from splicing"
        )


# --------------------------------------------------------------------------
# Precedence safety
# --------------------------------------------------------------------------

def test_precedence_is_preserved_under_splicing():
    """
    Parenthesizing the replacement must prevent precedence drift.

    In `(a + b) * c`, mutating the inner sum must not change evaluation order.
    """
    source = "def f(a, b, c):\n    return (a + b) * c\n"
    mutants = _find(generate_mutants(source, "src/f.py"), "arith_+_to_-")

    assert mutants
    namespace: dict = {}
    exec(compile(mutants[0].mutated_source, "src/f.py", "exec"), namespace)

    # (2 - 3) * 4 == -4, whereas a precedence bug would give 2 - (3 * 4) == -10.
    assert namespace["f"](2, 3, 4) == -4


def test_mutation_changes_runtime_behaviour():
    """A mutant must be observably different, or it cannot be killed by a test."""
    source = "def add(a, b):\n    return a + b\n"
    mutants = _find(generate_mutants(source, "src/add.py"), "arith_+_to_-")

    assert mutants
    namespace: dict = {}
    exec(compile(mutants[0].mutated_source, "src/add.py", "exec"), namespace)
    assert namespace["add"](5, 3) == 2


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_provenance_fields_are_populated():
    mutants = generate_mutants("def f(a, b):\n    return a + b\n", "src/f.py")

    assert mutants
    mutant = mutants[0]
    assert isinstance(mutant, Mutant)
    assert mutant.file_path == "src/f.py"
    assert mutant.line >= 1
    assert mutant.operator
    assert mutant.original_snippet
    assert mutant.mutated_snippet
    assert mutant.mutant_id.startswith("mut_")


def test_to_dict_excludes_full_source():
    mutants = generate_mutants("def f(a, b):\n    return a + b\n", "src/f.py")
    payload = mutants[0].to_dict()

    assert "mutated_source" not in payload
    assert payload["mutant_id"].startswith("mut_")


def test_mutant_is_immutable():
    mutants = generate_mutants("def f(a, b):\n    return a + b\n", "src/f.py")

    with pytest.raises(Exception):
        mutants[0].line = 99  # type: ignore[misc]


# --------------------------------------------------------------------------
# Realistic source
# --------------------------------------------------------------------------

def test_generates_broad_operator_coverage_on_realistic_module():
    source = (
        '"""Payment helpers."""\n'
        "\n"
        "TAX_RATE = 5\n"
        "\n"
        "def fee(amount, is_premium):\n"
        '    """Compute the fee."""\n'
        "    if amount <= 0:\n"
        "        return 0\n"
        "    base = amount * TAX_RATE\n"
        "    if is_premium and amount > 100:\n"
        "        base = base - 10\n"
        "    return base\n"
    )
    mutants = generate_mutants(source, "src/payment.py")
    labels = _operators(mutants)

    # At least four distinct operator families should fire on this module.
    families = {label.split("_")[0] for label in labels}
    assert len(families) >= 4, f"thin coverage: {labels}"

    for mutant in mutants:
        ast.parse(mutant.mutated_source, filename=mutant.file_path)
