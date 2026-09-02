"""
ConfTest AST Mutation Operators.

Generates single-node source mutations used as *real* injected faults.
Each mutant is a genuine, compilable code change whose effect on the test
suite is later MEASURED by executing pytest -- never simulated.

Design notes
------------
1. Exactly ONE AST node is mutated per mutant, so any resulting test
   failure is attributable to one file and one line.
2. Mutations are applied by surgical text splicing over the original
   source, not by ``ast.unparse`` of the whole module. This keeps the
   rest of the file byte-identical, so the resulting diff is a true
   one-line change (which the diff/churn features depend on).
3. Every mutant is re-parsed before being emitted. Anything that does not
   compile is discarded.
4. ``ast.col_offset`` is a UTF-8 *byte* offset, so splicing is performed on
   the encoded line to stay correct for non-ASCII source.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from conftest.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Operator tables
# --------------------------------------------------------------------------

# Arithmetic binary operator swaps.
ARITHMETIC_SWAPS: Dict[type, List[type]] = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.Div],
    ast.Div: [ast.Mult],
    ast.FloorDiv: [ast.Div],
    ast.Mod: [ast.Mult],
    ast.Pow: [ast.Mult],
}

# Comparison operator swaps: boundary (off-by-one) and direction faults.
COMPARISON_SWAPS: Dict[type, List[type]] = {
    ast.Lt: [ast.LtE, ast.Gt],
    ast.LtE: [ast.Lt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Lt],
    ast.GtE: [ast.Gt, ast.LtE],
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
    ast.In: [ast.NotIn],
    ast.NotIn: [ast.In],
    ast.Is: [ast.IsNot],
    ast.IsNot: [ast.Is],
}

# Boolean connective swaps.
BOOLEAN_SWAPS: Dict[type, List[type]] = {
    ast.And: [ast.Or],
    ast.Or: [ast.And],
}

# Human-readable operator symbols, used for building the operator label.
_OP_SYMBOL: Dict[type, str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in",
    ast.Is: "is", ast.IsNot: "is not",
    ast.And: "and", ast.Or: "or",
}

# Marker string injected when mutating an empty string literal.
STRING_MUTANT_SENTINEL = "conftest_mutant"


@dataclass(frozen=True)
class Mutant:
    """A single compilable source mutation with full provenance."""

    mutant_id: str
    file_path: str
    line: int
    operator: str
    original_snippet: str
    mutated_snippet: str
    mutated_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance fields, excluding the full mutated source."""
        return {
            "mutant_id": self.mutant_id,
            "file_path": self.file_path,
            "line": self.line,
            "operator": self.operator,
            "original_snippet": self.original_snippet,
            "mutated_snippet": self.mutated_snippet,
        }


# --------------------------------------------------------------------------
# Node exclusion analysis
# --------------------------------------------------------------------------

def _collect_docstring_nodes(tree: ast.AST) -> Set[int]:
    """Identify Constant nodes that serve as module/class/function docstrings."""
    docstrings: Set[int] = set()
    containers = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

    for node in ast.walk(tree):
        if not isinstance(node, containers):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                docstrings.add(id(first.value))

    return docstrings


def _collect_annotation_nodes(tree: ast.AST) -> Set[int]:
    """Identify nodes inside type annotations, which carry no runtime behaviour."""
    excluded: Set[int] = set()

    def mark_subtree(root: Optional[ast.AST]) -> None:
        if root is None:
            return
        for descendant in ast.walk(root):
            excluded.add(id(descendant))

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            mark_subtree(node.annotation)
        elif isinstance(node, ast.arg):
            mark_subtree(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark_subtree(node.returns)

    return excluded


def _collect_fstring_nodes(tree: ast.AST) -> Set[int]:
    """Identify nodes inside f-strings, where byte-offset splicing is unreliable."""
    excluded: Set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for descendant in ast.walk(node):
                excluded.add(id(descendant))

    return excluded


def _collect_loop_condition_nodes(tree: ast.AST) -> Set[int]:
    """
    Identify ``while`` loop test expressions.

    Forcing a loop condition to a constant truth value produces a guaranteed
    infinite loop rather than a realistic fault, so such nodes are never
    replaced wholesale by the conditional operator.
    """
    excluded: Set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            excluded.add(id(node.test))

    return excluded


# --------------------------------------------------------------------------
# Source splicing
# --------------------------------------------------------------------------

def _splice_node(
    source_lines: List[str],
    node: ast.AST,
    replacement: str,
) -> Optional[Tuple[str, str, int]]:
    """
    Replace the exact source span of ``node`` with ``replacement``.

    Returns:
        Tuple of (mutated_source, original_snippet, line_number), or None if
        the node spans multiple lines or carries incomplete position data.
    """
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)

    if lineno is None or end_lineno is None or col is None or end_col is None:
        return None

    # Restrict to single-line spans so every mutant is a true one-line diff.
    if lineno != end_lineno:
        return None
    if not (1 <= lineno <= len(source_lines)):
        return None

    # col_offset is a UTF-8 byte offset, so slice the encoded line.
    line_bytes = source_lines[lineno - 1].encode("utf-8")
    if end_col > len(line_bytes) or col > end_col:
        return None

    try:
        prefix = line_bytes[:col].decode("utf-8")
        original_snippet = line_bytes[col:end_col].decode("utf-8")
        suffix = line_bytes[end_col:].decode("utf-8")
    except UnicodeDecodeError:
        return None

    mutated_lines = list(source_lines)
    mutated_lines[lineno - 1] = prefix + replacement + suffix

    return "\n".join(mutated_lines), original_snippet, lineno


def _make_mutant_id(file_path: str, line: int, operator: str, snippet: str) -> str:
    """Build a stable, collision-resistant identifier for a mutant."""
    payload = f"{file_path}|{line}|{operator}|{snippet}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"mut_{digest}"


def _build_mutant(
    file_path: str,
    source_lines: List[str],
    node: ast.AST,
    replacement: str,
    operator: str,
) -> Optional[Mutant]:
    """Splice, validate, and package a single mutation attempt."""
    spliced = _splice_node(source_lines, node, replacement)
    if spliced is None:
        return None

    mutated_source, original_snippet, lineno = spliced

    # Reject no-op mutations.
    if original_snippet == replacement:
        return None

    # Reject anything that no longer compiles.
    try:
        ast.parse(mutated_source, filename=file_path)
    except (SyntaxError, ValueError):
        return None

    return Mutant(
        mutant_id=_make_mutant_id(file_path, lineno, operator, original_snippet),
        file_path=file_path,
        line=lineno,
        operator=operator,
        original_snippet=original_snippet,
        mutated_snippet=replacement,
        mutated_source=mutated_source,
    )


def _unparse_parenthesized(node: ast.AST) -> Optional[str]:
    """
    Unparse an expression node, wrapped in parentheses.

    Parenthesizing guarantees the spliced replacement cannot alter operator
    precedence relative to the surrounding expression.
    """
    try:
        return f"({ast.unparse(node)})"
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"Unparse failed for {type(node).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------

def _mutate_binops(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """Swap arithmetic binary operators (a + b -> a - b)."""
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or id(node) in excluded:
            continue

        for replacement_op in ARITHMETIC_SWAPS.get(type(node.op), []):
            original_op = node.op
            node.op = replacement_op()
            replacement = _unparse_parenthesized(node)
            node.op = original_op

            if replacement is None:
                continue

            label = (
                f"arith_{_OP_SYMBOL.get(type(original_op), '?')}"
                f"_to_{_OP_SYMBOL.get(replacement_op, '?')}"
            )
            mutant = _build_mutant(file_path, lines, node, replacement, label)
            if mutant is not None:
                mutants.append(mutant)

    return mutants


def _mutate_comparisons(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """Swap comparison operators (a < b -> a <= b), covering boundary faults."""
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or id(node) in excluded:
            continue

        for index, op in enumerate(node.ops):
            for replacement_op in COMPARISON_SWAPS.get(type(op), []):
                node.ops[index] = replacement_op()
                replacement = _unparse_parenthesized(node)
                node.ops[index] = op

                if replacement is None:
                    continue

                label = (
                    f"cmp_{_OP_SYMBOL.get(type(op), '?')}"
                    f"_to_{_OP_SYMBOL.get(replacement_op, '?')}"
                )
                mutant = _build_mutant(file_path, lines, node, replacement, label)
                if mutant is not None:
                    mutants.append(mutant)

    return mutants


def _mutate_boolops(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """Swap boolean connectives (and -> or) and strip logical negation."""
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if id(node) in excluded:
            continue

        if isinstance(node, ast.BoolOp):
            for replacement_op in BOOLEAN_SWAPS.get(type(node.op), []):
                original_op = node.op
                node.op = replacement_op()
                replacement = _unparse_parenthesized(node)
                node.op = original_op

                if replacement is None:
                    continue

                label = (
                    f"bool_{_OP_SYMBOL.get(type(original_op), '?')}"
                    f"_to_{_OP_SYMBOL.get(replacement_op, '?')}"
                )
                mutant = _build_mutant(file_path, lines, node, replacement, label)
                if mutant is not None:
                    mutants.append(mutant)

        # Remove a logical negation: `not x` -> `x`
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            replacement = _unparse_parenthesized(node.operand)
            if replacement is None:
                continue
            mutant = _build_mutant(file_path, lines, node, replacement, "bool_drop_not")
            if mutant is not None:
                mutants.append(mutant)

    return mutants


def _mutate_constants(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """Perturb literal constants (booleans, integers, strings)."""
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in excluded:
            continue

        value = node.value
        candidates: List[Tuple[str, str]] = []

        # bool must be checked before int, since bool subclasses int.
        if isinstance(value, bool):
            candidates.append((repr(not value), f"const_bool_{value}_to_{not value}"))
        elif isinstance(value, int):
            candidates.append((repr(value + 1), "const_int_increment"))
        elif isinstance(value, str):
            if value == "":
                candidates.append((repr(STRING_MUTANT_SENTINEL), "const_str_fill"))
            else:
                candidates.append(("''", "const_str_empty"))

        for replacement, label in candidates:
            mutant = _build_mutant(file_path, lines, node, replacement, label)
            if mutant is not None:
                mutants.append(mutant)

    return mutants


def _mutate_returns(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """Void a return value (`return x` -> `return None`), a contract violation."""
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or id(node) in excluded:
            continue

        # Skip bare `return` and `return None`, which are already void.
        if node.value is None:
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            continue

        mutant = _build_mutant(file_path, lines, node, "return None", "return_to_None")
        if mutant is not None:
            mutants.append(mutant)

    return mutants


def _mutate_conditionals(
    tree: ast.AST, file_path: str, lines: List[str], excluded: Set[int]
) -> List[Mutant]:
    """
    Force an ``if`` condition to a constant truth value, killing one branch.

    ``while`` conditions are excluded upstream to avoid guaranteed infinite loops.
    """
    mutants: List[Mutant] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if id(node.test) in excluded:
            continue

        for replacement, label in (("True", "cond_to_True"), ("False", "cond_to_False")):
            mutant = _build_mutant(file_path, lines, node.test, replacement, label)
            if mutant is not None:
                mutants.append(mutant)

    return mutants


# Registry of all enabled operators.
MUTATION_OPERATORS = (
    _mutate_binops,
    _mutate_comparisons,
    _mutate_boolops,
    _mutate_constants,
    _mutate_returns,
    _mutate_conditionals,
)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def generate_mutants(source: str, file_path: str) -> List[Mutant]:
    """
    Enumerate every valid single-node mutant of a Python source file.

    Args:
        source: Full text of the source file.
        file_path: Repository-relative path, recorded for provenance.

    Returns:
        Deterministically ordered list of compilable mutants. Empty if the
        source does not parse.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError) as exc:
        logger.warning(f"Skipping unparseable file {file_path}: {exc}")
        return []

    lines = source.split("\n")

    # Nodes whose mutation would be meaningless, unsafe, or unsplicable.
    excluded: Set[int] = set()
    excluded |= _collect_docstring_nodes(tree)
    excluded |= _collect_annotation_nodes(tree)
    excluded |= _collect_fstring_nodes(tree)
    excluded |= _collect_loop_condition_nodes(tree)

    mutants: List[Mutant] = []
    for operator_fn in MUTATION_OPERATORS:
        try:
            mutants.extend(operator_fn(tree, file_path, lines, excluded))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Operator {operator_fn.__name__} failed on {file_path}: {exc}")

    # Deduplicate by mutant_id, preserving first occurrence.
    unique: Dict[str, Mutant] = {}
    for mutant in mutants:
        unique.setdefault(mutant.mutant_id, mutant)

    # Sort for reproducible ordering under seeded sampling.
    return sorted(unique.values(), key=lambda m: (m.line, m.operator, m.mutant_id))
