"""Live per-PR feature extraction for the conformal advisory job.

The offline pipeline (:mod:`data.features`) builds features from a *labelled*
mined-history CSV and cannot run on a live checkout. This module fills that gap
for the advisory CI job: it turns the real PR diff plus the repository checkout
into candidate records in the exact shape :meth:`SelectionPipeline.predict`
consumes, so the advisory comment reflects the change under review instead of a
fixed sample.

Only the statically-computable columns are produced. The eight historical
features (per-test pass/fail track record, ``co_change_frequency``) require a
persisted test-outcome store that CI does not carry, so they are left absent and
median-imputed by the fitted preprocessor; ``mutant_operator`` is a training-corpus
artefact and is dropped (its one-hot encoder tolerates the missing value). See the
plan and ``CONFORMAL_IMPLEMENTATION_AUDIT.md`` for the out-of-distribution caveat.

Every function catches its own errors and degrades to a neutral value or ``[]``;
:func:`build_live_candidates` never raises, so the advisory step cannot go red.
GitPython is not installed in the advisory job, so git is driven through the CLI.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from data.features import dice, jaccard, path_overlap_ratio, tokenize_identifier

_CONFIG_RE = re.compile(r"\.(?:ya?ml|toml|cfg|ini|json)$", re.IGNORECASE)
_FIX_RE = re.compile(r"\b(fix|fixes|fixed|bug|bugfix|hotfix|patch)\b", re.IGNORECASE)
_REFACTOR_RE = re.compile(r"\b(refactor|refactoring|cleanup|clean\s?up|rename|reorg)\b", re.IGNORECASE)


def _is_test_path(path: str) -> bool:
    """True for a *test file* by pytest convention — not any path containing "test".

    Substring-matching the whole path (the original behaviour) misclassifies every
    file in a repo whose directory name contains "test": this project lives under
    ``conformal-test-selection/``, so a real source change was never seen as source,
    ``changed_sources`` was always empty, and the advisory silently fell back to the
    committed sample instead of scoring the live PR. Judge by the filename
    convention :func:`discover_tests` uses, plus residence in a ``test``/``tests`` dir.
    """
    parts = path.lower().split("/")
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in {"test", "tests"} for part in parts[:-1])


def _is_python(path: str) -> bool:
    return path.endswith(".py")


def _is_source_path(path: str) -> bool:
    """A code file that is not a test file (the change the tests might exercise)."""
    return _is_python(path) and not _is_test_path(path)

def _run_git(args: List[str], repo_root: str) -> Optional[str]:
    """Run a git command in ``repo_root``; return stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def pr_diff(repo_root: str, base_sha: Optional[str], head_sha: Optional[str]) -> Dict[str, Any]:
    """Collect the PR's changed files and commit message from git.

    Uses the merge-base form ``base...head`` so the whole PR is seen rather than a
    single commit. Falls back to ``HEAD~1..HEAD`` when no base SHA is supplied
    (e.g. a ``workflow_dispatch`` run).

    Returns:
        ``{"files": [(added, removed, path), ...], "message": str}``. ``files`` is
        empty when git is unavailable or nothing changed (the fallback signal).
    """
    if base_sha and head_sha and base_sha != head_sha:
        numstat = _run_git(["diff", "--numstat", f"{base_sha}...{head_sha}"], repo_root)
        message = _run_git(["log", f"{base_sha}..{head_sha}", "--format=%B"], repo_root)
    else:
        numstat = _run_git(["diff", "--numstat", "HEAD~1", "HEAD"], repo_root)
        message = _run_git(["log", "-1", "--format=%B"], repo_root)

    files: List[Tuple[int, int, str]] = []
    for line in (numstat or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added_s, removed_s, path = parts
        added = 0 if added_s == "-" else int(added_s) if added_s.isdigit() else 0
        removed = 0 if removed_s == "-" else int(removed_s) if removed_s.isdigit() else 0
        files.append((added, removed, path.strip().replace("\\", "/")))

    return {"files": files, "message": (message or "").strip()}


def diff_level_features(diff: Dict[str, Any]) -> Dict[str, float]:
    """The 14 PR-level columns; identical across every candidate test."""
    files = diff.get("files", [])
    message = diff.get("message", "")
    paths = [p for _, _, p in files]

    lines_added = float(sum(a for a, _, _ in files))
    lines_removed = float(sum(r for _, r, _ in files))
    src_files = [p for p in paths if _is_source_path(p)]
    test_files = [p for p in paths if _is_python(p) and _is_test_path(p)]

    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "total_churn": lines_added + lines_removed,
        "files_changed": float(len(paths)),
        "commit_message_length": float(len(message)),
        "has_source_change": 1.0 if src_files else 0.0,
        "has_test_change": 1.0 if test_files else 0.0,
        "has_config_change": 1.0 if any(_CONFIG_RE.search(p) for p in paths) else 0.0,
        "diff_num_src_files": float(len(src_files)),
        "diff_num_test_files": float(len(test_files)),
        "diff_has_python": 1.0 if any(_is_python(p) for p in paths) else 0.0,
        "diff_is_fix_commit": 1.0 if _FIX_RE.search(message) else 0.0,
        "diff_is_refactor_commit": 1.0 if _REFACTOR_RE.search(message) else 0.0,
        "diff_msg_word_count": float(len(message.split())),
    }

def _imports_of(tree: ast.AST) -> Set[str]:
    """Every imported module/symbol name in a parsed module."""
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.add(module)
                    imports.add(f"{module}.{alias.name}")
                else:
                    imports.add(alias.name)
    return imports


def _complexity_of(tree: ast.AST) -> float:
    """Cyclomatic-complexity estimate: 1 + one per branch/boolean decision point."""
    complexity = 1.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert)):
            complexity += 1.0
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
        elif isinstance(node, ast.IfExp):
            complexity += 1.0
    return complexity


def ast_metrics(abs_path: Path) -> Dict[str, Any]:
    """Structural AST metrics of a test file; neutral values if it cannot be parsed.

    Returns the five ``ast_*`` columns plus the file's import set (used to derive
    the dependency proxies) under the extra key ``imports``.
    """
    neutral = {
        "ast_test_file_functions_count": 0.0,
        "ast_test_file_classes_count": 0.0,
        "ast_test_file_imports_count": 0.0,
        "ast_test_file_complexity": 1.0,
        "ast_test_is_parameterized": 0.0,
        "imports": set(),
    }
    try:
        code = abs_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(code, filename=str(abs_path))
    except (OSError, SyntaxError, ValueError):
        return neutral

    functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    imports = _imports_of(tree)
    parameterized = 1.0 if "parametrize" in code and "pytest" in code else 0.0

    return {
        "ast_test_file_functions_count": float(functions),
        "ast_test_file_classes_count": float(classes),
        "ast_test_file_imports_count": float(len(imports)),
        "ast_test_file_complexity": _complexity_of(tree),
        "ast_test_is_parameterized": parameterized,
        "imports": imports,
    }


def discover_tests(repo_root: str) -> List[Dict[str, Any]]:
    """Enumerate test functions via AST (``test_*.py`` / ``*_test.py``).

    AST-first (not pytest-first) on purpose: the advisory job installs only the
    conformal dependencies, so a repo-wide ``pytest --collect-only`` would try to
    import unrelated test modules (e.g. the ``src/conftest`` engine's) and fail.
    Returns one entry per ``test_*`` function / ``Test*`` method.
    """
    root = Path(repo_root).resolve()
    found: List[Dict[str, Any]] = []
    for dirpath, _, filenames in os.walk(root):
        # Skip vendored / VCS / cache trees to keep discovery cheap and relevant.
        if any(part in {".git", ".venv", "venv", "node_modules", "__pycache__"} for part in Path(dirpath).parts):
            continue
        for name in filenames:
            if not name.endswith(".py") or not (name.startswith("test_") or name.endswith("_test.py")):
                continue
            abs_path = Path(dirpath) / name
            try:
                rel = abs_path.relative_to(root).as_posix()
                tree = ast.parse(abs_path.read_text(encoding="utf-8", errors="replace"), filename=str(abs_path))
            except (OSError, SyntaxError, ValueError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or node.name.endswith("Test")):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                            found.append({"test_id": f"{rel}::{node.name}::{method.name}",
                                          "test_path": rel, "test_function": method.name})
                elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    found.append({"test_id": f"{rel}::{node.name}", "test_path": rel, "test_function": node.name})
    return found

def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _direct_import(imports: Set[str], source_path: str) -> bool:
    """Does a test importing ``imports`` reference the changed source module?

    Matches on the module stem (``session`` for ``src/auth/session.py``) or on a
    dotted-path suffix (``auth.session``), so both ``import src.auth.session`` and
    ``from auth import session`` count.
    """
    stem = Path(source_path).stem
    tokens: Set[str] = set()
    for imp in imports:
        tokens.update(imp.split("."))
    if stem in tokens:
        return True
    dotted = source_path[:-3].replace("/", ".") if source_path.endswith(".py") else source_path.replace("/", ".")
    return any(imp.endswith(dotted) or dotted.endswith(imp) or imp.endswith("." + stem) for imp in imports)


def _prior_mod_count(repo_root: str, path: str) -> float:
    """How many commits previously touched ``path`` (VCS history, not test outcomes)."""
    out = _run_git(["log", "--oneline", "--", path], repo_root)
    if out is None:
        return 0.0
    return float(sum(1 for line in out.splitlines() if line.strip()))

def build_live_candidates(
    repo_root: str,
    base_sha: Optional[str],
    head_sha: Optional[str],
    repo_name: str,
    max_tests: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Turn a live PR into candidate records the pipeline can score.

    Args:
        repo_root: Path to the checked-out repository.
        base_sha, head_sha: PR base and head commits (``base...head`` diff). When
            absent, falls back to the last commit (``HEAD~1..HEAD``).
        repo_name: Value for the ``repo`` categorical (unseen → all-zero one-hot).
        max_tests: Optional safety cap on candidates processed.

    Returns:
        One record per discovered test in ``modelled_columns`` shape, or ``[]``
        when there is no diff, no changed source file, or no test (the signal for
        the caller to fall back to the committed sample). Never raises.
    """
    try:
        diff = pr_diff(repo_root, base_sha, head_sha)
        changed_sources = sorted({p for _, _, p in diff.get("files", []) if _is_source_path(p)})
        if not changed_sources:
            return []
        tests = discover_tests(repo_root)
        if not tests:
            return []
        if max_tests and len(tests) > max_tests:
            tests = tests[:max_tests]

        diff_feats = diff_level_features(diff)
        root = Path(repo_root).resolve()

        for t in tests:
            t["_ast"] = ast_metrics(root / t["test_path"])
            t["_path_tokens"] = tokenize_identifier(t["test_path"])
            t["_func_tokens"] = tokenize_identifier(t["test_function"])

        src_info: Dict[str, Dict[str, Any]] = {}
        for src in changed_sources:
            reverse = sum(1 for t in tests if _direct_import(t["_ast"]["imports"], src))
            src_info[src] = {
                "tokens": tokenize_identifier(src),
                "reverse_deps": float(reverse),
                "prior_mods": _prior_mod_count(repo_root, src),
            }

        records: List[Dict[str, Any]] = []
        for t in tests:
            best_src, best_sim, best_ov = changed_sources[0], -1.0, -1.0
            for src in changed_sources:
                sim = dice(src_info[src]["tokens"], t["_path_tokens"])
                overlap = path_overlap_ratio(src, t["test_path"])
                if (sim, overlap) > (best_sim, best_ov):
                    best_src, best_sim, best_ov = src, sim, overlap

            info = src_info[best_src]
            ast_m = t["_ast"]
            direct = _direct_import(ast_m["imports"], best_src)
            coupled = 1.0 if info["tokens"] & (t["_path_tokens"] | t["_func_tokens"]) else 0.0
            records.append({
                "test_id": t["test_id"],
                "test_path": t["test_path"],
                "changed_file_path": best_src,
                **diff_feats,
                "file_test_similarity": max(best_sim, 0.0),
                "lexical_similarity": jaccard(info["tokens"], t["_func_tokens"]),
                "path_overlap": max(best_ov, 0.0),
                "same_module": 1.0 if _dirname(best_src) == _dirname(t["test_path"]) else 0.0,
                "test_name_length": float(len(t["test_function"])),
                "dependency_distance": 1.0 - max(best_ov, 0.0),
                "ast_test_file_functions_count": ast_m["ast_test_file_functions_count"],
                "ast_test_file_classes_count": ast_m["ast_test_file_classes_count"],
                "ast_test_file_imports_count": ast_m["ast_test_file_imports_count"],
                "ast_test_file_complexity": ast_m["ast_test_file_complexity"],
                "ast_test_is_parameterized": ast_m["ast_test_is_parameterized"],
                "dep_is_direct_import": 1.0 if direct else 0.0,
                "dep_name_heuristic_coupled": coupled,
                "dep_is_reachable": 1.0 if (direct or best_ov > 0) else 0.0,
                "dep_max_reverse_dependencies": info["reverse_deps"],
                "dep_test_total_out_degree": ast_m["ast_test_file_imports_count"],
                "repo": repo_name,
                "hist_changed_files_prior_mod_count": info["prior_mods"],
            })
        return records
    except Exception:  # noqa: BLE001 - advisory extraction must never raise
        return []




