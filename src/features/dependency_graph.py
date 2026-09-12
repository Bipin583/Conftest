"""
ConfTest Dependency Graph & Static Call-Path Analyzer
Maps static call paths and import hierarchies between modified classes/functions and test files.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **conftest.features.dependency_graph** (DependencyGraphBuilder).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "conftest.features.dependency_graph"

import ast
import os
from typing import Dict, List, Set

class StaticDependencyGraph:
    """
    Parses Python source and test files to build an in-memory import and call dependency graph.
    """
    GENERIC_ROOTS = {"src", "tests", "test", "lib", "app", "core", "pkg"}

    def __init__(self):
        self.import_graph: Dict[str, Set[str]] = {} # file -> set of imported modules
        self.call_graph: Dict[str, Set[str]] = {}   # func/class -> set of called functions

    def analyze_imports(self, file_path: str, code_content: str) -> Set[str]:
        """
        Parses all imported module and symbol names from Python code using the standard AST library.
        """
        imported_symbols: Set[str] = set()
        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_symbols.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imported_symbols.add(module)
                    for alias in node.names:
                        imported_symbols.add(f"{module}.{alias.name}")
        except Exception:
            for line in code_content.splitlines():
                if "import " in line:
                    imported_symbols.update(line.replace("import", "").replace("from", "").strip().split())

        self.import_graph[file_path] = imported_symbols
        return imported_symbols

    def is_test_directly_dependent(self, test_imports: Set[str], modified_files: List[str]) -> bool:
        """
        Checks if any modified file or its base module name is imported by the test suite.
        """
        for mod_file in modified_files:
            # Normalize path: 'src/auth/login.py' -> module tokens ['auth', 'login']
            clean_name = mod_file.replace("/", ".").replace("\\", ".").replace(".py", "")
            tokens = [t for t in clean_name.split(".") if t and t.lower() not in self.GENERIC_ROOTS]
            
            # Check if any non-generic token or compound path matches imports
            for tok in tokens:
                for imp in test_imports:
                    imp_parts = imp.split(".")
                    if tok in imp_parts:
                        return True
            if clean_name in test_imports:
                return True
        return False

    def compute_dependency_score(self, test_file: str, modified_files: List[str]) -> float:
        """
        Returns a normalized dependency overlap score between 0.0 and 1.0.
        """
        test_imports = self.import_graph.get(test_file, set())
        if not test_imports:
            return 0.1

        if self.is_test_directly_dependent(test_imports, modified_files):
            return 1.0

        test_tokens = set(t for t in test_file.replace("/", ".").replace("\\", ".").split(".") if t not in self.GENERIC_ROOTS)
        score = 0.0
        for mod_file in modified_files:
            mod_tokens = set(t for t in mod_file.replace("/", ".").replace("\\", ".").split(".") if t not in self.GENERIC_ROOTS)
            if test_tokens.union(mod_tokens):
                overlap = len(test_tokens.intersection(mod_tokens)) / len(test_tokens.union(mod_tokens))
                score = max(score, overlap)

        return float(min(score, 0.9))
