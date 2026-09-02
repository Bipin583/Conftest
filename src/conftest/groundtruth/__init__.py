"""
ConfTest Ground-Truth Acquisition Package.

Produces regression-test-selection datasets whose failure labels are
MEASURED by executing real test suites, never simulated.

Modules:
    mutators:         AST-level single-node mutation operators.
    mutation_harness: Applies mutants, runs real suites, records real outcomes.

Design invariant
----------------
No module in this package may draw a random number to decide a test
outcome. Randomness is permitted ONLY for selecting *which* mutants to
sample. Every ``label_failed`` value must trace back to a real pytest run.
"""

__all__ = ["mutators", "mutation_harness"]
