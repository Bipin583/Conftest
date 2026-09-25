"""Package marker for the `data` layer of conformal-test-selection.

This file is deliberately non-empty. The development machine has an unrelated
editable install that places another `models` package on `sys.path`; without a
real `__init__.py` here, Python's import scan would resolve `models.train` to
that package instead of this one, because a regular package beats a namespace
package no matter where each sits on the path.
"""
