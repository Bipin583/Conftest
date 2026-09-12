"""
Superseded package: NOT AUTHORITATIVE.

The maintained implementation is `conftest.features`. These modules predate the
`conftest` package and are kept only because tests under `tests/` still import
them; fixes land in `src/conftest/` and are not mirrored here.

`pyproject.toml` sets `pythonpath = ["src"]`, which makes `import features`
resolve to this package rather than to `conftest.features` -- so an import written
without the `conftest.` prefix silently gets the unmaintained code. Always
import `conftest.features`.
"""

__superseded_by__ = "conftest.features"
