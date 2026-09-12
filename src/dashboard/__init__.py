"""
Superseded package: NOT AUTHORITATIVE.

The maintained implementation is `conftest.api`. These modules predate the
`conftest` package and are kept only because tests under `tests/` still import
them; fixes land in `src/conftest/` and are not mirrored here.

`pyproject.toml` sets `pythonpath = ["src"]`, which makes `import dashboard`
resolve to this package rather than to `conftest.api` -- so an import written
without the `conftest.` prefix silently gets the unmaintained code. Always
import `conftest.api`.
"""

__superseded_by__ = "conftest.api"
