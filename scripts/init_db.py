"""
Database initializer entry point: ``python scripts/init_db.py``.

The implementation lives in :mod:`conftest.db.init_db` so that the API and the
tests can initialise a schema without shelling out. This script exists because
the documented workflow (README, DOCUMENTATION.md, the CI recipe) names
``scripts/init_db.py``, and a documented command that does not exist is a
documentation defect. It delegates; it does not reimplement.
"""

from __future__ import annotations

import sys

from conftest.config import settings
from conftest.db.init_db import init_db


def main() -> int:
    print(f"Initializing schema at: {settings.database_url}")
    if not init_db():
        print("FAILED: schema initialization did not complete. See the log above.", file=sys.stderr)
        return 1
    print("OK: all tables created (idempotent -- existing tables are left alone).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
