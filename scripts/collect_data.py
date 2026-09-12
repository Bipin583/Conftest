"""
Alias entry point: ``python scripts/collect_data.py`` -> ``collect_repository_data.py``.

The documented workflow names ``scripts/collect_data.py``. The real collector is
``scripts/collect_repository_data.py``; this file forwards argv to it verbatim so
there is exactly one implementation and one set of flags to keep correct.

Run ``python scripts/collect_data.py --help`` to see the collector's own flags.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).with_name("collect_repository_data.py")

if __name__ == "__main__":
    if not TARGET.exists():  # pragma: no cover - guards a broken checkout
        raise SystemExit(f"missing implementation: {TARGET}")
    # argv[0] is rewritten so the delegate's --help prints its own usage.
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
