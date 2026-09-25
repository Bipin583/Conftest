"""Root pytest configuration: make the project importable before collection.

The development machine has an unrelated editable install that exposes another
package named ``models``. A regular package beats a namespace package during
Python's import scan regardless of path order, so ``import models.train`` can
resolve to the wrong project unless this directory is genuinely first on
``sys.path``. Inserting at position 0 here -- in a root ``conftest.py``, which
pytest imports before it collects anything -- is what makes ``pytest`` behave
identically to ``python cli.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
