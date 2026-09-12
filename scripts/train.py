"""
Alias entry point: ``python scripts/train.py`` -> ``train_ensemble.py``.

The documented workflow names ``scripts/train.py``. What ships is the 5-seed
LightGBM ensemble, so this forwards to ``scripts/train_ensemble.py`` rather than
to the single-model trainer -- training the single model here would produce an
artifact that no shipping code path loads.

Use ``scripts/train_model.py`` directly for the single-model ablation baseline,
and ``scripts/calibrate_model.py`` afterwards to fit the calibrator.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).with_name("train_ensemble.py")

if __name__ == "__main__":
    if not TARGET.exists():  # pragma: no cover - guards a broken checkout
        raise SystemExit(f"missing implementation: {TARGET}")
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
