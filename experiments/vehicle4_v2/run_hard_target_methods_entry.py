#!/usr/bin/env python3
"""Direct entry point with the PyTorch 2.7 checkpoint compatibility flag."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from experiments.vehicle4_v2.run_hard_target_methods import main

if __name__ == "__main__":
    main()
