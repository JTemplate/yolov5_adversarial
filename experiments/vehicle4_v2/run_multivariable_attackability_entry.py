#!/usr/bin/env python3
"""Direct-execution entry point for multivariable attackability analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.vehicle4_v2.run_multivariable_attackability import adjusted_effects, analysis

analysis.adjusted_effects = adjusted_effects

if __name__ == "__main__":
    analysis.main()
