#!/usr/bin/env python3
"""Run hard-target methods with per-process physical GPU selection.

``train_patch.py`` calls ``select_device`` after importing torch, so a child
must receive the physical device in its config as well as in
CUDA_VISIBLE_DEVICES. This wrapper patches config generation at runtime while
leaving the experiment orchestrator itself unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from experiments.vehicle4_v2 import run_hard_target_methods as pipeline  # noqa: E402


_original_make_config = pipeline.make_config


def make_config(template, args, method, seed, index):
    path, run_name = _original_make_config(template, args, method, seed, index)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["device"] = f"cuda:{pipeline.PHYSICAL_DEVICES[index]}"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, run_name


pipeline.make_config = make_config

if __name__ == "__main__":
    pipeline.main()
