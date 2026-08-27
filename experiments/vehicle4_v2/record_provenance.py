#!/usr/bin/env python3
"""Capture immutable environment and input provenance for the v2 run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    dataset_manifest_path = ROOT / "data/visdrone_vehicle4_v2/dataset_manifest.json"
    output_path = ROOT / "runs/vehicle4_v2/provenance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [
        ROOT / "yolov5s.pt",
        ROOT / "yolov5m.pt",
        dataset_manifest_path,
        ROOT / "data/hyps/hyp.scratch-low.yaml",
        ROOT / "adv_patch_gen/configs/vehicle4_v2_full_10.json",
    ]
    payload = {
        "captured_at": time.time(),
        "working_directory": str(ROOT),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status_short": command("git", "status", "--short").splitlines(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpus": command(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ).splitlines(),
        "inputs": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs
        },
        "pretrained_weight_provenance": {
            "repository": "Ultralytics/YOLOv5",
            "version": "v7-compatible COCO checkpoints",
            "transport": "hf-mirror.com after official hosts were unreachable",
            "integrity": "SHA256 matched the Ultralytics Hugging Face repository metadata",
        },
        "dataset_manifest": json.loads(dataset_manifest_path.read_text(encoding="utf-8")),
        "protocol": {
            "detector_seed": 0,
            "patch_seeds": [42, 123, 2026],
            "evaluation_transform_seeds": [7101, 7102, 7103, 7104, 7105],
            "source_model": "YOLOv5s",
            "targets": ["YOLOv5s", "YOLOv5m", "FasterRCNN-ResNet50-FPN"],
            "primary_metric": "relative mAP50-95 drop",
            "secondary_metric": "ASR@confidence=0.4,IoU=0.5",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
