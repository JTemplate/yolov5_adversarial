#!/usr/bin/env python3
"""Train the three registered Vehicle4 v2 patches in parallel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 2026]
DEFAULT_DEVICES = ["cuda:0", "cuda:1", "cuda:3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("adv_patch_gen/configs/vehicle4_v2_full_10.json"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/vehicle4_v2/detectors/yolov5s_seed0/weights/best.pt"),
    )
    parser.add_argument("--devices", nargs=3, default=DEFAULT_DEVICES)
    parser.add_argument("--output-root", type=Path, default=Path("runs/vehicle4_v2"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tensorboard-port", type=int, default=19200)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Source YOLOv5s checkpoint is missing: {args.weights}")
    base = json.loads(args.template.read_text(encoding="utf-8"))
    config_dir = args.output_root / "patch_configs"
    log_dir = args.output_root / "pipeline" / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    started_at = time.time()
    for index, (seed, device) in enumerate(zip(SEEDS, args.devices)):
        run_name = f"vehicle4_v2_full10_seed{seed}"
        existing = sorted(args.output_root.joinpath("patch_training").glob(f"*_{run_name}"))
        complete = [path for path in existing if path.joinpath("patches/e_10.png").is_file()]
        if complete:
            continue
        config = dict(base)
        config.update(
            {
                "weights_file": str(args.weights),
                "patch_name": run_name,
                "device": device,
                "tensorboard_port": args.tensorboard_port + index,
                "batch_size": args.batch_size,
            }
        )
        config_path = config_dir / f"{run_name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        log_path = log_dir / f"train_patch_seed{seed}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PATCH_SEED": str(seed),
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": device.rsplit(":", 1)[-1],
            }
        )
        process = subprocess.Popen(
            [sys.executable, "train_patch.py", "--cfg", str(config_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((seed, run_name, process, log_handle, log_path))

    failures = []
    for seed, run_name, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append({"seed": seed, "return_code": return_code, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"Patch training failures: {failures}")

    registry = {
        "source_model": "yolov5s_vehicle4_v2_seed0",
        "source_weights": str(args.weights),
        "source_weights_sha256": file_sha256(args.weights),
        "started_at": started_at,
        "completed_at": time.time(),
        "patches": [],
    }
    for seed in SEEDS:
        run_name = f"vehicle4_v2_full10_seed{seed}"
        candidates = sorted(args.output_root.joinpath("patch_training").glob(f"*_{run_name}"))
        candidates = [path for path in candidates if path.joinpath("patches/e_10.png").is_file()]
        if not candidates:
            raise FileNotFoundError(f"No completed patch run for seed {seed}")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        patch_path = run_dir / "patches" / "e_10.png"
        registry["patches"].append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "patch": str(patch_path),
                "sha256": file_sha256(patch_path),
            }
        )
    registry_path = args.output_root / "patch_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(registry_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
