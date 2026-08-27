#!/usr/bin/env python3
"""Run the B0/B1/B2 first-round hard-target method comparison.

Nine patch trainings are executed in three waves (one wave per method, three
seeds per wave) on physical GPUs 0, 1, and 3. Rendering and YOLOv5s evaluation
then run method-by-method on GPU 3. The process is restartable: completed
patches, renders, and summaries are skipped and every stage is recorded in a
heartbeat JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 123, 2026]
PHYSICAL_DEVICES = ["0", "1", "3"]
METHODS = ["B0", "B1", "B2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiment_data/vehicle4_v2/runs/method_comparison_v1")
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "experiment_data/vehicle4_v2/analysis/method_comparison_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_heartbeat(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_patch(run_root: Path, method: str, seed: int) -> Path | None:
    run_name = f"hard_target_{method}_seed{seed}"
    candidates = sorted(run_root.joinpath(method, "patch_training").glob(f"*_{run_name}"))
    candidates = [path for path in candidates if path.joinpath("patches/e_10.png").is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def make_config(template: dict, args, method: str, seed: int, index: int) -> tuple[Path, str]:
    method_root = args.run_root / method
    config_dir = args.run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"hard_target_{method}_seed{seed}"
    config = dict(template)
    for key in ["image_dir", "label_dir", "triplet_printfile", "weights_file"]:
        config[key] = str((ROOT / config[key]).resolve())
    config.update({
        "log_dir": str((method_root / "patch_training").resolve()),
        "patch_name": run_name,
        "device": "cuda:0",  # physical GPU is selected through CUDA_VISIBLE_DEVICES
        "tensorboard_port": 19300 + METHODS.index(method) * 10 + index,
        "batch_size": args.batch_size,
        "n_epochs": args.epochs,
        "loss_variant": method,
    })
    path = config_dir / f"{run_name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, run_name


def train_wave(args, template: dict, method: str, heartbeat: Path) -> None:
    method_root = args.run_root / method
    (method_root / "logs").mkdir(parents=True, exist_ok=True)
    processes = []
    for index, (seed, physical_gpu) in enumerate(zip(SEEDS, PHYSICAL_DEVICES)):
        existing = completed_patch(args.run_root, method, seed)
        if existing:
            continue
        config_path, run_name = make_config(template, args, method, seed, index)
        log_path = method_root / "logs" / f"train_seed{seed}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update({"PATCH_SEED": str(seed), "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": physical_gpu})
        command = [sys.executable, str(ROOT / "experiments/vehicle4_v2/train_patch_variant.py"), "--cfg", str(config_path), "--variant", method]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        processes.append({"seed": seed, "gpu": physical_gpu, "run_name": run_name, "process": process, "log": log_handle, "log_path": str(log_path), "started_at": time.time()})
    while processes:
        state = []
        for job in processes:
            code = job["process"].poll()
            state.append({"seed": job["seed"], "gpu": job["gpu"], "run_name": job["run_name"], "pid": job["process"].pid, "return_code": code, "log": job["log_path"], "elapsed_seconds": time.time() - job["started_at"]})
        write_heartbeat(heartbeat, {"stage": "training", "method": method, "updated_at": time.time(), "jobs": state})
        finished = [job for job in processes if job["process"].poll() is not None]
        for job in finished:
            code = job["process"].returncode
            job["log"].close()
            processes.remove(job)
            if code:
                raise RuntimeError(f"{method} seed {job['seed']} failed with exit code {code}; see {job['log_path']}")
        if processes:
            time.sleep(30)
    missing = [seed for seed in SEEDS if completed_patch(args.run_root, method, seed) is None]
    if missing:
        raise RuntimeError(f"{method} finished without e_10.png for seeds {missing}")


def write_registry(args, method: str) -> Path:
    records = []
    for seed in SEEDS:
        run_dir = completed_patch(args.run_root, method, seed)
        assert run_dir is not None
        patch_path = run_dir / "patches/e_10.png"
        records.append({"seed": seed, "run_dir": str(run_dir), "patch": str(patch_path), "sha256": sha256(patch_path)})
    path = args.run_root / method / "patch_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": method, "source_model": "yolov5s_vehicle4_v2_seed0", "patches": records}, indent=2) + "\n", encoding="utf-8")
    return path


def run_logged(args, method: str, stage: str, command: list[str], env: dict | None = None) -> None:
    method_root = args.run_root / method
    log_dir = method_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    marker = method_root / f"{stage}.complete.json"
    if marker.exists(): return
    heartbeat = args.run_root / "heartbeat.json"
    write_heartbeat(heartbeat, {"stage": stage, "method": method, "updated_at": time.time(), "command": command})
    with (log_dir / f"{stage}.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"{method} {stage} failed with exit code {result.returncode}; see {log_dir / (stage + '.log')}")
    marker.write_text(json.dumps({"completed_at": time.time()}) + "\n", encoding="utf-8")


def main():
    args = parse_args(); args.run_root.mkdir(parents=True, exist_ok=True); args.analysis_root.mkdir(parents=True, exist_ok=True)
    template_path = ROOT / "adv_patch_gen/configs/vehicle4_v2_full_10.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    weights = (ROOT / template["weights_file"]).resolve()
    if not weights.is_file(): raise FileNotFoundError(weights)
    heartbeat = args.run_root / "heartbeat.json"
    write_heartbeat(heartbeat, {"stage": "starting", "updated_at": time.time(), "methods": METHODS, "seeds": SEEDS, "weights": str(weights)})
    for method in METHODS:
        train_wave(args, template, method, heartbeat)
        registry = write_registry(args, method)
        if not args.skip_evaluation:
            render_env = os.environ.copy(); render_env["CUDA_VISIBLE_DEVICES"] = "3"
            render_command = [sys.executable, str(ROOT / "experiments/vehicle4_v2/render_attacks.py"), "--patch-registry", str(registry), "--output-root", str(args.run_root / method / "rendered"), "--device", "cuda:0"]
            run_logged(args, method, "render", render_command, render_env)
            eval_command = [sys.executable, str(ROOT / "experiments/vehicle4_v2/evaluate_shared.py"), "--model", "yolov5s", "--weights", str(weights), "--render-root", str(args.run_root / method / "rendered"), "--output-root", str(args.run_root / method / "evaluation"), "--device", "cuda:0", "--batch-size", "8"]
            run_logged(args, method, "evaluation", eval_command, render_env)
        write_heartbeat(heartbeat, {"stage": "method_complete", "method": method, "updated_at": time.time()})
    if not args.skip_evaluation:
        command = [sys.executable, str(ROOT / "experiments/vehicle4_v2/summarize_hard_target_methods.py"), "--run-root", str(args.run_root), "--analysis-root", str(args.analysis_root)]
        with (args.run_root / "summary.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError("method summary failed; see summary.log")
    write_heartbeat(heartbeat, {"stage": "complete", "updated_at": time.time(), "methods": METHODS})
    print(args.run_root / "heartbeat.json")


if __name__ == "__main__":
    main()
