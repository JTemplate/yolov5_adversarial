#!/usr/bin/env python3
"""Wait for detector training and execute all dependent v2 stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/vehicle4_v2"
STATE_DIR = RUN_ROOT / "pipeline"
LOG_DIR = STATE_DIR / "logs"
STATE_LOG = STATE_DIR / "events.jsonl"


def record(event: str, **details) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.time(), "event": event, **details}
    with STATE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def process_running(pattern: str) -> bool:
    result = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def wait_for_detectors() -> None:
    targets = {
        "yolov5s": RUN_ROOT / "detectors/yolov5s_seed0/weights/best.pt",
        "yolov5m": RUN_ROOT / "detectors/yolov5m_seed0/weights/best.pt",
        "fasterrcnn": RUN_ROOT / "detectors/fasterrcnn_seed0/complete.json",
    }
    patterns = {
        "yolov5s": "python train.py --weights yolov5s.pt",
        "yolov5m": "python train.py --weights yolov5m.pt",
        "fasterrcnn": "train_fasterrcnn.py.*fasterrcnn_seed0",
    }
    while True:
        status = {
            name: {"artifact": path.is_file(), "process_running": process_running(patterns[name])}
            for name, path in targets.items()
        }
        yolo_results_ok = {}
        for name in ["yolov5s", "yolov5m"]:
            results_path = RUN_ROOT / f"detectors/{name}_seed0/results.csv"
            line_count = (
                len(results_path.read_text(encoding="utf-8").splitlines())
                if results_path.exists()
                else 0
            )
            yolo_results_ok[name] = line_count >= 101
            status[name]["results_rows"] = max(line_count - 1, 0)
        complete = (
            all(item["artifact"] and not item["process_running"] for item in status.values())
            and all(yolo_results_ok.values())
        )
        STATE_DIR.joinpath("heartbeat.json").write_text(
            json.dumps({"time": time.time(), "detectors": status}, indent=2) + "\n",
            encoding="utf-8",
        )
        if complete:
            record("detectors_complete", status=status)
            return
        failed = [name for name, item in status.items() if not item["artifact"] and not item["process_running"]]
        failed.extend(
            name
            for name in ["yolov5s", "yolov5m"]
            if status[name]["artifact"] and not status[name]["process_running"] and not yolo_results_ok[name]
        )
        if failed:
            raise RuntimeError(f"Detector training stopped without a checkpoint: {failed}")
        time.sleep(60)


def run_logged(name: str, command: list[str], environment: dict | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    marker = STATE_DIR / f"{name}.complete.json"
    if marker.exists():
        record("stage_skipped", stage=name, marker=str(marker))
        return
    record("stage_started", stage=name, command=command)
    with LOG_DIR.joinpath(f"{name}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if result.returncode:
        record("stage_failed", stage=name, return_code=result.returncode)
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}")
    marker.write_text(json.dumps({"completed_at": time.time()}) + "\n", encoding="utf-8")
    record("stage_completed", stage=name)


def main() -> int:
    record("pipeline_monitor_started")
    wait_for_detectors()
    run_logged("patch_training", [sys.executable, "experiments/vehicle4_v2/train_patch_set.py"])

    render_environment = os.environ.copy()
    render_environment["CUDA_VISIBLE_DEVICES"] = "3"
    run_logged(
        "render_attacks",
        [sys.executable, "experiments/vehicle4_v2/render_attacks.py", "--device", "cuda:0"],
        render_environment,
    )

    evaluations = [
        ("yolov5s", "0", RUN_ROOT / "detectors/yolov5s_seed0/weights/best.pt", "8"),
        ("yolov5m", "1", RUN_ROOT / "detectors/yolov5m_seed0/weights/best.pt", "8"),
        ("fasterrcnn", "2", RUN_ROOT / "detectors/fasterrcnn_seed0/best.pt", "2"),
    ]
    processes = []
    for model, gpu, weights, batch_size in evaluations:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log_handle = LOG_DIR.joinpath(f"evaluate_{model}.log").open("a", encoding="utf-8")
        command = [
            sys.executable,
            "experiments/vehicle4_v2/evaluate_shared.py",
            "--model",
            model,
            "--weights",
            str(weights),
            "--device",
            "cuda:0",
            "--batch-size",
            batch_size,
        ]
        record("evaluation_started", model=model, command=command)
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
        processes.append((model, process, log_handle))
    failures = []
    for model, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        record("evaluation_finished", model=model, return_code=return_code)
        if return_code:
            failures.append(model)
    if failures:
        raise RuntimeError(f"Evaluation failed: {failures}")
    STATE_DIR.joinpath("evaluation.complete.json").write_text(
        json.dumps({"completed_at": time.time()}) + "\n", encoding="utf-8"
    )
    run_logged("aggregate", [sys.executable, "experiments/vehicle4_v2/aggregate_results.py"])
    record("pipeline_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
