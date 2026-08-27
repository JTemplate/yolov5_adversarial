#!/usr/bin/env python3
"""Aggregate paired transfer metrics across patch and transform seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ["yolov5s", "yolov5m", "fasterrcnn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("runs/vehicle4_v2/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis/vehicle4_v2_transfer"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser.parse_args()


def hierarchical_ci(values: dict[int, dict[int, float]], replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    patch_seeds = sorted(values)
    samples = []
    for _ in range(replicates):
        selected_patch_seeds = rng.choice(patch_seeds, size=len(patch_seeds), replace=True)
        replicate_values = []
        for patch_seed in selected_patch_seeds:
            transforms = sorted(values[int(patch_seed)])
            selected_transforms = rng.choice(transforms, size=len(transforms), replace=True)
            replicate_values.extend(values[int(patch_seed)][int(index)] for index in selected_transforms)
        samples.append(float(np.mean(replicate_values)))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_asr_image_rows(learned: dict, random_result: dict) -> np.ndarray:
    """Return paired per-image [clean, learned_attacked, random_attacked] rows."""
    learned_by_image = {int(row["image_id"]): row for row in learned["asr"]["per_image"]}
    random_by_image = {int(row["image_id"]): row for row in random_result["asr"]["per_image"]}
    if learned_by_image.keys() != random_by_image.keys():
        raise ValueError("Learned/random ASR image sets are not paired")
    rows = []
    for image_id in sorted(learned_by_image):
        learned_row = learned_by_image[image_id]
        random_row = random_by_image[image_id]
        clean_detections = int(learned_row["clean_detections"])
        if clean_detections != int(random_row["clean_detections"]):
            raise ValueError(f"Clean detection count differs for image {image_id}")
        rows.append(
            [
                clean_detections,
                int(learned_row["attacked"]),
                int(random_row["attacked"]),
            ]
        )
    if not rows:
        raise ValueError("No clean reference detections are available for image bootstrap")
    return np.asarray(rows, dtype=np.int64)


def hierarchical_asr_ci(values: dict[int, dict[int, np.ndarray]], replicates: int, seed: int) -> list[float]:
    """Bootstrap paired ASR gain over patch, transform, then image levels."""
    rng = np.random.default_rng(seed)
    patch_seeds = np.asarray(sorted(values), dtype=np.int64)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate_index in range(replicates):
        selected_patch_seeds = rng.choice(patch_seeds, size=len(patch_seeds), replace=True)
        clean_total = 0
        learned_attacked_total = 0
        random_attacked_total = 0
        for patch_seed in selected_patch_seeds:
            transforms = np.asarray(sorted(values[int(patch_seed)]), dtype=np.int64)
            selected_transforms = rng.choice(transforms, size=len(transforms), replace=True)
            for transform_index in selected_transforms:
                rows = values[int(patch_seed)][int(transform_index)]
                selected_images = rng.integers(0, len(rows), size=len(rows))
                totals = rows[selected_images].sum(axis=0)
                clean_total += int(totals[0])
                learned_attacked_total += int(totals[1])
                random_attacked_total += int(totals[2])
        samples[replicate_index] = (
            (learned_attacked_total - random_attacked_total) / clean_total if clean_total else np.nan
        )
    if np.isnan(samples).any():
        raise ValueError("An image-level ASR bootstrap replicate had no clean detections")
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    detailed_rows = []
    aggregate_rows = []
    aggregate_json = {
        "primary_metric": "relative_map50_95_drop",
        "paired_gain": "learned_relative_drop_minus_random_relative_drop",
        "bootstrap": {
            "relative_map_drop_levels": ["patch_seed", "transform_seed"],
            "paired_asr_gain_levels": ["patch_seed", "transform_seed", "image_id"],
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "models": {},
    }
    for model_index, model in enumerate(MODELS):
        summary_path = args.evaluation_root / model / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        conditions = summary["conditions"]
        clean_map = conditions["clean"]["metrics"]["map50_95"]
        paired_gains: dict[int, dict[int, float]] = {}
        paired_asr_rows: dict[int, dict[int, np.ndarray]] = {}
        learned_drops = []
        random_drops = []
        learned_asrs = []
        random_asrs = []
        for name, learned in conditions.items():
            metadata = learned["condition"]
            if metadata.get("kind") != "learned":
                continue
            patch_seed = int(metadata["patch_seed"])
            transform_index = int(metadata["transform_index"])
            random_name = name.removesuffix("_learned") + "_random"
            random_result = conditions[random_name]
            learned_drop = float(learned["relative_map_drop"])
            random_drop = float(random_result["relative_map_drop"])
            gain = learned_drop - random_drop
            paired_gains.setdefault(patch_seed, {})[transform_index] = gain
            paired_asr_rows.setdefault(patch_seed, {})[transform_index] = paired_asr_image_rows(learned, random_result)
            learned_drops.append(learned_drop)
            random_drops.append(random_drop)
            learned_asrs.append(float(learned["asr"]["asr"]))
            random_asrs.append(float(random_result["asr"]["asr"]))
            detailed_rows.append(
                {
                    "model": model,
                    "patch_seed": patch_seed,
                    "transform_index": transform_index,
                    "transform_seed": metadata["transform_seed"],
                    "clean_map50_95": clean_map,
                    "learned_map50_95": learned["metrics"]["map50_95"],
                    "random_map50_95": random_result["metrics"]["map50_95"],
                    "learned_relative_drop": learned_drop,
                    "random_relative_drop": random_drop,
                    "paired_gain": gain,
                    "learned_asr": learned["asr"]["asr"],
                    "random_asr": random_result["asr"]["asr"],
                }
            )
        ci = hierarchical_ci(paired_gains, args.bootstrap_replicates, args.bootstrap_seed + model_index)
        asr_ci = hierarchical_asr_ci(
            paired_asr_rows,
            args.bootstrap_replicates,
            args.bootstrap_seed + 1000 + model_index,
        )
        aggregate = {
            "clean_map50_95": clean_map,
            "learned_relative_drop_mean": float(np.mean(learned_drops)),
            "learned_relative_drop_std": float(np.std(learned_drops, ddof=1)),
            "random_relative_drop_mean": float(np.mean(random_drops)),
            "random_relative_drop_std": float(np.std(random_drops, ddof=1)),
            "paired_gain_mean": float(np.mean(np.asarray(learned_drops) - np.asarray(random_drops))),
            "paired_gain_95ci": ci,
            "learned_asr_mean": float(np.mean(learned_asrs)),
            "random_asr_mean": float(np.mean(random_asrs)),
            "paired_asr_gain_mean": float(np.mean(np.asarray(learned_asrs) - np.asarray(random_asrs))),
            "paired_asr_gain_95ci": asr_ci,
            "transfer_detected": ci[0] > 0.0,
            "practically_meaningful_5pp": float(np.mean(np.asarray(learned_drops) - np.asarray(random_drops))) >= 0.05,
        }
        aggregate_json["models"][model] = aggregate
        aggregate_rows.append(
            {
                "model": model,
                **aggregate,
                "paired_gain_95ci": json.dumps(ci),
                "paired_asr_gain_95ci": json.dumps(asr_ci),
            }
        )

    with args.output_root.joinpath("paired_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)
    with args.output_root.joinpath("model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_rows[0].keys())
        writer.writeheader()
        writer.writerows(aggregate_rows)
    args.output_root.joinpath("model_summary.json").write_text(
        json.dumps(aggregate_json, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root / "model_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
