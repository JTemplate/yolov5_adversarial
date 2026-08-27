#!/usr/bin/env python3
"""Stratify paired learned/random ASR by clean-detection attributes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MODELS = ["yolov5s", "yolov5m", "fasterrcnn"]
CLASS_NAMES = {1: "car", 2: "van", 3: "truck", 4: "bus"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("experiment_data/vehicle4_v2/runs/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("experiment_data/vehicle4_v2/analysis/attackability"))
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group(predictions, confidence):
    grouped = defaultdict(list)
    for prediction in predictions:
        if float(prediction["score"]) >= confidence:
            grouped[int(prediction["image_id"])].append(prediction)
    return grouped


def box_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def survival(clean, attacked, threshold):
    """Formal evaluator's greedy same-class IoU matching."""
    used, result = set(), []
    for reference in clean:
        candidates = [
            (box_iou(reference["bbox"], item["bbox"]), index)
            for index, item in enumerate(attacked)
            if index not in used and item["category_id"] == reference["category_id"]
        ]
        if candidates:
            best, index = max(candidates)
            if best >= threshold:
                used.add(index)
                result.append(True)
                continue
        result.append(False)
    return result


def size_bucket(area):
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def area_bucket(area):
    if area < 100:
        return "<100 px2"
    if area < 250:
        return "100–<250 px2"
    if area < 500:
        return "250–<500 px2"
    if area < 1000:
        return "500–<1,000 px2"
    if area < 2500:
        return "1,000–<2,500 px2"
    return ">=2,500 px2"


def confidence_bucket(score):
    if score < 0.5:
        return "0.40–<0.50"
    if score < 0.6:
        return "0.50–<0.60"
    if score < 0.7:
        return "0.60–<0.70"
    if score < 0.8:
        return "0.70–<0.80"
    if score < 0.9:
        return "0.80–<0.90"
    return ">=0.90"


def density_bucket(count):
    if count <= 10:
        return "1–10 detections/image"
    if count <= 20:
        return "11–20 detections/image"
    if count <= 30:
        return "21–30 detections/image"
    if count <= 40:
        return "31–40 detections/image"
    return ">40 detections/image"


def make_records(model, evaluation_root, confidence, iou_threshold):
    model_root = evaluation_root / model
    summary = load(model_root / "summary.json")
    clean = group(load(model_root / "clean/predictions.json"), confidence)
    records, indexes = [], {}
    for image_id in sorted(clean):
        references = clean[image_id]
        for object_index, reference in enumerate(references):
            _, _, width, height = [float(value) for value in reference["bbox"]]
            area = max(0.0, width * height)
            category = CLASS_NAMES.get(int(reference["category_id"]), str(reference["category_id"]))
            size = size_bucket(area)
            indexes[(image_id, object_index)] = len(records)
            records.append(
                {
                    "model": model,
                    "image_id": image_id,
                    "clean_prediction_index": object_index,
                    "class": category,
                    "size": size,
                    "bbox_area": area_bucket(area),
                    "bbox_area_px2": area,
                    "bbox_area_fraction": area / (640.0 * 640.0),
                    "clean_confidence": float(reference["score"]),
                    "clean_confidence_bucket": confidence_bucket(float(reference["score"])),
                    "image_density": len(references),
                    "image_density_bucket": density_bucket(len(references)),
                    "class_size": f"{category}|{size}",
                    "condition_pairs": 0,
                    "learned_attacked_count": 0,
                    "random_attacked_count": 0,
                }
            )
    learned_conditions = [item for item in summary["conditions"].values() if item["condition"].get("kind") == "learned"]
    if len(learned_conditions) != 15:
        raise ValueError(f"Expected 15 learned conditions for {model}, found {len(learned_conditions)}")
    for learned_result in learned_conditions:
        learned_name = learned_result["condition"]["name"]
        random_name = learned_name.removesuffix("_learned") + "_random"
        learned = group(load(model_root / learned_name / "predictions.json"), confidence)
        random = group(load(model_root / random_name / "predictions.json"), confidence)
        for image_id, references in clean.items():
            learned_ok = survival(references, learned.get(image_id, []), iou_threshold)
            random_ok = survival(references, random.get(image_id, []), iou_threshold)
            for object_index, (learned_survived, random_survived) in enumerate(zip(learned_ok, random_ok)):
                record = records[indexes[(image_id, object_index)]]
                record["condition_pairs"] += 1
                record["learned_attacked_count"] += int(not learned_survived)
                record["random_attacked_count"] += int(not random_survived)
    for record in records:
        if record["condition_pairs"] != 15:
            raise ValueError(f"Incomplete condition pairing for {model}")
        pairs = record["condition_pairs"]
        record["learned_asr"] = record["learned_attacked_count"] / pairs
        record["random_asr"] = record["random_attacked_count"] / pairs
        record["paired_asr_gain"] = record["learned_asr"] - record["random_asr"]
    return records


DIMENSIONS = {
    "class": "Detected class",
    "size": "COCO size from clean predicted bbox area",
    "bbox_area": "Fixed clean predicted bbox-area bins",
    "clean_confidence_bucket": "Clean reference confidence",
    "image_density_bucket": "Clean reference detections per image",
    "class_size": "Class x COCO size interaction",
}


def summarize(records, dimension):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[dimension]].append(record)
    rows = []
    for bucket, items in grouped.items():
        trials = sum(item["condition_pairs"] for item in items)
        learned_count = sum(item["learned_attacked_count"] for item in items)
        random_count = sum(item["random_attacked_count"] for item in items)
        learned_asr = learned_count / trials
        random_asr = random_count / trials
        rows.append(
            {
                "model": items[0]["model"],
                "dimension": dimension,
                "dimension_definition": DIMENSIONS[dimension],
                "bucket": bucket,
                "unique_clean_objects": len(items),
                "condition_pairs": items[0]["condition_pairs"],
                "reference_trials": trials,
                "learned_attacked_trials": learned_count,
                "random_attacked_trials": random_count,
                "learned_asr": learned_asr,
                "random_asr": random_asr,
                "learned_minus_random_asr_gain": learned_asr - random_asr,
            }
        )
    return rows


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_records, all_buckets = [], []
    for model in MODELS:
        records = make_records(model, args.evaluation_root, args.confidence, args.iou)
        all_records.extend(records)
        for dimension in DIMENSIONS:
            all_buckets.extend(summarize(records, dimension))
    object_fields = [
        "model",
        "image_id",
        "clean_prediction_index",
        "class",
        "size",
        "bbox_area",
        "bbox_area_px2",
        "bbox_area_fraction",
        "clean_confidence",
        "clean_confidence_bucket",
        "image_density",
        "image_density_bucket",
        "class_size",
        "condition_pairs",
        "learned_attacked_count",
        "random_attacked_count",
        "learned_asr",
        "random_asr",
        "paired_asr_gain",
    ]
    write_csv(args.output_root / "object_attackability.csv", all_records, object_fields)
    write_csv(args.output_root / "bucket_attackability.csv", all_buckets)
    manifest = {
        "protocol": "vehicle4_v2_clean_prediction_attackability_stratification",
        "models": MODELS,
        "asr_confidence_threshold": args.confidence,
        "asr_iou_threshold": args.iou,
        "matching": "greedy same-category IoU matching, identical to evaluate_shared.calculate_asr",
        "condition_pairs_per_model": 15,
        "dimensions": DIMENSIONS,
        "size_buckets": {"small": "area < 32^2 px2", "medium": "32^2 <= area < 96^2 px2", "large": "area >= 96^2 px2"},
        "bbox_area_buckets": [
            "<100 px2",
            "100–<250 px2",
            "250–<500 px2",
            "500–<1,000 px2",
            "1,000–<2,500 px2",
            ">=2,500 px2",
        ],
        "confidence_buckets": ["0.40–<0.50", "0.50–<0.60", "0.60–<0.70", "0.70–<0.80", "0.80–<0.90", ">=0.90"],
        "density_buckets": [
            "1–10 detections/image",
            "11–20 detections/image",
            "21–30 detections/image",
            "31–40 detections/image",
            ">40 detections/image",
        ],
        "reference_unit": "detector clean prediction, not ground-truth object; this matches the formal ASR protocol",
        "learned_asr": "learned attacked trials / clean reference trials",
        "random_asr": "random attacked trials / clean reference trials",
        "gain": "learned_asr - random_asr",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "README.md").write_text(
        "# Vehicle4 v2 attackability stratification\n\n"
        "bucket_attackability.csv contains pooled Learned ASR, Random ASR, and Learned minus Random ASR gain for every model and bucket. "
        "object_attackability.csv contains one row per clean reference detection and its 15 paired condition outcomes. "
        "ASR uses clean confidence >= 0.4 and same-class attacked IoU >= 0.5, matching the formal evaluator.\n",
        encoding="utf-8",
    )
    print(args.output_root / "bucket_attackability.csv")


if __name__ == "__main__":
    main()
