#!/usr/bin/env python3
"""Build the audited four-vehicle VisDrone dataset used by the v2 study.

The original VisDrone category ids are car=4, van=5, truck=6, bus=9.
This script creates a new, versioned dataset and never mutates the legacy
``data/visdrone_data`` labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASS_MAP = {4: 0, 5: 1, 6: 2, 9: 3}
CLASS_NAMES = ["car", "van", "truck", "bus"]
SPLIT_SEED = 20260825


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/visdrone_data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/visdrone_vehicle4_v2"))
    parser.add_argument("--internal-val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_visdrone(
    annotation_path: Path,
) -> tuple[list[tuple[int, int, int, int, int]], int]:
    records = []
    seen = set()
    duplicates_removed = 0
    with annotation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().strip(",").split(",")
            if not line.strip():
                continue
            if len(fields) < 8:
                raise ValueError(f"{annotation_path}:{line_number}: expected 8 fields")
            x, y, width, height, score, category, _, occlusion = map(int, fields[:8])
            if score != 1 or occlusion > 2 or category not in CLASS_MAP:
                continue
            if width <= 0 or height <= 0:
                continue
            record = (CLASS_MAP[category], x, y, width, height)
            if record in seen:
                duplicates_removed += 1
                continue
            seen.add(record)
            records.append(record)
    return records, duplicates_removed


def inventory(split_root: Path) -> dict[str, dict[str, object]]:
    images = {path.stem: path for path in split_root.joinpath("images").glob("*") if path.is_file()}
    annotations = {path.stem: path for path in split_root.joinpath("annotations").glob("*.txt")}
    if set(images) != set(annotations):
        missing_images = sorted(set(annotations) - set(images))[:5]
        missing_annotations = sorted(set(images) - set(annotations))[:5]
        raise RuntimeError(
            f"Image/annotation mismatch in {split_root}: "
            f"missing_images={missing_images}, missing_annotations={missing_annotations}"
        )

    result: dict[str, dict[str, object]] = {}
    for stem in sorted(images):
        image_path = images[stem]
        with Image.open(image_path) as image:
            width, height = image.size
        boxes, duplicates_removed = read_visdrone(annotations[stem])
        result[stem] = {
            "image": image_path.resolve(),
            "annotation": annotations[stem].resolve(),
            "width": width,
            "height": height,
            "boxes": boxes,
            "duplicates_removed": duplicates_removed,
            "group": stem.split("_", 1)[0],
        }
    return result


def split_score(
    selected: set[str],
    group_vectors: dict[str, tuple[int, Counter]],
    target_images: float,
    target_classes: list[float],
) -> float:
    image_count = sum(group_vectors[group][0] for group in selected)
    class_count = Counter()
    for group in selected:
        class_count.update(group_vectors[group][1])
    image_error = ((image_count - target_images) / max(target_images, 1.0)) ** 2
    class_error = sum(
        ((class_count[index] - target_classes[index]) / max(target_classes[index], 1.0)) ** 2
        for index in range(len(CLASS_NAMES))
    )
    return 2.0 * image_error + class_error / len(CLASS_NAMES)


def count_score(
    image_count: int,
    class_count: Counter,
    target_images: float,
    target_classes: list[float],
) -> float:
    image_error = ((image_count - target_images) / max(target_images, 1.0)) ** 2
    class_error = sum(
        ((class_count[index] - target_classes[index]) / max(target_classes[index], 1.0)) ** 2
        for index in range(len(CLASS_NAMES))
    )
    return 2.0 * image_error + class_error / len(CLASS_NAMES)


def choose_validation_groups(records: dict[str, dict[str, object]], fraction: float, seed: int) -> set[str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records.values():
        grouped[str(record["group"])].append(record)

    group_vectors: dict[str, tuple[int, Counter]] = {}
    total_classes = Counter()
    for group, items in grouped.items():
        counts = Counter(box[0] for item in items for box in item["boxes"])
        group_vectors[group] = (len(items), counts)
        total_classes.update(counts)

    target_images = len(records) * fraction
    target_classes = [total_classes[index] * fraction for index in range(len(CLASS_NAMES))]
    groups = sorted(grouped)
    rng = random.Random(seed)

    # Randomized search is deterministic under ``seed`` and handles a few very
    # large video groups better than a simple first-fit split.
    best: set[str] | None = None
    best_score = float("inf")
    for _ in range(5000):
        order = groups.copy()
        rng.shuffle(order)
        candidate: set[str] = set()
        candidate_images = 0
        candidate_classes = Counter()
        for group in order:
            group_images, group_classes = group_vectors[group]
            current_score = count_score(candidate_images, candidate_classes, target_images, target_classes)
            proposed_classes = candidate_classes + group_classes
            proposed_score = count_score(
                candidate_images + group_images,
                proposed_classes,
                target_images,
                target_classes,
            )
            if proposed_score < current_score or candidate_images < 0.8 * target_images:
                candidate.add(group)
                candidate_images += group_images
                candidate_classes = proposed_classes
        score = count_score(candidate_images, candidate_classes, target_images, target_classes)
        if score < best_score:
            best, best_score = candidate, score

    if not best:
        raise RuntimeError("Could not construct a group-aware internal validation split")
    return best


def relative_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def write_yolo_label(path: Path, record: dict[str, object]) -> Counter:
    width = int(record["width"])
    height = int(record["height"])
    counts = Counter()
    rows = []
    for category, x, y, box_width, box_height in record["boxes"]:
        x_center = (x + box_width / 2.0) / width
        y_center = (y + box_height / 2.0) / height
        rows.append(f"{category} {x_center:.8f} {y_center:.8f} {box_width / width:.8f} {box_height / height:.8f}\n")
        counts[category] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows), encoding="utf-8")
    return counts


def write_dataset_yaml(output_root: Path) -> None:
    content = """# Generated by experiments/vehicle4_v2/prepare_dataset.py
path: data/visdrone_vehicle4_v2
train: train.txt
val: internal_val.txt
test: official_val.txt
nc: 4
names:
  0: car
  1: van
  2: truck
  3: bus
"""
    output_root.joinpath("dataset.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if output_root.exists():
        print(
            f"Refusing to overwrite existing dataset: {output_root}. Move it aside explicitly before rebuilding.",
            file=sys.stderr,
        )
        return 2
    if not 0.05 <= args.internal_val_fraction <= 0.30:
        raise ValueError("internal-val-fraction must be between 0.05 and 0.30")

    train_records = inventory(args.source_root / "VisDrone2019-DET-train")
    official_records = inventory(args.source_root / "VisDrone2019-DET-val")
    validation_groups = choose_validation_groups(train_records, args.internal_val_fraction, args.split_seed)

    split_records = {"train": {}, "internal_val": {}, "official_val": official_records}
    for stem, record in train_records.items():
        split = "internal_val" if record["group"] in validation_groups else "train"
        split_records[split][stem] = record

    output_root.mkdir(parents=True)
    class_rows = []
    membership_rows = []
    for split, records in split_records.items():
        split_counts = Counter()
        list_lines = []
        for stem, record in sorted(records.items()):
            image_source = Path(record["image"])
            image_destination = output_root / "images" / split / image_source.name
            label_destination = output_root / "labels" / split / f"{stem}.txt"
            relative_link(image_source, image_destination)
            split_counts.update(write_yolo_label(label_destination, record))
            list_lines.append(f"./images/{split}/{image_source.name}\n")
            membership_rows.append(
                {
                    "image": image_source.name,
                    "split": split,
                    "group": record["group"],
                    "objects": len(record["boxes"]),
                }
            )
        output_root.joinpath(f"{split}.txt").write_text("".join(list_lines), encoding="utf-8")
        class_rows.append(
            {
                "split": split,
                "images": len(records),
                "duplicates_removed": sum(int(record["duplicates_removed"]) for record in records.values()),
                **{CLASS_NAMES[index]: split_counts[index] for index in range(len(CLASS_NAMES))},
                "total_objects": sum(split_counts.values()),
            }
        )

    with output_root.joinpath("class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)
    with output_root.joinpath("split_membership.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=membership_rows[0].keys())
        writer.writeheader()
        writer.writerows(membership_rows)

    write_dataset_yaml(output_root)
    manifest = {
        "dataset_version": "vehicle4_v2",
        "source_root": str(args.source_root.resolve()),
        "class_map_visdrone_to_v2": {str(key): value for key, value in CLASS_MAP.items()},
        "class_names": CLASS_NAMES,
        "filters": {
            "score": 1,
            "maximum_occlusion": 2,
            "deduplicate_exact_boxes": True,
        },
        "split_seed": args.split_seed,
        "split_method": "video-group-aware randomized optimization",
        "internal_val_fraction_requested": args.internal_val_fraction,
        "validation_groups": sorted(validation_groups),
        "counts": class_rows,
        "files": {
            name: {"sha256": sha256(output_root / name)}
            for name in [
                "dataset.yaml",
                "train.txt",
                "internal_val.txt",
                "official_val.txt",
                "class_counts.csv",
                "split_membership.csv",
            ]
        },
    }
    output_root.joinpath("dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    train_groups = {str(record["group"]) for record in split_records["train"].values()}
    internal_groups = {str(record["group"]) for record in split_records["internal_val"].values()}
    if train_groups & internal_groups:
        raise AssertionError("Group leakage detected")
    if sum(row["images"] for row in class_rows) != len(train_records) + len(official_records):
        raise AssertionError("Image counts do not reconcile")
    if any(row[name] == 0 for row in class_rows for name in CLASS_NAMES):
        raise AssertionError("At least one split is missing a target class")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote audited dataset to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
