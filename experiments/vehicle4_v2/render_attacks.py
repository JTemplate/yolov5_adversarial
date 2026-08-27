#!/usr/bin/env python3
"""Render model-independent clean/random/learned inputs from ground-truth boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer


EVALUATION_TRANSFORM_SEEDS = [7101, 7102, 7103, 7104, 7105]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/visdrone_vehicle4_v2"))
    parser.add_argument("--patch-registry", type=Path, default=Path("runs/vehicle4_v2/patch_registry.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/vehicle4_v2/rendered"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def padded_image_and_labels(image_path: Path, label_path: Path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    square = max(width, height)
    left = (square - width) // 2
    top = (square - height) // 2
    padded = Image.new("RGB", (square, square), color=(127, 127, 127))
    padded.paste(image, (left, top))
    padded = padded.resize((640, 640), Image.Resampling.BILINEAR)

    labels = []
    if label_path.stat().st_size:
        for row in label_path.read_text(encoding="utf-8").splitlines():
            category, x_center, y_center, box_width, box_height = map(float, row.split())
            labels.append(
                [
                    category,
                    (x_center * width + left) / square,
                    (y_center * height + top) / square,
                    box_width * width / square,
                    box_height * height / square,
                ]
            )
    return padded, torch.tensor(labels, dtype=torch.float32).reshape(-1, 5)


def apply_patch_chunked(
    image_tensor: torch.Tensor,
    patch: torch.Tensor,
    labels: torch.Tensor,
    transformer: PatchTransformer,
    applier: PatchApplier,
    transform_seed: int,
) -> torch.Tensor:
    if not len(labels):
        return image_tensor.clone()
    seed_all(transform_seed)
    result = image_tensor.unsqueeze(0)
    for start in range(0, len(labels), 32):
        label_chunk = labels[start : start + 32].unsqueeze(0).to(image_tensor.device)
        transformed = transformer(
            patch,
            label_chunk,
            (640, 640),
            use_mul_add_gau=True,
            do_transforms=True,
            do_rotate=True,
            rand_loc=True,
        )
        result = applier(result, transformed)
    return result.squeeze(0)


def save_tensor(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.detach().cpu().clamp(0, 1)).save(destination, compress_level=2)


def main() -> int:
    args = parse_args()
    registry = json.loads(args.patch_registry.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)

    image_lines = args.dataset_root.joinpath("official_val.txt").read_text(encoding="utf-8").splitlines()
    image_paths = [args.dataset_root / line.removeprefix("./") for line in image_lines if line.strip()]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    transformer = PatchTransformer(
        t_size_frac=(0.25, 0.40),
        mul_gau_mean=(0.5, 0.8),
        mul_gau_std=0.1,
        x_off_loc=(-0.25, 0.25),
        y_off_loc=(-0.25, 0.25),
        dev=device,
    ).to(device)
    applier = PatchApplier(1.0).to(device)

    clean_dir = args.output_root / "conditions" / "clean" / "images"
    shared_label_dir = args.output_root / "labels"
    padded_cache = []
    for image_path in image_paths:
        label_path = args.dataset_root / "labels" / "official_val" / f"{image_path.stem}.txt"
        padded, labels = padded_image_and_labels(image_path, label_path)
        image_tensor = TF.pil_to_tensor(padded).float().div(255.0).to(device)
        padded_cache.append((image_path, image_tensor, labels))
        clean_path = clean_dir / f"{image_path.stem}.png"
        if not clean_path.exists():
            save_tensor(image_tensor, clean_path)
        label_rows = [" ".join(f"{value:.8f}" for value in row.tolist()) + "\n" for row in labels]
        shared_label_dir.mkdir(parents=True, exist_ok=True)
        shared_label_dir.joinpath(f"{image_path.stem}.txt").write_text("".join(label_rows), encoding="utf-8")

    conditions = [{"name": "clean", "kind": "clean"}]
    for patch_record in registry["patches"]:
        patch_seed = int(patch_record["seed"])
        patch_path = Path(patch_record["patch"])
        learned_patch = TF.pil_to_tensor(Image.open(patch_path).convert("RGB")).float().div(255.0).to(device)
        generator = torch.Generator(device="cpu").manual_seed(900000 + patch_seed)
        random_patch = torch.rand(learned_patch.shape, generator=generator).to(device)
        for repeat_index, evaluation_seed in enumerate(EVALUATION_TRANSFORM_SEEDS):
            pair_name = f"patch{patch_seed}_transform{repeat_index}"
            condition_dirs = {
                "learned": args.output_root / "conditions" / f"{pair_name}_learned" / "images",
                "random": args.output_root / "conditions" / f"{pair_name}_random" / "images",
            }
            for kind, directory in condition_dirs.items():
                conditions.append(
                    {
                        "name": directory.parent.name,
                        "kind": kind,
                        "patch_seed": patch_seed,
                        "transform_index": repeat_index,
                        "transform_seed": evaluation_seed,
                        "patch_sha256": patch_record["sha256"] if kind == "learned" else None,
                        "random_patch_seed": 900000 + patch_seed if kind == "random" else None,
                    }
                )
            for image_index, (image_path, image_tensor, labels) in enumerate(padded_cache):
                per_image_seed = evaluation_seed * 100000 + image_index
                learned = apply_patch_chunked(
                    image_tensor, learned_patch, labels, transformer, applier, per_image_seed
                )
                random_image = apply_patch_chunked(
                    image_tensor, random_patch, labels, transformer, applier, per_image_seed
                )
                save_tensor(learned, condition_dirs["learned"] / f"{image_path.stem}.png")
                save_tensor(random_image, condition_dirs["random"] / f"{image_path.stem}.png")

    manifest = {
        "protocol": "vehicle4_v2_shared_ground_truth_rendering",
        "created_at": time.time(),
        "dataset_manifest_sha256": sha256(args.dataset_root / "dataset_manifest.json"),
        "patch_registry_sha256": sha256(args.patch_registry),
        "images": len(image_paths),
        "input_size": [640, 640],
        "lossless_format": "PNG",
        "ground_truth_placement": True,
        "transform_seeds": EVALUATION_TRANSFORM_SEEDS,
        "conditions": conditions,
    }
    args.output_root.joinpath("conditions.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root / "conditions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
