#!/usr/bin/env python3
"""Fine-tune Faster R-CNN on the audited Vehicle4 v2 split."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

CLASS_NAMES = ["car", "van", "truck", "bus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/visdrone_vehicle4_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vehicle4_v2/detectors/fasterrcnn_seed0"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Vehicle4Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        training: bool,
        max_images: int | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self.training = training
        lines = root.joinpath(f"{split}.txt").read_text(encoding="utf-8").splitlines()
        self.image_paths = [root / line.removeprefix("./") for line in lines if line.strip()]
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        square_size = max(original_width, original_height)
        left = (square_size - original_width) // 2
        top = (square_size - original_height) // 2
        padded = Image.new("RGB", (square_size, square_size), color=(127, 127, 127))
        padded.paste(image, (left, top))
        padded = padded.resize((640, 640), Image.Resampling.BILINEAR)

        label_path = self.root / "labels" / self.split / f"{image_path.stem}.txt"
        boxes = []
        labels = []
        if label_path.stat().st_size:
            for row in label_path.read_text(encoding="utf-8").splitlines():
                category, x_center, y_center, width, height = map(float, row.split())
                x1 = (x_center - width / 2.0) * original_width
                y1 = (y_center - height / 2.0) * original_height
                x2 = (x_center + width / 2.0) * original_width
                y2 = (y_center + height / 2.0) * original_height
                scale = 640.0 / square_size
                boxes.append(
                    [
                        (x1 + left) * scale,
                        (y1 + top) * scale,
                        (x2 + left) * scale,
                        (y2 + top) * scale,
                    ]
                )
                labels.append(int(category) + 1)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        image_tensor = TF.pil_to_tensor(padded).float().div(255.0)
        if self.training and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            if len(boxes_tensor):
                old_x1 = boxes_tensor[:, 0].clone()
                old_x2 = boxes_tensor[:, 2].clone()
                boxes_tensor[:, 0] = 640.0 - old_x2
                boxes_tensor[:, 2] = 640.0 - old_x1

        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes_tensor)
            else torch.zeros(0, dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(boxes_tensor), dtype=torch.int64),
        }
        return image_tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=640,
        max_size=640,
        trainable_backbone_layers=5,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, len(CLASS_NAMES) + 1)
    return model


def coco_metrics(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    images_json = []
    annotations_json = []
    predictions_json = []
    annotation_id = 1
    with torch.no_grad():
        for images, targets in loader:
            images_device = [image.to(device, non_blocking=True) for image in images]
            outputs = model(images_device)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                images_json.append({"id": image_id, "width": 640, "height": 640})
                for box, label, area in zip(target["boxes"], target["labels"], target["area"]):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    annotations_json.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": float(area),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu()]
                    predictions_json.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label.detach().cpu()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score.detach().cpu()),
                        }
                    )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_predictions = coco_gt.loadRes(predictions_json)
        evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
    }


def atomic_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    train_dataset = Vehicle4Dataset(args.dataset_root, "train", training=True, max_images=args.max_train_images)
    val_dataset = Vehicle4Dataset(args.dataset_root, "internal_val", training=False, max_images=args.max_val_images)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=args.workers > 1,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_map = -math.inf
    stale_epochs = 0
    last_path = args.output_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint["best_map50_95"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        loss_totals = Counter()
        batches = 0
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                losses = model(images, targets)
                total_loss = sum(losses.values())
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                loss_totals[name] += float(value.detach().cpu())
            loss_totals["total"] += float(total_loss.detach().cpu())
            batches += 1

        validation = coco_metrics(model, val_loader, device)
        scheduler.step()
        improved = validation["map50_95"] > best_map + 1e-6
        if improved:
            best_map = validation["map50_95"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_losses": {name: value / max(batches, 1) for name, value in loss_totals.items()},
            "validation": validation,
            "best_map50_95": best_map,
            "stale_epochs": stale_epochs,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_map50_95": best_map,
            "stale_epochs": stale_epochs,
            "class_names": CLASS_NAMES,
            "input_size": 640,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        atomic_save(checkpoint, last_path)
        if improved:
            atomic_save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs", flush=True)
            break

    args.output_dir.joinpath("complete.json").write_text(
        json.dumps({"best_map50_95": best_map, "completed_at": time.time()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
