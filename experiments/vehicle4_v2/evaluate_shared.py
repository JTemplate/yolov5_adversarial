#!/usr/bin/env python3
"""Run one detector over every shared rendered condition and score it."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as TF

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

from experiments.vehicle4_v2.train_fasterrcnn import CLASS_NAMES, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov5s", "yolov5m", "fasterrcnn"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=Path("runs/vehicle4_v2/rendered"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/vehicle4_v2/evaluation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--asr-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--asr-iou", type=float, default=0.5)
    return parser.parse_args()


def build_ground_truth(render_root: Path) -> tuple[Path, list[Path]]:
    image_paths = sorted(render_root.joinpath("conditions/clean/images").glob("*.png"))
    ground_truth_path = render_root / "ground_truth_coco.json"
    if ground_truth_path.exists():
        return ground_truth_path, image_paths
    images = []
    annotations = []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths):
        images.append({"id": image_id, "file_name": image_path.name, "width": 640, "height": 640})
        label_path = render_root / "labels" / f"{image_path.stem}.txt"
        if label_path.stat().st_size:
            for row in label_path.read_text(encoding="utf-8").splitlines():
                category, x_center, y_center, width, height = map(float, row.split())
                width_pixels = width * 640.0
                height_pixels = height * 640.0
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(category) + 1,
                        "bbox": [
                            x_center * 640.0 - width_pixels / 2.0,
                            y_center * 640.0 - height_pixels / 2.0,
                            width_pixels,
                            height_pixels,
                        ],
                        "area": width_pixels * height_pixels,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
    }
    # All three detector evaluators start concurrently. Publish the shared
    # ground-truth file atomically so another evaluator can never observe a
    # partially written JSON document.
    temporary_path = ground_truth_path.with_name(
        f".{ground_truth_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary_path, ground_truth_path)
    return ground_truth_path, image_paths


def load_detector(args: argparse.Namespace, device: torch.device):
    if args.model.startswith("yolov5"):
        return DetectMultiBackend(str(args.weights), device=device, dnn=False, data=None, fp16=False).eval()
    checkpoint = torch.load(args.weights, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def infer_yolo(model, image_paths, device, batch_size, conf_threshold, nms_iou):
    predictions = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = [TF.pil_to_tensor(Image.open(path).convert("RGB")).float().div(255.0) for path in batch_paths]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            raw = model(batch)[0]
            outputs = non_max_suppression(
                raw, conf_thres=conf_threshold, iou_thres=nms_iou, max_det=300
            )
        for offset, output in enumerate(outputs):
            image_id = start + offset
            for x1, y1, x2, y2, score, category in output.detach().cpu().tolist():
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
    return predictions


def infer_fasterrcnn(model, image_paths, device, batch_size):
    predictions = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = [
            TF.pil_to_tensor(Image.open(path).convert("RGB")).float().div(255.0).to(device)
            for path in batch_paths
        ]
        with torch.no_grad():
            outputs = model(tensors)
        for offset, output in enumerate(outputs):
            image_id = start + offset
            for box, category, score in zip(output["boxes"], output["labels"], output["scores"]):
                x1, y1, x2, y2 = [float(value) for value in box.detach().cpu()]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category.detach().cpu()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.detach().cpu()),
                    }
                )
    return predictions


def score_coco(coco_gt: COCO, predictions: list[dict]) -> dict[str, object]:
    if not predictions:
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0, "per_class_map50_95": {name: 0.0 for name in CLASS_NAMES}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_predictions = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    result = {
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "per_class_map50_95": {},
    }
    for category_id, name in enumerate(CLASS_NAMES, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            per_class = COCOeval(coco_gt, coco_predictions, "bbox")
            per_class.params.catIds = [category_id]
            per_class.evaluate()
            per_class.accumulate()
            per_class.summarize()
        result["per_class_map50_95"][name] = float(per_class.stats[0])
    return result


def box_iou(box_a, box_b) -> float:
    ax1, ay1, width_a, height_a = box_a
    bx1, by1, width_b, height_b = box_b
    ax2, ay2 = ax1 + width_a, ay1 + height_a
    bx2, by2 = bx1 + width_b, by1 + height_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = width_a * height_a + width_b * height_b - intersection
    return intersection / union if union > 0 else 0.0


def calculate_asr(clean_predictions, attacked_predictions, confidence_threshold, iou_threshold):
    clean_by_image = defaultdict(list)
    attacked_by_image = defaultdict(list)
    for prediction in clean_predictions:
        if prediction["score"] >= confidence_threshold:
            clean_by_image[prediction["image_id"]].append(prediction)
    for prediction in attacked_predictions:
        if prediction["score"] >= confidence_threshold:
            attacked_by_image[prediction["image_id"]].append(prediction)
    total_clean = 0
    survived = 0
    per_image = []
    for image_id, clean_items in clean_by_image.items():
        attacked_items = attacked_by_image[image_id]
        used = set()
        image_survived = 0
        for clean in clean_items:
            candidates = [
                (box_iou(clean["bbox"], attacked["bbox"]), index)
                for index, attacked in enumerate(attacked_items)
                if index not in used and attacked["category_id"] == clean["category_id"]
            ]
            if candidates:
                best_iou, best_index = max(candidates)
                if best_iou >= iou_threshold:
                    used.add(best_index)
                    survived += 1
                    image_survived += 1
        total_clean += len(clean_items)
        per_image.append(
            {
                "image_id": image_id,
                "clean_detections": len(clean_items),
                "survived": image_survived,
                "attacked": len(clean_items) - image_survived,
            }
        )
    return {
        "asr": 1.0 - survived / total_clean if total_clean else None,
        "clean_reference_detections": total_clean,
        "survived": survived,
        "per_image": per_image,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ground_truth_path, clean_image_paths = build_ground_truth(args.render_root)
    coco_gt = COCO(str(ground_truth_path))
    manifest = json.loads(args.render_root.joinpath("conditions.json").read_text(encoding="utf-8"))
    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_detector(args, device)
    condition_results = {}
    clean_predictions = None
    ordered_conditions = sorted(manifest["conditions"], key=lambda item: item["name"] != "clean")
    for condition in ordered_conditions:
        name = condition["name"]
        condition_output = output_dir / name
        condition_output.mkdir(parents=True, exist_ok=True)
        predictions_path = condition_output / "predictions.json"
        metrics_path = condition_output / "metrics.json"
        if predictions_path.exists():
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        else:
            image_paths = [
                args.render_root / "conditions" / name / "images" / clean_path.name
                for clean_path in clean_image_paths
            ]
            started = time.time()
            if args.model.startswith("yolov5"):
                predictions = infer_yolo(
                    model,
                    image_paths,
                    device,
                    args.batch_size,
                    args.confidence_threshold,
                    args.nms_iou,
                )
            else:
                predictions = infer_fasterrcnn(model, image_paths, device, args.batch_size)
            predictions_path.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
            condition["inference_seconds"] = time.time() - started
        if name == "clean":
            clean_predictions = predictions
        if clean_predictions is None:
            raise RuntimeError("Clean condition must be evaluated first")
        metrics = score_coco(coco_gt, predictions)
        asr = None if name == "clean" else calculate_asr(
            clean_predictions,
            predictions,
            args.asr_confidence_threshold,
            args.asr_iou,
        )
        payload = {"model": args.model, "condition": condition, "metrics": metrics, "asr": asr}
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        condition_results[name] = payload
        print(json.dumps({"condition": name, "map50_95": metrics["map50_95"], "asr": asr["asr"] if asr else None}), flush=True)

    clean_map = condition_results["clean"]["metrics"]["map50_95"]
    for name, result in condition_results.items():
        attack_map = result["metrics"]["map50_95"]
        result["relative_map_drop"] = (clean_map - attack_map) / clean_map if clean_map > 0 else None
    output_dir.joinpath("summary.json").write_text(
        json.dumps({"model": args.model, "weights": str(args.weights), "conditions": condition_results}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
