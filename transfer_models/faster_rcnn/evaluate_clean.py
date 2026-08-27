import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path("data/visdrone_data/VisDrone2019-DET-val/images")

OUTPUT_DIR = Path("transfer_models/faster_rcnn/results")

OUTPUT_JSON = OUTPUT_DIR / "clean_detections.json"
SUMMARY_JSON = OUTPUT_DIR / "clean_summary.json"

CONF_THRESH = 0.4


# COCO category IDs
VEHICLE_CLASSES = {
    3: "car",
    6: "bus",
    8: "truck",
}


# ============================================================
# Device
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# Model
# ============================================================

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(weights=weights)

model.to(device)
model.eval()


# ============================================================
# Images
# ============================================================

image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print("Images:", len(image_paths))
print("Confidence threshold:", CONF_THRESH)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Evaluation
# ============================================================

all_results = []

total_detections = 0

class_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
}

score_sum = 0.0

start_time = time.time()


for image_path in tqdm(image_paths, desc="Evaluating clean images"):
    image = Image.open(image_path).convert("RGB")

    image_tensor = pil_to_tensor(image).float() / 255.0

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    detections = []

    for box, label, score in zip(
        boxes,
        labels,
        scores,
    ):
        label_id = int(label)
        score_value = float(score)

        if label_id not in VEHICLE_CLASSES:
            continue

        if score_value < CONF_THRESH:
            continue

        class_name = VEHICLE_CLASSES[label_id]

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]

        detections.append(
            {
                "class_id": label_id,
                "class_name": class_name,
                "score": score_value,
                "bbox_xyxy": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

        total_detections += 1

        class_counts[class_name] += 1

        score_sum += score_value

    all_results.append(
        {
            "image": image_path.name,
            "width": image.width,
            "height": image.height,
            "detections": detections,
        }
    )


elapsed = time.time() - start_time


# ============================================================
# Statistics
# ============================================================

num_images = len(image_paths)

avg_detections = total_detections / num_images

avg_score = score_sum / total_detections if total_detections > 0 else 0.0


summary = {
    "model": "fasterrcnn_resnet50_fpn",
    "weights": "COCO DEFAULT",
    "num_images": num_images,
    "confidence_threshold": CONF_THRESH,
    "total_vehicle_detections": total_detections,
    "average_vehicle_detections_per_image": avg_detections,
    "average_detection_score": avg_score,
    "class_counts": class_counts,
    "elapsed_seconds": elapsed,
}


# ============================================================
# Save
# ============================================================

with open(OUTPUT_JSON, "w") as f:
    json.dump(
        all_results,
        f,
        indent=2,
    )


with open(SUMMARY_JSON, "w") as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Print
# ============================================================

print()
print("=" * 70)
print("Faster R-CNN Clean Baseline")
print("=" * 70)

print(f"Images                : {num_images}")

print(f"Vehicle detections    : {total_detections}")

print(f"Detections / image    : {avg_detections:.2f}")

print(f"Average confidence    : {avg_score:.3f}")

print()

for class_name, count in class_counts.items():
    print(f"{class_name:5s} detections        : {count}")


print()
print(f"Elapsed               : {elapsed:.1f} s")

print()
print(
    "Detection JSON        :",
    OUTPUT_JSON,
)

print(
    "Summary JSON          :",
    SUMMARY_JSON,
)
