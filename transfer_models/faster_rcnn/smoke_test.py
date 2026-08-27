from pathlib import Path

import torch
from PIL import Image

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms.functional import pil_to_tensor


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Faster R-CNN
# --------------------------------------------------

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.to(device)
model.eval()


# --------------------------------------------------
# Find one VisDrone validation image
# --------------------------------------------------

image_dir = Path(
    "data/visdrone_data/VisDrone2019-DET-val/images"
)

images = sorted(image_dir.glob("*.jpg"))

if not images:
    raise RuntimeError(
        f"No JPG images found in {image_dir}"
    )

image_path = images[0]

print("Image:", image_path)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = Image.open(image_path).convert("RGB")

image_tensor = (
    pil_to_tensor(image).float() / 255.0
)

image_tensor = image_tensor.to(device)


# --------------------------------------------------
# Inference
# --------------------------------------------------

with torch.no_grad():
    output = model([image_tensor])[0]


boxes = output["boxes"].cpu()
labels = output["labels"].cpu()
scores = output["scores"].cpu()


# --------------------------------------------------
# COCO vehicle classes
#
# car   = 3
# bus   = 6
# truck = 8
# --------------------------------------------------

vehicle_classes = {
    3: "car",
    6: "bus",
    8: "truck",
}

conf_thresh = 0.4


print()
print("=" * 60)
print("Vehicle detections")
print("=" * 60)

count = 0

for box, label, score in zip(
    boxes,
    labels,
    scores,
):

    label_id = int(label)
    score_value = float(score)

    if (
        label_id in vehicle_classes
        and score_value >= conf_thresh
    ):

        count += 1

        x1, y1, x2, y2 = box.tolist()

        print(
            f"{count:3d} | "
            f"{vehicle_classes[label_id]:5s} | "
            f"score={score_value:.3f} | "
            f"box=({x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f})"
        )


print()
print("Total vehicle detections:", count)
