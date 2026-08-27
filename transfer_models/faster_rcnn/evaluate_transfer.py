import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageOps
from tqdm import tqdm

from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from adv_patch_gen.utils.patch import (
    PatchTransformer,
    PatchApplier,
)


# ============================================================
# Configuration inherited from original YOLOv5 Patch pipeline
# ============================================================

MODEL_IN_SIZE = (640, 640)

PATCH_SIZE = (64, 64)

TARGET_SIZE_FRAC = (0.25, 0.40)

MUL_GAU_MEAN = (0.5, 0.8)
MUL_GAU_STD = 0.1

X_OFF_LOC = (-0.25, 0.25)
Y_OFF_LOC = (-0.25, 0.25)

PATCH_ALPHA = 1.0

CONF_THRESH = 0.4
IOU_THRESH = 0.5


# COCO vehicle categories used by Faster R-CNN
VEHICLE_CLASSES = {
    3: "car",
    6: "bus",
    8: "truck",
}


# ============================================================
# Utility
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_to_square(image):
    """
    Centre-pad image to square.

    The original project uses pad_to_square before resizing to 640x640.
    Gray padding is used here to match the visual behaviour of the
    original pipeline.
    """

    width, height = image.size

    if width == height:
        return image

    size = max(width, height)

    pad_left = (size - width) // 2
    pad_right = size - width - pad_left

    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top

    return ImageOps.expand(
        image,
        border=(
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ),
        fill=(127, 127, 127),
    )


def box_iou(box_a, box_b):
    """
    xyxy IoU
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Faster R-CNN inference
# ============================================================

def detect_vehicles(model, image_tensor, device):
    with torch.no_grad():
        result = model(
            [image_tensor.to(device)]
        )[0]

    detections = []

    boxes = result["boxes"].detach().cpu()
    labels = result["labels"].detach().cpu()
    scores = result["scores"].detach().cpu()

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

        detections.append(
            {
                "class_id": label_id,
                "class_name": VEHICLE_CLASSES[
                    label_id
                ],
                "score": score_value,
                "bbox": [
                    float(v)
                    for v in box.tolist()
                ],
            }
        )

    return detections


# ============================================================
# ASR matching
# ============================================================

def count_surviving_clean_detections(
    clean_detections,
    attacked_detections,
):
    """
    Class-aware greedy matching.

    A clean detection survives when an attacked prediction:

    1. has the same Faster R-CNN class
    2. IoU >= 0.5

    Otherwise the clean reference detection is treated
    as successfully attacked.
    """

    used = set()

    survived = 0

    for clean_det in clean_detections:

        best_iou = 0.0
        best_idx = None

        for idx, attacked_det in enumerate(
            attacked_detections
        ):

            if idx in used:
                continue

            if (
                attacked_det["class_id"]
                != clean_det["class_id"]
            ):
                continue

            iou = box_iou(
                clean_det["bbox"],
                attacked_det["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if (
            best_idx is not None
            and best_iou >= IOU_THRESH
        ):
            survived += 1
            used.add(best_idx)

    return survived


# ============================================================
# Load YOLOv5 source detections
# ============================================================

def load_source_clean_results(path):
    with open(path, "r") as f:
        data = json.load(f)

    grouped = defaultdict(list)

    for det in data:

        image_id = str(det["image_id"])

        x, y, w, h = det["bbox"]

        grouped[image_id].append(
            {
                "class_id": int(
                    det["category_id"]
                ),
                "bbox_xywh": [
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                ],
            }
        )

    return grouped


def source_detections_to_labels(
    detections,
):
    """
    Convert YOLO clean_results boxes:

        x, y, w, h in 640x640 pixel coordinates

    into the label format expected by PatchTransformer:

        class, x_center, y_center, width, height

    normalized to 0..1.
    """

    labels = []

    for det in detections:

        x, y, w, h = det["bbox_xywh"]

        x_center = x + w / 2.0
        y_center = y + h / 2.0

        labels.append(
            [
                det["class_id"],
                x_center / 640.0,
                y_center / 640.0,
                w / 640.0,
                h / 640.0,
            ]
        )

    if not labels:
        return None

    return torch.tensor(
        labels,
        dtype=torch.float32,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patch",
        required=True,
        help="YOLOv5-trained e_10.png",
    )

    parser.add_argument(
        "--source-clean-json",
        required=True,
        help=(
            "YOLOv5 clean_results.json "
            "used for patch placement"
        ),
    )

    parser.add_argument(
        "--image-dir",
        default=(
            "data/visdrone_data/"
            "VisDrone2019-DET-val/images"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "transfer_models/faster_rcnn/"
            "results/transfer"
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Faster R-CNN
    # --------------------------------------------------------

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Original PatchTransformer / PatchApplier
    # --------------------------------------------------------

    patch_transformer = PatchTransformer(
        TARGET_SIZE_FRAC,
        MUL_GAU_MEAN,
        MUL_GAU_STD,
        X_OFF_LOC,
        Y_OFF_LOC,
        device,
    ).to(device)

    patch_applier = PatchApplier(
        PATCH_ALPHA
    ).to(device)

    # --------------------------------------------------------
    # Learned patch
    # --------------------------------------------------------

    patch_image = Image.open(
        args.patch
    ).convert("RGB")

    patch_image = transforms.Resize(
        PATCH_SIZE
    )(patch_image)

    learned_patch = transforms.ToTensor()(
        patch_image
    ).to(device)

    # --------------------------------------------------------
    # Fixed random patch baseline
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(12345)

    random_patch = torch.rand(
        learned_patch.shape,
        generator=generator,
    ).to(device)

    # --------------------------------------------------------
    # Source YOLOv5 clean detections
    # --------------------------------------------------------

    source_results = (
        load_source_clean_results(
            args.source_clean_json
        )
    )

    print(
        "YOLO source images:",
        len(source_results),
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_dir = Path(args.image_dir)

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
    )

    if args.max_images is not None:
        image_paths = image_paths[
            : args.max_images
        ]

    print(
        "Images:",
        len(image_paths),
    )

    resize = transforms.Resize(
        MODEL_IN_SIZE
    )

    to_tensor = transforms.ToTensor()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_clean = 0

    total_learned_survived = 0
    total_random_survived = 0

    per_image_results = []

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for index, image_path in enumerate(
        tqdm(
            image_paths,
            desc="Transfer evaluation",
        )
    ):

        image_id = image_path.stem

        source_boxes = source_results.get(
            image_id,
            [],
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = pad_to_square(image)

        image = resize(image)

        image_tensor = to_tensor(
            image
        ).to(device)

        # ====================================================
        # CLEAN Faster R-CNN
        # ====================================================

        clean_detections = detect_vehicles(
            model,
            image_tensor,
            device,
        )

        total_clean += len(
            clean_detections
        )

        # ====================================================
        # No YOLO source detections:
        # cannot determine placement using source detector.
        # Keep image unchanged.
        # ====================================================

        labels = (
            source_detections_to_labels(
                source_boxes
            )
        )

        if labels is None:

            learned_tensor = image_tensor
            random_tensor = image_tensor

        else:

            labels = (
                labels.unsqueeze(0)
                .to(device)
            )

            img_batch = (
                image_tensor.unsqueeze(0)
            )

            # ------------------------------------------------
            # Learned patch
            #
            # Reset RNG so learned/random patch receive
            # equivalent random transformation draws.
            # ------------------------------------------------

            transform_seed = (
                args.seed * 100000
                + index
            )

            seed_all(
                transform_seed
            )

            learned_batch_t = (
                patch_transformer(
                    learned_patch,
                    labels,
                    MODEL_IN_SIZE,
                    use_mul_add_gau=True,
                    do_transforms=True,
                    do_rotate=True,
                    rand_loc=True,
                )
            )

            learned_tensor = (
                patch_applier(
                    img_batch.clone(),
                    learned_batch_t,
                )
                .squeeze(0)
            )

            # ------------------------------------------------
            # Random baseline
            # ------------------------------------------------

            seed_all(
                transform_seed
            )

            random_batch_t = (
                patch_transformer(
                    random_patch,
                    labels,
                    MODEL_IN_SIZE,
                    use_mul_add_gau=True,
                    do_transforms=True,
                    do_rotate=True,
                    rand_loc=True,
                )
            )

            random_tensor = (
                patch_applier(
                    img_batch.clone(),
                    random_batch_t,
                )
                .squeeze(0)
            )

        # ====================================================
        # Faster R-CNN attacked detections
        # ====================================================

        learned_detections = (
            detect_vehicles(
                model,
                learned_tensor,
                device,
            )
        )

        random_detections = (
            detect_vehicles(
                model,
                random_tensor,
                device,
            )
        )

        learned_survived = (
            count_surviving_clean_detections(
                clean_detections,
                learned_detections,
            )
        )

        random_survived = (
            count_surviving_clean_detections(
                clean_detections,
                random_detections,
            )
        )

        total_learned_survived += (
            learned_survived
        )

        total_random_survived += (
            random_survived
        )

        per_image_results.append(
            {
                "image": image_path.name,
                "source_yolo_boxes": len(
                    source_boxes
                ),
                "clean_frcnn": len(
                    clean_detections
                ),
                "learned_frcnn": len(
                    learned_detections
                ),
                "random_frcnn": len(
                    random_detections
                ),
                "learned_survived": (
                    learned_survived
                ),
                "random_survived": (
                    random_survived
                ),
            }
        )

    # ========================================================
    # Final ASR
    # ========================================================

    if total_clean == 0:
        raise RuntimeError(
            "No Faster R-CNN clean detections."
        )

    learned_asr = (
        1.0
        - total_learned_survived
        / total_clean
    )

    random_asr = (
        1.0
        - total_random_survived
        / total_clean
    )

    transfer_gain = (
        learned_asr - random_asr
    )

    summary = {
        "source_model": "YOLOv5",
        "target_model": (
            "FasterRCNN-ResNet50-FPN"
        ),
        "patch": args.patch,
        "source_clean_json": (
            args.source_clean_json
        ),
        "images": len(image_paths),
        "confidence_threshold": (
            CONF_THRESH
        ),
        "iou_threshold": IOU_THRESH,
        "clean_reference_detections": (
            total_clean
        ),
        "random_survived": (
            total_random_survived
        ),
        "learned_survived": (
            total_learned_survived
        ),
        "random_asr": random_asr,
        "learned_transfer_asr": (
            learned_asr
        ),
        "transfer_gain": transfer_gain,
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    with open(
        output_dir / "per_image.json",
        "w",
    ) as f:

        json.dump(
            per_image_results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "YOLOv5 -> Faster R-CNN "
        "Zero-shot Transfer"
    )
    print("=" * 72)

    print(
        f"Images                  : "
        f"{len(image_paths)}"
    )

    print(
        f"Clean references        : "
        f"{total_clean}"
    )

    print(
        f"Random survived         : "
        f"{total_random_survived}"
    )

    print(
        f"Learned survived        : "
        f"{total_learned_survived}"
    )

    print()

    print(
        f"Random Patch ASR        : "
        f"{random_asr:.4f} "
        f"({random_asr * 100:.2f}%)"
    )

    print(
        f"YOLOv5 Patch ASR        : "
        f"{learned_asr:.4f} "
        f"({learned_asr * 100:.2f}%)"
    )

    print(
        f"Transfer Gain           : "
        f"{transfer_gain * 100:+.2f} pp"
    )

    print()

    print(
        "Saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()
