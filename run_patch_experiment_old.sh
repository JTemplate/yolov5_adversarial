#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_patch_experiment.sh <config.json> <experiment_name> "<epochs_to_test>"
#
# Example:
#   bash run_patch_experiment.sh adv_patch_gen/configs/train20.json train20 "1 5 10 20"
#
# Notes:
#   1) Run this script from the yolov5_adversarial project root.
#   2) It assumes config.patch_name is the same as <experiment_name>.
#   3) It trains first, then evaluates the requested saved patch epochs.

CFG="${1:-}"
EXP_NAME="${2:-}"
EPOCHS="${3:-}"

WEIGHTS="${WEIGHTS:-runs/train/s_coco_e300_4Class_Vehicle/weights/best.pt}"
VAL_DIR="${VAL_DIR:-data/visdrone_data/VisDrone2019-DET-val/images}"

if [[ -z "$CFG" || -z "$EXP_NAME" || -z "$EPOCHS" ]]; then
    echo "Usage:"
    echo "  bash run_patch_experiment.sh <config.json> <experiment_name> \"<epochs>\""
    echo
    echo "Example:"
    echo "  bash run_patch_experiment.sh adv_patch_gen/configs/train20.json train20 \"1 5 10 20\""
    exit 1
fi

if [[ ! -f "train_patch.py" || ! -f "test_patch.py" ]]; then
    echo "ERROR: Run this script from the yolov5_adversarial project root."
    exit 1
fi

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config not found: $CFG"
    exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Detector weights not found: $WEIGHTS"
    exit 1
fi

if [[ ! -d "$VAL_DIR" ]]; then
    echo "ERROR: Validation image directory not found: $VAL_DIR"
    exit 1
fi

mkdir -p logs runs/test_adversarial

STAMP="$(date +%Y%m%d-%H%M%S)"
TRAIN_LOG="logs/${EXP_NAME}_${STAMP}_train.log"

echo "============================================================"
echo "Experiment : $EXP_NAME"
echo "Config     : $CFG"
echo "Weights    : $WEIGHTS"
echo "Val images : $VAL_DIR"
echo "Test epochs: $EPOCHS"
echo "Train log  : $TRAIN_LOG"
echo "============================================================"

echo
echo "[1/3] Training..."
python train_patch.py --cfg "$CFG" 2>&1 | tee "$TRAIN_LOG"

RUN_DIR="$(ls -dt runs/train_adversarial/*_"${EXP_NAME}" 2>/dev/null | head -1 || true)"

if [[ -z "$RUN_DIR" ]]; then
    echo
    echo "ERROR: Could not find a training run ending in _${EXP_NAME}"
    echo "Check that patch_name in $CFG equals: $EXP_NAME"
    exit 1
fi

echo
echo "Training run found:"
echo "  $RUN_DIR"

echo
echo "[2/3] Evaluating saved patches..."

for E in $EPOCHS; do
    PATCH="$RUN_DIR/patches/e_${E}.png"

    if [[ ! -f "$PATCH" ]]; then
        echo "WARNING: Patch not found, skipping epoch $E: $PATCH"
        continue
    fi

    OUT_ROOT="runs/test_adversarial/${EXP_NAME}_e${E}_fullval"
    EVAL_LOG="logs/${EXP_NAME}_e${E}_${STAMP}_eval.log"

    echo
    echo "------------------------------------------------------------"
    echo "Testing epoch $E"
    echo "Patch : $PATCH"
    echo "Output: $OUT_ROOT"
    echo "Log   : $EVAL_LOG"
    echo "------------------------------------------------------------"

    python test_patch.py         --cfg "$RUN_DIR/cfg.json"         -w "$WEIGHTS"         -p "$PATCH"         --id "$VAL_DIR"         --sd "$OUT_ROOT"         2>&1 | tee "$EVAL_LOG"

    PATCH_STATS="$(find "$OUT_ROOT" -name "patch_map_stats.txt" -type f -print0 2>/dev/null | xargs -0 -r ls -t | head -1 || true)"
    NOISE_STATS="$(find "$OUT_ROOT" -name "noise_map_stats.txt" -type f -print0 2>/dev/null | xargs -0 -r ls -t | head -1 || true)"
    CLEAN_STATS="$(find "$OUT_ROOT" -name "clean_map_stats.txt" -type f -print0 2>/dev/null | xargs -0 -r ls -t | head -1 || true)"

    echo
    echo "Summary for epoch $E:"
    if [[ -n "$PATCH_STATS" ]]; then
        echo "  Adversarial patch:"
        grep -E "Average Precision.*IoU=0.50:0.95.*area=.*all|Attack success rate.*area=.*all" "$PATCH_STATS" | head -2 || true
    fi
    if [[ -n "$NOISE_STATS" ]]; then
        echo "  Random patch:"
        grep -E "Average Precision.*IoU=0.50:0.95.*area=.*all|Attack success rate.*area=.*all" "$NOISE_STATS" | head -2 || true
    fi
    if [[ -n "$CLEAN_STATS" ]]; then
        echo "  Clean reference:"
        grep -E "Average Precision.*IoU=0.50:0.95.*area=.*all" "$CLEAN_STATS" | head -1 || true
    fi
done

echo
echo "[3/3] Done."
echo "Training run : $RUN_DIR"
echo "Training log : $TRAIN_LOG"
echo
echo "Next:"
echo "  python compare_results.py --root runs/test_adversarial --out analysis"
