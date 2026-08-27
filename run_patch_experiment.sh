#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash run_patch_experiment.sh <config.json> <experiment_name> "<epochs>" [seed]
#
# Example:
# bash run_patch_experiment.sh \
#   adv_patch_gen/configs/ablation_full_10.json \
#   ablation_full_10 \
#   "10" \
#   42

CFG="${1:-}"
EXP_NAME="${2:-}"
EPOCHS="${3:-}"
SEED="${4:-}"

WEIGHTS="${WEIGHTS:-runs/train/s_coco_e300_4Class_Vehicle/weights/best.pt}"
VAL_DIR="${VAL_DIR:-data/visdrone_data/VisDrone2019-DET-val/images}"

if [[ -z "$CFG" || -z "$EXP_NAME" || -z "$EPOCHS" ]]; then
    echo "Usage:"
    echo '  bash run_patch_experiment.sh <config.json> <experiment_name> "<epochs>" [seed]'
    echo
    echo "Example:"
    echo '  bash run_patch_experiment.sh adv_patch_gen/configs/ablation_full_10.json ablation_full_10 "10" 42'
    exit 1
fi

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config not found: $CFG"
    exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Weights not found: $WEIGHTS"
    exit 1
fi

if [[ ! -d "$VAL_DIR" ]]; then
    echo "ERROR: Validation directory not found: $VAL_DIR"
    exit 1
fi

if [[ -n "$SEED" ]]; then
    if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Seed must be an integer."
        exit 1
    fi

    RUN_NAME="${EXP_NAME}_seed${SEED}"
else
    RUN_NAME="$EXP_NAME"
fi

mkdir -p logs/generated_configs
mkdir -p runs/test_adversarial

STAMP="$(date +%Y%m%d-%H%M%S)"

GENERATED_CFG="logs/generated_configs/${RUN_NAME}_${STAMP}.json"
TRAIN_LOG="logs/${RUN_NAME}_${STAMP}_train.log"
META_FILE="logs/${RUN_NAME}_${STAMP}_meta.txt"

# ------------------------------------------------------------
# 自动生成一份本次实验配置
# 并自动修改 patch_name
# ------------------------------------------------------------

python - "$CFG" "$GENERATED_CFG" "$RUN_NAME" <<'PY'
import json
import sys

src = sys.argv[1]
dst = sys.argv[2]
run_name = sys.argv[3]

with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["patch_name"] = run_name

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)

print("Generated config:", dst)
print("patch_name:", run_name)
PY

# ------------------------------------------------------------
# 保存实验信息
# ------------------------------------------------------------

{
    echo "experiment=$EXP_NAME"
    echo "run_name=$RUN_NAME"
    echo "seed=${SEED:-None}"
    echo "source_config=$CFG"
    echo "generated_config=$GENERATED_CFG"
    echo "test_epochs=$EPOCHS"
    echo "weights=$WEIGHTS"
    echo "validation=$VAL_DIR"
    echo "timestamp=$STAMP"
} > "$META_FILE"

echo "============================================================"
echo "Experiment : $EXP_NAME"
echo "Run name   : $RUN_NAME"
echo "Seed       : ${SEED:-None}"
echo "Config     : $CFG"
echo "Generated  : $GENERATED_CFG"
echo "Weights    : $WEIGHTS"
echo "Val images : $VAL_DIR"
echo "Test epochs: $EPOCHS"
echo "Train log  : $TRAIN_LOG"
echo "Metadata   : $META_FILE"
echo "============================================================"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

echo
echo "[1/3] Training..."

if [[ -n "$SEED" ]]; then
    PATCH_SEED="$SEED" \
    python train_patch.py \
        --cfg "$GENERATED_CFG" \
        2>&1 | tee "$TRAIN_LOG"
else
    python train_patch.py \
        --cfg "$GENERATED_CFG" \
        2>&1 | tee "$TRAIN_LOG"
fi

# ------------------------------------------------------------
# 找到刚才的训练目录
# ------------------------------------------------------------

RUN_DIR="$(ls -dt runs/train_adversarial/*_"${RUN_NAME}" 2>/dev/null | head -1 || true)"

if [[ -z "$RUN_DIR" ]]; then
    echo
    echo "ERROR: Could not find training run:"
    echo "  *_${RUN_NAME}"
    exit 1
fi

echo
echo "Training run found:"
echo "  $RUN_DIR"

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

echo
echo "[2/3] Evaluating..."

for E in $EPOCHS
do
    PATCH="$RUN_DIR/patches/e_${E}.png"

    if [[ ! -f "$PATCH" ]]; then
        echo
        echo "WARNING:"
        echo "Patch not found:"
        echo "  $PATCH"
        echo "Skipping epoch $E"
        continue
    fi

    OUT_ROOT="runs/test_adversarial/${RUN_NAME}_e${E}_fullval"

    EVAL_LOG="logs/${RUN_NAME}_e${E}_${STAMP}_eval.log"

    echo
    echo "------------------------------------------------------------"
    echo "Testing epoch $E"
    echo "Seed   : ${SEED:-None}"
    echo "Patch  : $PATCH"
    echo "Output : $OUT_ROOT"
    echo "------------------------------------------------------------"

    python test_patch.py \
        --cfg "$RUN_DIR/cfg.json" \
        -w "$WEIGHTS" \
        -p "$PATCH" \
        --id "$VAL_DIR" \
        --sd "$OUT_ROOT" \
        2>&1 | tee "$EVAL_LOG"

    PATCH_STATS="$(find "$OUT_ROOT" \
        -name "patch_map_stats.txt" \
        -type f \
        -print0 2>/dev/null \
        | xargs -0 -r ls -t \
        | head -1 || true)"

    NOISE_STATS="$(find "$OUT_ROOT" \
        -name "noise_map_stats.txt" \
        -type f \
        -print0 2>/dev/null \
        | xargs -0 -r ls -t \
        | head -1 || true)"

    CLEAN_STATS="$(find "$OUT_ROOT" \
        -name "clean_map_stats.txt" \
        -type f \
        -print0 2>/dev/null \
        | xargs -0 -r ls -t \
        | head -1 || true)"

    echo
    echo "========== Epoch $E Summary =========="

    if [[ -n "$PATCH_STATS" ]]; then

        echo
        echo "Adversarial Patch:"

        grep -E \
        "Average Precision.*IoU=0.50:0.95.*area=.*all|Attack success rate.*area=.*all" \
        "$PATCH_STATS" \
        | head -2 || true
    fi

    if [[ -n "$NOISE_STATS" ]]; then

        echo
        echo "Random Patch:"

        grep -E \
        "Average Precision.*IoU=0.50:0.95.*area=.*all|Attack success rate.*area=.*all" \
        "$NOISE_STATS" \
        | head -2 || true
    fi

    if [[ -n "$CLEAN_STATS" ]]; then

        echo
        echo "Clean:"

        grep -E \
        "Average Precision.*IoU=0.50:0.95.*area=.*all" \
        "$CLEAN_STATS" \
        | head -1 || true
    fi

done

echo
echo "[3/3] Finished."
echo
echo "Training run:"
echo "  $RUN_DIR"
echo
echo "Training log:"
echo "  $TRAIN_LOG"
echo
echo "Metadata:"
echo "  $META_FILE"
echo
echo "Test result prefix:"
echo "  runs/test_adversarial/${RUN_NAME}_e*_fullval"
