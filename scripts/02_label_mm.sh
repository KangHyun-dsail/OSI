#!/usr/bin/env bash
#
# Stage 2a — Mask2Former (MM) object-presence labeling.
#
# Conda env : geneval   (run `conda activate geneval` BEFORE this script)
# Needs     : mmdet installed and the Mask2Former checkpoint at
#             $MODEL_PATH/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
#             (the geneval repo itself is NOT required — object names come from
#             assets/object_names.txt and the filename prompts).
#
# Output    : $OUT = {"<stem>": {"<obj1>": 0/1, "<obj2>": 0/1}, ...}
#             consumed by Stage 3 (label_merge.py).
#
set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLES_DIR="${SAMPLES_DIR:-./data/training/samples}"
OUT="${OUT:-./data/training/mask2former_labels.json}"
MODEL_PATH="${MODEL_PATH:-geneval/checkpoints}"
GPU="${GPU:-0}"

echo "[stage2a] env=geneval  samples=$SAMPLES_DIR  out=$OUT  gpu=$GPU"

CUDA_VISIBLE_DEVICES="$GPU" python label_mask2former.py \
  --samples_dir "$SAMPLES_DIR" \
  --out "$OUT" \
  --model_path "$MODEL_PATH"

echo "[stage2a] done -> $OUT"
