#!/usr/bin/env bash
#
# Stage 3-5 — Merge labels -> extract features -> train MMS classifier.
# All three steps share the same env, so they are chained here.
#
# Conda env : osi   (run `conda activate osi` BEFORE this script)
# Needs     : Stage 1 outputs ($DATAS_DIR/*.pkl via the merged jsonl) and the two label
#             files from Stage 2a/2b ($MM_LABELS and $BLIP_LABELS).
#
# Output    : $CLASSIFIER_DIR/{weight.pkl,accuracy.pkl}
#             (directly loadable at inference via --classifier_dir).
#
set -euo pipefail
cd "$(dirname "$0")/.."

MM_LABELS="${MM_LABELS:-./data/training/mask2former_labels.json}"
BLIP_LABELS="${BLIP_LABELS:-./data/training/blip_labels.json}"
SAMPLES_DIR="${SAMPLES_DIR:-./data/training/samples}"
MERGED_JSONL="${MERGED_JSONL:-gen_text_real_two_object_512_MM_BLIP.jsonl}"
FEATURES_DIR="${FEATURES_DIR:-./data/features}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:-classifier_ckpt/flux_reproduced}"

echo "[stage3] merge MM + BLIP labels -> $MERGED_JSONL"
python label_merge.py \
  --mask2former_labels "$MM_LABELS" \
  --blip_labels "$BLIP_LABELS" \
  --samples_dir "$SAMPLES_DIR" \
  --output "$MERGED_JSONL"

echo "[stage4] extract key tensors -> $FEATURES_DIR"
python feature_extract.py \
  --mm_blip_jsonl "$MERGED_JSONL" \
  --output_dir "$FEATURES_DIR"

echo "[stage5] train MMS classifier -> $CLASSIFIER_DIR"
python train_classifier_multi_object_real.py \
  --input_dir "$FEATURES_DIR" \
  --output_dir "$CLASSIFIER_DIR"

echo "[stage3-5] done -> $CLASSIFIER_DIR/{weight,accuracy}.pkl"
