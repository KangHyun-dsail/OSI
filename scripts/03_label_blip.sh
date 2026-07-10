#!/usr/bin/env bash
#
# Stage 2b — BLIP-VQA object-presence labeling.
#
# Conda env : compbench   (run `conda activate compbench` BEFORE this script)
# Needs     : the T2I-CompBench repo at ./T2I-CompBench (see README "Benchmark setup").
#             BLIP-VQA weights download automatically on first run.
#
# Output    : $OUT = {"<stem>": {"<obj1>": score, "<obj2>": score}, ...}
#             consumed by Stage 3 (label_merge.py).
#
set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLES_DIR="${SAMPLES_DIR:-./data/training/samples}"
OUT="${OUT:-./data/training/blip_labels.json}"
BLIPVQA_DIR="${BLIPVQA_DIR:-T2I-CompBench/BLIPvqa_eval}"
GPU="${GPU:-0}"

echo "[stage2b] env=compbench  samples=$SAMPLES_DIR  out=$OUT  gpu=$GPU"

CUDA_VISIBLE_DEVICES="$GPU" python label_blip.py \
  --samples_dir "$SAMPLES_DIR" \
  --out "$OUT" \
  --blipvqa_dir "$BLIPVQA_DIR"

echo "[stage2b] done -> $OUT"
