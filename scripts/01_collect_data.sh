#!/usr/bin/env bash
#
# Stage 1 — Collect training data (images + attention key vectors).
#
# Conda env : osi      (run `conda activate osi` BEFORE this script)
# Needs     : a free CUDA GPU, HuggingFace access to the gated FLUX.1-dev / SD3.5 weights.
#
# Output (under $OUTPUT_DIR):
#   samples/{prompt}_{idx:06d}.png   generated images
#   datas/{prompt}_{idx:06d}.pkl     per-timestep/layer/head key vectors
#
# Override any variable from the environment, e.g.:
#   MODEL=sd3 NUM_SAMPLES=8 GPU=1 bash scripts/01_collect_data.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-flux}"                 # flux | sd3
NUM_SAMPLES="${NUM_SAMPLES:-3000}"     # set small (e.g. 8) for a smoke test
GPU="${GPU:-0}"

if [ "$MODEL" = "sd3" ]; then
  SCRIPT=generate_dataset_sd3.py
  OUTPUT_DIR="${OUTPUT_DIR:-./data/training_sd3}"
else
  SCRIPT=generate_dataset_flux.py
  OUTPUT_DIR="${OUTPUT_DIR:-./data/training}"
fi

echo "[stage1] env=osi  model=$MODEL  num_samples=$NUM_SAMPLES  gpu=$GPU  out=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES"

echo "[stage1] done -> $OUTPUT_DIR/{samples,datas}"
