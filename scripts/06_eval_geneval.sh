#!/usr/bin/env bash
#
# Stage 7 — Evaluate generated images with GenEval (Mask2Former).
#
# Conda env : geneval   (run `conda activate geneval` BEFORE this script)
# Needs     : the geneval repo at ./geneval, the Mask2Former checkpoint at
#             $MODEL_PATH/<detector>.pth, and images produced by Stage 6.
#
# Pass the SAME MODEL/ALPHA/NUM_HEAD/INT_END/SEED you used in scripts/05_generate.sh
# so the run-folder name matches. Alternatively set NAME=<run folder> directly.
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-flux}"
GENEVAL_TYPE="${GENEVAL_TYPE:-two_object_100}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
INPUT_DIR="${INPUT_DIR:-./outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MODEL_PATH="${MODEL_PATH:-geneval/checkpoints}"

if [ "$MODEL" = "sd3" ]; then
  ALPHA="${ALPHA:-7.5}"; NUM_HEAD="${NUM_HEAD:-100}"; INT_END="${INT_END:-}"
  PREFIX="osi_sd3"
else
  ALPHA="${ALPHA:-5.0}"; NUM_HEAD="${NUM_HEAD:-300}"; INT_END="${INT_END:-15}"
  PREFIX="osi"
fi

# Derive the run-folder name (override with NAME=... to evaluate an arbitrary folder).
END_STR=""; [ -n "$INT_END" ] && END_STR="_end${INT_END}"
NAME="${NAME:-${PREFIX}_alpha${ALPHA}_head${NUM_HEAD}${END_STR}_seed${SEED}}"

echo "[stage7] env=geneval  type=$GENEVAL_TYPE  name=$NAME  gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python evaluate_geneval.py \
  --geneval_type "$GENEVAL_TYPE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH" \
  --name "$NAME"

echo "[stage7] scoring $OUTPUT_DIR/$GENEVAL_TYPE/$NAME.jsonl"
python geneval/evaluation/summary_scores.py "$OUTPUT_DIR/$GENEVAL_TYPE/$NAME.jsonl"
