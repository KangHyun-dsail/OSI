#!/usr/bin/env bash
#
# Stage 6 — Generate steered images for a benchmark.
#
# Conda env : osi   (run `conda activate osi` BEFORE this script)
# Needs     : a trained classifier at $CLASSIFIER_DIR (classifier_ckpt/flux or
#             classifier_ckpt/sd3 ship with the repo; use flux_reproduced from Stage 5),
#             and the benchmark prompt files (see README "Benchmark setup").
#
# Variables (model-appropriate defaults are chosen automatically):
#   MODEL=flux|sd3   BENCH=geneval|compbench
#   GENEVAL_TYPE     prompt set, e.g. two_object_100 (geneval) or color_val_seen_phrase (compbench)
#   ALPHA NUM_HEAD INT_END SEED GPU OUTPUT_DIR CLASSIFIER_DIR
#
# The run-folder name is printed at the end; pass the SAME ALPHA/NUM_HEAD/INT_END/SEED
# to scripts/06_eval_geneval.sh so it can find the images.
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-flux}"                 # flux | sd3
BENCH="${BENCH:-geneval}"             # geneval | compbench
SEED="${SEED:-42}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

if [ "$MODEL" = "sd3" ]; then
  ALPHA="${ALPHA:-7.5}"; NUM_HEAD="${NUM_HEAD:-100}"
  INT_END="${INT_END:-}"              # sd3 default: no intervention_end
  CLASSIFIER_DIR="${CLASSIFIER_DIR:-classifier_ckpt/sd3}"
else
  ALPHA="${ALPHA:-5.0}"; NUM_HEAD="${NUM_HEAD:-300}"
  INT_END="${INT_END:-15}"           # flux default matches generate_geneval_flux.py
  CLASSIFIER_DIR="${CLASSIFIER_DIR:-classifier_ckpt/flux}"
fi

if [ "$BENCH" = "compbench" ]; then
  GENEVAL_TYPE="${GENEVAL_TYPE:-color_val_seen_phrase}"
  [ "$MODEL" = "sd3" ] && SCRIPT=generate_compbench_sd3.py || SCRIPT=generate_compbench_flux.py
else
  GENEVAL_TYPE="${GENEVAL_TYPE:-two_object_100}"
  [ "$MODEL" = "sd3" ] && SCRIPT=generate_geneval_sd3.py || SCRIPT=generate_geneval_flux.py
fi

END_ARGS=()
[ -n "$INT_END" ] && END_ARGS=(--intervention_end "$INT_END")

echo "[stage6] env=osi  model=$MODEL  bench=$BENCH  type=$GENEVAL_TYPE  gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" \
  --geneval_type "$GENEVAL_TYPE" \
  --alpha "$ALPHA" \
  --num_head "$NUM_HEAD" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --classifier_dir "$CLASSIFIER_DIR" \
  "${END_ARGS[@]}"

# Reproduce the run-folder name the generate script created.
PREFIX="osi"; [ "$MODEL" = "sd3" ] && PREFIX="osi_sd3"
END_STR=""; [ -n "$INT_END" ] && END_STR="_end${INT_END}"
if [ "$BENCH" = "geneval" ]; then
  RUN_NAME="${PREFIX}_alpha${ALPHA}_head${NUM_HEAD}${END_STR}_seed${SEED}"
  echo "[stage6] done. run name: $RUN_NAME"
  echo "[stage6] evaluate with:  MODEL=$MODEL GENEVAL_TYPE=$GENEVAL_TYPE ALPHA=$ALPHA NUM_HEAD=$NUM_HEAD INT_END=$INT_END SEED=$SEED bash scripts/06_eval_geneval.sh"
else
  echo "[stage6] done. CompBench images under $OUTPUT_DIR/<category>/. Evaluate via T2I-CompBench tools (see README)."
fi
