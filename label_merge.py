"""
Stage 3 of the training-data pipeline: merge Mask2Former (MM) and BLIP-VQA labels.

Reads the two per-object label files produced by label_mask2former.py and label_blip.py
and applies a consensus rule to emit a dual-labeled JSONL consumed by feature_extract.py.

Inputs:
  - mask2former_labels.json : {stem: {obj: 0/1}}
  - blip_labels.json        : {stem: {obj: float score}}

Output:
  - gen_text_real_two_object_512_MM_BLIP.jsonl :
      {"filename": "<samples_dir>/<stem>.png", "matched_groups": "{\"dog\": 1, \"elephant\": 0}"}

Consensus rule:
  MM=1 AND BLIP > present_thresh  -> label 1 (present)
  MM=0 AND BLIP < absent_thresh   -> label 0 (absent)
  otherwise                       -> None   (discarded downstream)
"""
import os
import json
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--mask2former_labels", type=str,
                    default="./data/training/mask2former_labels.json")
parser.add_argument("--blip_labels", type=str,
                    default="./data/training/blip_labels.json")
parser.add_argument("--samples_dir", type=str, default="./data/training/samples",
                    help="Used to build the filename path feature_extract.py resolves "
                         "to the corresponding datas/*.pkl")
parser.add_argument("--output", type=str,
                    default="gen_text_real_two_object_512_MM_BLIP.jsonl")
parser.add_argument("--present_thresh", type=float, default=0.9,
                    help="BLIP score above which a MM-detected object counts as present")
parser.add_argument("--absent_thresh", type=float, default=0.3,
                    help="BLIP score below which a MM-absent object counts as absent")
args = parser.parse_args()

with open(args.mask2former_labels) as f:
    m2f = json.load(f)   # {stem: {obj: 0/1}}
with open(args.blip_labels) as f:
    blip = json.load(f)  # {stem: {obj: float}}

merged = []
present_count = absent_count = 0

for stem, obj_labels in m2f.items():
    if stem not in blip:
        continue
    matched = {}
    for obj, mm_val in obj_labels.items():
        blip_score = blip[stem].get(obj)
        if mm_val is None or blip_score is None:
            matched[obj] = None
        elif mm_val == 1 and blip_score > args.present_thresh:
            matched[obj] = 1
            present_count += 1
        elif mm_val == 0 and blip_score < args.absent_thresh:
            matched[obj] = 0
            absent_count += 1
        else:
            matched[obj] = None

    filename = os.path.join(args.samples_dir, f"{stem}.png")
    merged.append({"filename": filename, "matched_groups": json.dumps(matched)})

print(f"Present: {present_count}, Absent: {absent_count}")

with open(args.output, "w") as fp:
    pd.DataFrame(merged).to_json(fp, orient="records", lines=True)

print(f"Saved to {args.output}")
