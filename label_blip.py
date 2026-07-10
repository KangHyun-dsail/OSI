"""
Stage 2b of the training-data pipeline: BLIP-VQA object-presence labeling.

Asks BLIP-VQA "a <obj>?" for each of the two prompt objects on every flat training
sample and records the raw yes-probability.

Conda env : compbench   (uses T2I-CompBench/BLIPvqa_eval; BLIP weights auto-download)
Input     : {samples_dir}/{prompt}_{idx:06d}.png   (from generate_dataset_flux.py)
Output    : {out} = {"<prompt>_<idx>": {"<obj1>": score, "<obj2>": score}, ...}

Object names are parsed from the filename, so no manifest jsonl is required.
"""
import os
import sys
import json
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--samples_dir", default="./data/training/samples",
                    help="Directory of flat {prompt}_{idx}.png training images")
parser.add_argument("--out", default="./data/training/blip_labels.json",
                    help="Output JSON: {stem: {obj: score}}")
parser.add_argument("--blipvqa_dir", default="T2I-CompBench/BLIPvqa_eval",
                    help="Path to the T2I-CompBench BLIPvqa_eval directory")
parser.add_argument("--blip_tmp", default="/tmp/blip_kv_two_object",
                    help="Scratch directory for BLIP intermediate files")
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.abspath(args.samples_dir)
OUT_PATH = os.path.abspath(args.out)
BLIPVQA_DIR = os.path.abspath(args.blipvqa_dir)

sys.path.insert(0, BLIPVQA_DIR)
os.chdir(BLIPVQA_DIR)
from BLIP.train_vqa_func import VQA_main  # noqa: E402
os.chdir(PROJECT_ROOT)


def parse_objs_from_stem(stem):
    """'a dog and an elephant_000042' -> ['dog', 'elephant'] (multi-word classes safe)."""
    prompt = stem.rsplit("_", 1)[0]
    objs = []
    for part in prompt.split(" and "):
        part = part.strip()
        if part.startswith("an "):
            part = part[3:]
        elif part.startswith("a "):
            part = part[2:]
        if part:
            objs.append(part)
    return objs


os.makedirs(args.blip_tmp, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

# ---------- build BLIP question list ----------
annotations = []
qid = 0
qid_map = {}  # qid -> (stem, obj)

stems = sorted(fn[:-4] for fn in os.listdir(SAMPLES_DIR) if fn.endswith(".png"))
for stem in stems:
    img_path = os.path.join(SAMPLES_DIR, f"{stem}.png")
    for obj in parse_objs_from_stem(stem):
        article = "an" if obj[0].lower() in "aeiou" else "a"
        annotations.append({
            "image": img_path,
            "question_id": qid,
            "question": f"{article} {obj}?",
            "dataset": "color",
        })
        qid_map[qid] = (stem, obj)
        qid += 1

print(f"Total BLIP questions: {len(annotations)} ({len(stems)} images)")

# ---------- run BLIP-VQA ----------
ann_dir = os.path.join(args.blip_tmp, "annotation")
vqa_dir = os.path.join(ann_dir, "VQA")
os.makedirs(vqa_dir, exist_ok=True)
with open(os.path.join(ann_dir, "vqa_test.json"), "w") as f:
    json.dump(annotations, f)

os.chdir(BLIPVQA_DIR)
VQA_main(ann_dir + "/", vqa_dir + "/")
os.chdir(PROJECT_ROOT)

# ---------- parse results ----------
with open(os.path.join(vqa_dir, "result", "vqa_result.json")) as f:
    vqa_results = json.load(f)
raw_scores = {r["question_id"]: float(r["answer"]) for r in vqa_results}

results = {}
for qid_r, (stem, obj) in qid_map.items():
    results.setdefault(stem, {})[obj] = round(raw_scores.get(qid_r, 0.0), 4)

with open(OUT_PATH, "w") as f:
    json.dump(results, f)

print(f"Done -> {OUT_PATH}  ({len(results)} samples)")
