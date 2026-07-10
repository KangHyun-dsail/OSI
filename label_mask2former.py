"""
Stage 2a of the training-data pipeline: Mask2Former (MM) object-presence labeling.

Runs the GenEval Mask2Former detector over the flat two-object training samples and,
for each of the two prompt objects, records whether it was detected.

Conda env : geneval   (needs mmdet + the Mask2Former checkpoint)
Input     : {samples_dir}/{prompt}_{idx:06d}.png   (from generate_dataset_flux.py)
Output    : {out} = {"<prompt>_<idx>": {"<obj1>": 0/1, "<obj2>": 0/1}, ...}

Object names are parsed from the filename (prompts are "a X and an Y"), so this script
does NOT require the geneval repo — only mmdet, the config, and the checkpoint.
"""
import os
import json
import argparse

import numpy as np
import mmdet
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--samples_dir", default="./data/training/samples",
                    help="Directory of flat {prompt}_{idx}.png training images")
parser.add_argument("--out", default="./data/training/mask2former_labels.json",
                    help="Output JSON: {stem: {obj: 0/1}}")
parser.add_argument("--object_names", default="assets/object_names.txt",
                    help="COCO class list (must match the detector's class order)")
parser.add_argument("--model_path", default="geneval/checkpoints",
                    help="Directory holding <detector>.pth")
parser.add_argument("--model_config", default=None,
                    help="Mask2Former config path (defaults to the one bundled with mmdet)")
parser.add_argument("--confidence_threshold", type=float, default=0.3)
parser.add_argument("--max_objects", type=int, default=16)
args = parser.parse_args()

DEVICE = "cuda"
MODEL_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
CONFIG_PATH = args.model_config or os.path.join(
    os.path.dirname(mmdet.__file__), f"../configs/mask2former/{MODEL_NAME}.py")
CKPT_PATH = os.path.join(args.model_path, f"{MODEL_NAME}.pth")

print("Loading Mask2Former ...")
object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

with open(args.object_names) as f:
    classnames = [line.strip() for line in f if line.strip()]


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


def detect_objects(img_path):
    result = inference_detector(object_detector, img_path)
    bbox = result[0] if isinstance(result, tuple) else result
    detected = {}
    for idx, classname in enumerate(classnames):
        ordering = np.argsort(bbox[idx][:, 4])[::-1]
        ordering = ordering[bbox[idx][ordering, 4] > args.confidence_threshold]
        ordering = ordering[:args.max_objects].tolist()
        if ordering:
            detected[classname] = len(ordering)
    return detected


os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

# Resume support
if os.path.exists(args.out):
    with open(args.out) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} already done")
else:
    results = {}

stems = sorted(fn[:-4] for fn in os.listdir(args.samples_dir) if fn.endswith(".png"))

for stem in tqdm(stems):
    if stem in results:
        continue
    img_path = os.path.join(args.samples_dir, f"{stem}.png")
    objs = parse_objs_from_stem(stem)
    try:
        detected = detect_objects(img_path)
        results[stem] = {obj: (1 if obj in detected else 0) for obj in objs}
    except Exception as e:
        print(f"[ERROR] {stem}: {e}")
        results[stem] = {obj: None for obj in objs}

with open(args.out, "w") as f:
    json.dump(results, f)

print(f"Done -> {args.out}  ({len(results)} samples)")
