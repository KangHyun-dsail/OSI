"""
Step 3 of the training data pipeline: extract key vectors from pickle files
and split into train/test sets for classifier training.

Input:
  - gen_text_real_two_object_512_MM_BLIP.jsonl : dual-labeled dataset

Works for both FLUX.1-dev and SD3.5: the layer/head counts are inferred from the
collected pickles (override with --layer_len / --head_len).

Output (per timestep t, layer l, head h):
  - {output_dir}/
      x_train_key_{t}_{l}_{h}.pt
      x_test_key_{t}_{l}_{h}.pt
      y_train_data.pkl
      y_test_data.pkl
      y_train_label.pkl
      y_test_label.pkl
"""

import torch
import os
import pickle
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mm_blip_jsonl", type=str,
                    default="gen_text_real_two_object_512_MM_BLIP.jsonl")
parser.add_argument("--output_dir", type=str,
                    default="./data/features")
parser.add_argument("--train_ratio", type=float, default=0.89)
parser.add_argument("--max_per_class", type=int, default=10,
                    help="Max absent samples per object class (balanced with present)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layer_len", type=int, default=None,
                    help="Number of attention layers to read (default: inferred from the "
                         "collected pickles; FLUX.1-dev has 57, SD3.5-Medium 24)")
parser.add_argument("--head_len", type=int, default=None,
                    help="Number of attention heads per layer (default: inferred)")
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

with open(args.mm_blip_jsonl) as f:
    data = [json.loads(line) for line in f]

# Collect absent/present file lists per object
false_dict = {}
true_dict = {}

for d in data:
    for k, v in json.loads(d['matched_groups']).items():
        if v == 0:
            false_dict.setdefault(k, []).append(d['filename'])
        elif v == 1:
            true_dict.setdefault(k, []).append(d['filename'])

# Cap absent at max_per_class
for k in false_dict:
    if len(false_dict[k]) > args.max_per_class:
        false_dict[k] = list(np.random.permutation(false_dict[k])[:args.max_per_class])

# Balance present to match absent count
true_dict_sample = {}
for k, v in true_dict.items():
    if k not in false_dict:
        continue
    false_len = len(false_dict[k])
    if false_len > len(v):
        false_dict[k] = list(np.random.permutation(false_dict[k])[:len(v)])
        true_dict_sample[k] = v
    else:
        true_dict_sample[k] = list(np.random.permutation(v)[:false_len])

if not true_dict_sample:
    raise SystemExit(
        "No object class ended up with both present and absent samples, so there is "
        "nothing to extract. Balancing needs each class to appear as present in some "
        "images and absent in others; collect more samples in Stage 1 (a few thousand "
        "is the usual scale) and re-run."
    )

# Train/test split by sample count within each object class
train_true_dict, train_false_dict = {}, {}
valid_true_dict, valid_false_dict = {}, {}

for k in true_dict_sample:
    n = len(true_dict_sample[k])
    n_train = max(1, int(n * args.train_ratio))
    idx = np.random.permutation(n)
    train_true_dict[k]  = [true_dict_sample[k][i] for i in idx[:n_train]]
    train_false_dict[k] = [false_dict[k][i]        for i in idx[:n_train]]
    valid_true_dict[k]  = [true_dict_sample[k][i] for i in idx[n_train:]]
    valid_false_dict[k] = [false_dict[k][i]        for i in idx[n_train:]]

def pkl_path_for(png_path):
    return png_path.replace('samples', 'datas').replace('.png', '.pkl')


def infer_key_dims(*file_dicts):
    """Read one collected pickle to learn how many layers/heads it stores.

    FLUX.1-dev yields 57 layers (19 double + 38 single blocks) and SD3.5-Medium 24,
    so these cannot be hardcoded if both models are to share this script.
    """
    for file_dict in file_dicts:
        for k, file_list in file_dict.items():
            for file_ in file_list:
                pkl_path = pkl_path_for(file_)
                if not os.path.exists(pkl_path):
                    continue
                with open(pkl_path, 'rb') as f:
                    hidden_states = pickle.load(f)
                for t in hidden_states:
                    keys = hidden_states[t]['key']
                    return len(keys), keys[0][k].squeeze(2).shape[1]
    raise SystemExit(
        "Could not infer key-vector dimensions: no pickle found next to the samples "
        f"(expected e.g. {pkl_path_for('.../samples/foo.png')}). Check that Stage 1 "
        "wrote its datas/ directory, or pass --layer_len/--head_len explicitly."
    )


if args.layer_len is not None and args.head_len is not None:
    layer_len, head_len = args.layer_len, args.head_len
else:
    inferred_layer, inferred_head = infer_key_dims(
        train_true_dict, train_false_dict, valid_true_dict, valid_false_dict
    )
    layer_len = args.layer_len if args.layer_len is not None else inferred_layer
    head_len = args.head_len if args.head_len is not None else inferred_head

print(f"Key vectors: layer_len={layer_len}, head_len={head_len}")

train_y_data, train_y_label = [], []
valid_y_data, valid_y_label = [], []
train_buckets = defaultdict(list)
valid_buckets = defaultdict(list)


def collect(file_dict, label, y_data, y_label, buckets):
    for k, file_list in tqdm(file_dict.items(), desc=f"label={label}"):
        for file_ in file_list:
            y_data.append(label)
            y_label.append(k)
            pkl_path = pkl_path_for(file_)
            with open(pkl_path, 'rb') as f:
                hidden_states = pickle.load(f)
            for t in hidden_states:
                if len(hidden_states[t]['key']) < layer_len:
                    raise SystemExit(
                        f"{pkl_path} stores {len(hidden_states[t]['key'])} layers but "
                        f"layer_len={layer_len} was requested. Drop --layer_len to infer it, "
                        f"and keep each model's samples in a separate run."
                    )
                for l in range(layer_len):
                    sl = hidden_states[t]['key'][l][k].squeeze(2)
                    for h in range(head_len):
                        buckets[(t, l, h)].append(sl[:, h])


collect(train_false_dict, 0, train_y_data, train_y_label, train_buckets)
collect(valid_false_dict, 0, valid_y_data, valid_y_label, valid_buckets)
collect(train_true_dict,  1, train_y_data, train_y_label, train_buckets)
collect(valid_true_dict,  1, valid_y_data, valid_y_label, valid_buckets)

os.makedirs(args.output_dir, exist_ok=True)

print("Saving train tensors...")
for (t, l, h), lst in tqdm(train_buckets.items()):
    if lst:
        torch.save(torch.cat(lst, dim=0), f"{args.output_dir}/x_train_key_{t}_{l}_{h}.pt")

print("Saving test tensors...")
for (t, l, h), lst in tqdm(valid_buckets.items()):
    if lst:
        torch.save(torch.cat(lst, dim=0), f"{args.output_dir}/x_test_key_{t}_{l}_{h}.pt")

with open(f"{args.output_dir}/y_train_data.pkl", "wb") as f:
    pickle.dump(train_y_data, f)
with open(f"{args.output_dir}/y_train_label.pkl", "wb") as f:
    pickle.dump(train_y_label, f)
with open(f"{args.output_dir}/y_test_data.pkl", "wb") as f:
    pickle.dump(valid_y_data, f)
with open(f"{args.output_dir}/y_test_label.pkl", "wb") as f:
    pickle.dump(valid_y_label, f)

print(f"Done. Train: {len(train_y_data)} samples, Test: {len(valid_y_data)} samples")
