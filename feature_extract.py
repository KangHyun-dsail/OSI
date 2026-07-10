"""
Step 3 of the training data pipeline: extract key vectors from pickle files
and split into train/test sets for classifier training.

Input:
  - gen_text_real_two_object_512_MM_BLIP.jsonl : dual-labeled dataset

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

head_len = 24
layer_len = 57

train_y_data, train_y_label = [], []
valid_y_data, valid_y_label = [], []
train_buckets = defaultdict(list)
valid_buckets = defaultdict(list)


def collect(file_dict, label, y_data, y_label, buckets):
    for k, file_list in tqdm(file_dict.items(), desc=f"label={label}"):
        for file_ in file_list:
            y_data.append(label)
            y_label.append(k)
            pkl_path = file_.replace('samples', 'datas').replace('.png', '.pkl')
            with open(pkl_path, 'rb') as f:
                hidden_states = pickle.load(f)
            for t in hidden_states:
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
