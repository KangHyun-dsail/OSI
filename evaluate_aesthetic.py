from pathlib import Path

import torch
import pyiqa
from PIL import Image
from tqdm import tqdm
import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--geneval_type", type=str, default="two_object_100")
parser.add_argument("--input_dir", type=str, default="./outputs",
                    help="Root directory containing generated images")
parser.add_argument("--output_dir", type=str, default="./results",
                    help="Root directory for evaluation results")
parser.add_argument("--name", type=str, default="osi_alpha5.0_head300_seed42",
                    help="Run folder name (matches the folder created by generate_osi_main_table.py)")
parser.add_argument("--metric", type=str, default="musiq", choices=["musiq", "maniqa"])
args, unknown = parser.parse_known_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
metric = pyiqa.create_metric(args.metric, device=device)

samples_dir = os.path.join(args.input_dir, args.geneval_type, args.name, 'samples')
scores = []
for file_path in tqdm(os.listdir(samples_dir)):
    if file_path.endswith('png'):
        try:
            image_path = os.path.join(samples_dir, file_path)
            score = metric(image_path).item()
            scores.append(score)
        except Exception as e:
            print(e)
            continue

print(args.name)
print(np.mean(scores))

out_dir = os.path.join(args.output_dir, args.geneval_type)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f'{args.name}_{args.metric}.txt'), 'w') as f:
    f.write(str(np.mean(scores)))