"""
Step 1 of the training data pipeline: generate two-object images and save
key vectors (hidden states) at each timestep for classifier training.

Output (written to --output_dir):
  samples/{prompt}_{idx:06d}.png   — generated images
  datas/{prompt}_{idx:06d}.pkl     — hidden states per timestep/layer/head
"""

import torch
from flux_osi_modules.pipeline import FluxPipeline_custom
import os
import pickle
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./data/training",
                    help="Root directory to save generated images and hidden states")
parser.add_argument("--num_samples", type=int, default=3000,
                    help="Number of two-object images to generate")
parser.add_argument("--object_names", type=str, default="assets/object_names.txt",
                    help="Text file listing candidate object classes (one per line)")
args = parser.parse_args()

pipe = FluxPipeline_custom.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.setting(mode='collect')

with open(args.object_names) as cls_file:
    objects = [line.strip() for line in cls_file if line.strip()]

samples_dir = os.path.join(args.output_dir, "samples")
datas_dir = os.path.join(args.output_dir, "datas")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(datas_dir, exist_ok=True)

for idx in tqdm(range(args.num_samples)):
    object_num = 2
    objs = []
    articles = []
    while len(objs) < object_num:
        obj = random.choice(objects)
        if obj not in objs:
            article = 'an' if obj[0] in 'aeiou' else 'a'
            objs.append(obj)
            articles.append(article)

    prompt = ''
    for i in range(object_num):
        if i == 0:
            prompt += f'{articles[i]} {objs[i]}'
        else:
            prompt += f' and {articles[i]} {objs[i]}'

    out, hidden_states = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        width=512,
        height=512,
        target_tokens=objs,
        num_inference_steps=20,
        generator=torch.Generator("cpu").manual_seed(idx),
    )

    out.images[0].save(os.path.join(samples_dir, f'{prompt}_{idx:06d}.png'))
    with open(os.path.join(datas_dir, f'{prompt}_{idx:06d}.pkl'), 'wb') as f:
        pickle.dump(hidden_states, f)
