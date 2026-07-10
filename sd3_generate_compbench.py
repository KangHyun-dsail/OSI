import torch
from sd3_osi_modules.pipeline import StableDiffusion3Pipeline_custom
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--geneval_type", type=str, default="non_spatial_val")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=7.5,
                    help='Intervention strength (positive = recovery). Default: 7.5')
parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Root directory for generated images")
parser.add_argument("--num_head", type=int, default=100)
parser.add_argument("--intervention_end", type=int, default=None)
parser.add_argument("--classifier_dir", type=str, default='classifier_ckpt/sd3',
                    help='Path to the trained SD3 classifier directory')
args = parser.parse_args()

pipe = StableDiffusion3Pipeline_custom.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.setting(
    direction_path=args.classifier_dir,
    num_head=args.num_head,
)

with open(f'T2I-CompBench/examples/dataset/{args.geneval_type}.json', 'r') as f:
    data = json.load(f)

end_str = f'_end{args.intervention_end}' if args.intervention_end is not None else ''

if 'phrase' in args.geneval_type:
    suffix = '_phrase'
    category = args.geneval_type.replace('_phrase', '')
elif 'attribute' in args.geneval_type:
    suffix = '_attribute'
    category = args.geneval_type.replace('_attribute', '')
else:
    suffix = ''
    category = args.geneval_type

run_name = f'osi_sd3_alpha{args.alpha}_head{args.num_head}{suffix}{end_str}_seed{args.seed}'
output_dir = os.path.join(args.output_dir, category, run_name, 'samples')
os.makedirs(output_dir, exist_ok=True)

for idx, d in tqdm(enumerate(data)):
    img_path = os.path.join(output_dir, f'{d["prompt"]}_{idx:06d}.png')
    if os.path.exists(img_path):
        continue

    try:
        out = pipe(
            prompt=d['prompt'],
            num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            target_tokens=d['objects'],
            alpha=args.alpha,
            intervention_end=15 if args.intervention_end is None else args.intervention_end,
        )
        out.images[0].save(img_path)
    except Exception as e:
        print(e)
        continue
