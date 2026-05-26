import torch
from flux_osi_modules.pipeline import FluxPipeline_custom
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True,
                    help='Text prompt, e.g. "a dog and a cat"')
parser.add_argument("--target_tokens", type=str, nargs='+', required=True,
                    help='Target concept tokens to collect key vectors for, e.g. dog cat')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="./collected",
                    help='Directory to save image and key vectors. Defaults to ./collected')
args = parser.parse_args()

pipe = FluxPipeline_custom.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.setting(mode='collect')

out, hidden_states = pipe(
    prompt=args.prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=torch.Generator("cpu").manual_seed(args.seed),
    width=512,
    height=512,
    target_tokens=args.target_tokens,
)

os.makedirs(args.output_dir, exist_ok=True)
stem = f"{args.prompt}_{args.seed}"
image_path = os.path.join(args.output_dir, f"{stem}.png")
data_path = os.path.join(args.output_dir, f"{stem}.pkl")

out.images[0].save(image_path)
with open(data_path, 'wb') as f:
    pickle.dump(hidden_states, f)

print(f"Image  → {image_path}")
print(f"Keys   → {data_path}")
