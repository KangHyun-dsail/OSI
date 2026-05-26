import torch
from sd3_osi_modules.pipeline import StableDiffusion3Pipeline_custom
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True,
                    help='Text prompt, e.g. "a dog and a cat"')
parser.add_argument("--target_tokens", type=str, nargs='+', required=True,
                    help='Target concept tokens to intervene on, e.g. dog cat')
parser.add_argument("--alpha", type=float, default=7.5,
                    help='Intervention strength (positive = recovery). Default: 7.5')
parser.add_argument("--num_head", type=int, default=100,
                    help='Number of top attention heads to use. Default: 100')
parser.add_argument("--intervention_end", type=int, default=15,
                    help='Denoising step at which to stop intervention (out of 28). Default: 15')
parser.add_argument("--classifier_dir", type=str,
                    default='classifier_ckpt/sd3',
                    help='Path to the trained classifier directory')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None,
                    help='Output image path. Defaults to <prompt>_<seed>.png')
args = parser.parse_args()

pipe = StableDiffusion3Pipeline_custom.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.setting(
    direction_path=args.classifier_dir,
    num_head=args.num_head,
)

out = pipe(
    prompt=args.prompt,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=torch.Generator("cpu").manual_seed(args.seed),
    width=1024,
    height=1024,
    target_tokens=args.target_tokens,
    alpha=args.alpha,
    intervention_end=args.intervention_end,
)

output_path = args.output or f"{args.prompt}_{args.seed}.png"
out.images[0].save(output_path)
print(f"Saved to {output_path}")
