import torch
from flux_osi_modules.pipeline import FluxPipeline_custom
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--geneval_type", type=str, default="two_object_100")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=5.0,
                    help='Intervention strength. Default: 5.0')
parser.add_argument("--num_head", type=int, default=300)
parser.add_argument("--intervention_end", type=int, default=15)
parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Root directory for generated images")
parser.add_argument("--classifier_dir", type=str, default='classifier_ckpt/flux',
                    help='Path to the trained FLUX classifier directory')
args = parser.parse_args()

pipe = FluxPipeline_custom.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.setting(
    direction_path=args.classifier_dir,
    num_head=args.num_head,
)

data = []
with open(f'geneval/prompts/prompts/evaluation_metadata_{args.geneval_type}.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

end_str = f'_end{args.intervention_end}' if args.intervention_end is not None else ''
run_name = f'osi_alpha{args.alpha}_head{args.num_head}{end_str}_seed{args.seed}'
output_dir = os.path.join(args.output_dir, args.geneval_type, run_name, 'samples')
os.makedirs(output_dir, exist_ok=True)

for idx, d in tqdm(enumerate(data)):
    img_path = os.path.join(output_dir, f'{d["prompt"]}_{idx:06d}.png')
    if os.path.exists(img_path):
        continue

    try:
        out = pipe(
            prompt=d['prompt'],
            num_inference_steps=30,
            guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            width=1024,
            height=1024,
            target_tokens=[c['class'] for c in d['include']],
            alpha=args.alpha,
            intervention_end=15 if args.intervention_end is None else args.intervention_end,
        )
        out.images[0].save(img_path)
    except Exception as e:
        print(e)
        continue
