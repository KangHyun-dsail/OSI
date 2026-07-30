# Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers

> Official code for the paper **"Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers"**
> ICML 2026 | [arXiv](https://arxiv.org/abs/2605.14270)


## Introduction

![Thumbnail](assets/thumbnail.png)

Multimodal Diffusion Transformers (MM-DiTs) have achieved remarkable progress in text-to-image generation, yet they frequently suffer from concept omission, where specified objects or attributes fail to emerge in the generated image. By performing linear probing on text tokens, we demonstrate that text embeddings can distinguish a characteristic `omission signal' representing the absence of target concepts. Leveraging this insight, we propose Omission Signal Intervention (OSI), which amplifies the omission signal to actively catalyze the generation of missing concepts. Comprehensive experiments on FLUX.1-Dev and SD3.5-Medium demonstrate that OSI significantly alleviates concept omission even in extreme scenarios.

## Setup

The inference demo and classifier training only need the main `osi` environment:

```bash
conda create -n osi python=3.10 -y
conda activate osi

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The GenEval / T2I-CompBench evaluation stages use separate environments
(`geneval`, `compbench`) because they depend on older, mutually incompatible stacks.
See **[INSTALL.md](INSTALL.md)** for all four environments (`osi`, `geneval`,
`compbench`, `pyiqa`).

## Pretrained Classifiers

Pre-trained classifiers are included in `classifier_ckpt/`:

| Model | Path |
|-------|------|
| FLUX.1-dev | `classifier_ckpt/flux/` |
| SD3.5-Medium | `classifier_ckpt/sd3/` |

## Quick Start

### FLUX.1-dev

```bash
CUDA_VISIBLE_DEVICES=0 python generate_demo_flux.py \
  --prompt "a dog and a cat" \
  --target_tokens dog cat \
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (required) | Text prompt |
| `--target_tokens` | (required) | Space-separated target concepts |
| `--alpha` | `5.0` | Steering strength |
| `--num_head` | `300` | Number of attention heads to steer |
| `--intervention_end` | `15` | Stop steering at this denoising step (out of 30) |
| `--classifier_dir` | `classifier_ckpt/flux` | Path to classifier directory |
| `--seed` | `42` | Random seed |
| `--output` | `{prompt}_{seed}.png` | Output image path |

### SD3.5-Medium

```bash
CUDA_VISIBLE_DEVICES=0 python generate_demo_sd3.py \
  --prompt "a dog and a cat" \
  --target_tokens dog cat \
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (required) | Text prompt |
| `--target_tokens` | (required) | Space-separated target concepts |
| `--alpha` | `7.5` | Steering strength |
| `--num_head` | `100` | Number of attention heads to steer |
| `--intervention_end` | `15` | Stop steering at this denoising step (out of 30) |
| `--classifier_dir` | `classifier_ckpt/sd3` | Path to classifier directory |
| `--seed` | `42` | Random seed |
| `--output` | `{prompt}_{seed}.png` | Output image path |

## Data Collection Demo

`collect_data_demo_flux.py` shows how we collect key vectors for classifier training. For a single prompt, it saves both the generated image and the attention key hidden states:

```bash
CUDA_VISIBLE_DEVICES=0 python collect_data_demo_flux.py \
  --prompt "a dog and a cat" \
  --target_tokens dog cat \
  --output_dir ./collected
# → collected/a dog and a cat_42.png
# → collected/a dog and a cat_42.pkl  (key vectors per timestep/layer/head)
```

## Reproducing the Classifier

The full training pipeline is split into modular stage scripts under `scripts/`.
Because the stages span multiple conda environments, run them **in order**, activating
the environment noted for each stage (there is intentionally no single master script
that switches environments). All scripts accept overrides via environment variables and
are safe to run stage-by-stage.

| Stage | Env | Command | Output |
|-------|-----|---------|--------|
| 1. Collect data | `osi` | `bash scripts/01_collect_data.sh` | `data/training/{samples,datas}` |
| 2a. MM label | `geneval` | `bash scripts/02_label_mm.sh` | `data/training/mask2former_labels.json` |
| 2b. BLIP label | `compbench` | `bash scripts/03_label_blip.sh` | `data/training/blip_labels.json` |
| 3-5. Train | `osi` | `bash scripts/04_train_pipeline.sh` | `classifier_ckpt/flux_reproduced/{weight,accuracy}.pkl` |

Stage 2 labels operate directly on the flat `data/training/samples/*.png`; object names
are parsed from the filenames (no manifest needed). Stage 3-5 chains
`label_merge.py` → `feature_extract.py` → `train_classifier_multi_object_real.py`.

```bash
# Stage 1 — collect key vectors (small run for a smoke test: NUM_SAMPLES=8)
conda activate osi
NUM_SAMPLES=3000 GPU=0 bash scripts/01_collect_data.sh

# Stage 2 — dual MM + BLIP labeling (needs the benchmark repos + weights, see below)
conda activate geneval  && bash scripts/02_label_mm.sh
conda activate compbench && bash scripts/03_label_blip.sh

# Stages 3-5 — merge labels, extract features, train the MMS classifier
conda activate osi
bash scripts/04_train_pipeline.sh
```

The resulting `classifier_ckpt/flux_reproduced/` is directly loadable at inference via
`--classifier_dir`. For SD3.5, pass `MODEL=sd3` to stage 1 and adjust the training
timesteps accordingly; the attention layer/head counts differ between the two models
(FLUX.1-dev 57 layers, SD3.5-Medium 24) but `feature_extract.py` infers them from the
collected pickles, so stages 3-5 need no further changes.

## Benchmark Setup

GenEval and T2I-CompBench are **not** committed here; clone the upstream repos and overlay
the OSI-specific files kept in `benchmark_patches/`. Full instructions (clone commands, which
files to copy, and which model weights to download) are in
**[benchmark_patches/README.md](benchmark_patches/README.md)**.

## Running the Benchmarks

After the benchmarks are set up (above):

```bash
# Generate steered images (env: osi). MODEL=flux|sd3, BENCH=geneval|compbench
conda activate osi
MODEL=flux BENCH=geneval GENEVAL_TYPE=two_object_100 GPU=0 bash scripts/05_generate.sh

# Evaluate GenEval results (env: geneval) — pass the SAME params as generation
conda activate geneval
MODEL=flux GENEVAL_TYPE=two_object_100 GPU=0 bash scripts/06_eval_geneval.sh
```

CompBench evaluation uses several upstream tools (BLIP-VQA / CLIPScore / UniDet); see
`benchmark_patches/README.md` and the T2I-CompBench README. Image-quality metrics
(MUSIQ / MANIQA) are computed by `evaluate_aesthetic.py` in the `pyiqa` env.

## Roadmap

- [x] Inference demo (FLUX.1-dev, SD3.5-Medium)
- [x] Pre-trained classifiers
- [x] Data collection demo
- [x] Data labeling pipeline
- [x] Classifier training code
- [x] Full benchmark generation & evaluation scripts


## Repository Structure

```
OSI/
├── classifier_ckpt/
│   ├── flux/                  # Pre-trained FLUX.1-dev classifier
│   └── sd3/                   # Pre-trained SD3.5-Medium classifier
├── flux_osi_modules/          # FluxPipeline_custom / FluxAttnProcessor_custom
├── sd3_osi_modules/           # StableDiffusion3Pipeline_custom / SD3AttnProcessor_custom
├── scripts/                   # Modular pipeline stage runners (01…06)
├── benchmark_patches/         # OSI overlay files for cloned geneval / T2I-CompBench
├── assets/                    # Object class pool (object_names.txt), README thumbnail
├── generate_demo_flux.py      # FLUX.1-dev single-image demo
├── generate_demo_sd3.py       # SD3.5-Medium single-image demo
├── collect_data_demo_flux.py  # Data collection demo (key vector saving)
├── generate_dataset_flux.py   # Batch data collection (FLUX)
├── generate_dataset_sd3.py    # Batch data collection (SD3.5)
├── label_mask2former.py       # MM (Mask2Former) presence labeling
├── label_blip.py              # BLIP-VQA presence labeling
├── label_merge.py             # Consensus-merge MM + BLIP labels
├── feature_extract.py         # Extract key tensors, train/test split
├── train_classifier_multi_object_real.py  # MMS direction-vector training
├── generate_geneval_flux.py   # FLUX GenEval batch generation
├── generate_geneval_sd3.py    # SD3.5 GenEval generation
├── generate_compbench_flux.py # FLUX T2I-CompBench generation
├── generate_compbench_sd3.py  # SD3.5 T2I-CompBench generation
├── evaluate_geneval.py        # GenEval scoring
├── evaluate_aesthetic.py      # MUSIQ / MANIQA quality metrics
├── INSTALL.md                 # All four conda environments
├── requirements.txt
└── README.md
```



## Citation

```bibtex
@article{baek2026diagnosing,
  title={Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers},
  author={Baek, Kanghyun and Lew, Jaihyun and Shin, Chaehun and Lee, Jungbeom and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2605.14270},
  year={2026}
}
```
