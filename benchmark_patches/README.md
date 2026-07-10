# Benchmark setup

The GenEval and T2I-CompBench benchmark repositories are **not** committed to this
repo (each is a large upstream clone with its own license and weights). Instead, clone
the upstream repositories yourself and overlay the OSI-specific files kept here.

## 1. Clone the upstream benchmarks (from the repo root)

```bash
git clone https://github.com/djghosh13/geneval.git
git clone https://github.com/Karine-Huang/T2I-CompBench.git
git clone -b 2.x https://github.com/open-mmlab/mmdetection.git   # GenEval detector backend
```

## 2. Overlay the OSI files

Copy everything under `benchmark_patches/` onto the freshly cloned repos, preserving
paths:

```bash
cp -r benchmark_patches/geneval/.       geneval/
cp -r benchmark_patches/T2I-CompBench/. T2I-CompBench/
```

## 3. What each overlaid file is

**GenEval** (`benchmark_patches/geneval/`)
- `prompts/prompts/evaluation_metadata_*.jsonl`, `generation_prompts_*.txt` —
  the OSI prompt subsets (`two_object_100` … `six_object_100`, `single_object`) read by
  `generate_osi_main_table.py` / `sd3_generate_geneval.py` / `evaluate_geneval.py`.
- `prompts/create_prompts.py` — generator used to produce the above (for provenance).

**T2I-CompBench** (`benchmark_patches/T2I-CompBench/`)
- `examples/dataset/*.json` — the CompBench prompt sets read by `generate_compbench.py`
  / `sd3_generate_compbench.py` (`color_val_seen_phrase`, `texture_val_seen_phrase`, …).

> BLIP-VQA training-data labeling (`label_blip.py`, used by `scripts/03_label_blip.sh`)
> calls the **stock** `T2I-CompBench/BLIPvqa_eval/BLIP/train_vqa_func.py` — no overlay
> needed beyond cloning the upstream repo.

## 4. Model weights (download separately)

- **GenEval detector** — Mask2Former
  `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth` → place in `geneval/checkpoints/`.
- **BLIP-VQA** — downloads automatically on first run of `BLIP_vqa_revised.py`.
- **CompBench UniDet / CLIPScore** — follow the upstream T2I-CompBench README for the
  spatial / numeracy / non-spatial evaluators.

See `INSTALL.md` for the matching conda environments (`geneval`, `compbench`).
