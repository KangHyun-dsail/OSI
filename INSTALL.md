# Installation Guide

## Main Environment (generation & training)

```bash
conda create -n osi python=3.10 -y
conda activate osi

# PyTorch — adjust CUDA version to match your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8: replace cu121 with cu118

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Benchmark repositories

The `geneval`, `T2I-CompBench`, and `mmdetection` repos are cloned separately and then
overlaid with the OSI files in `benchmark_patches/`. Do this **before** the environment
steps below — see [benchmark_patches/README.md](benchmark_patches/README.md).

## GenEval Evaluation Environment

GenEval uses mmdetection 2.x which requires older PyTorch (1.13.1):

```bash
conda create -n geneval python=3.9 -y
conda activate geneval

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

cd mmdetection && pip install -e . --no-build-isolation && cd ..

pip install open-clip-torch==2.20.0 clip-benchmark==1.4.0 pandas tqdm
pip install "numpy<2.0" "opencv-python==4.8.1.78"
```

Download the Mask2Former checkpoint:
```bash
mkdir -p geneval/checkpoints
# Download mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth to geneval/checkpoints/
```

## T2I-CompBench Evaluation Environment

```bash
conda create -n compbench python=3.9 -y
conda activate compbench
# Follow T2I-CompBench/README.md for setup
```

## Aesthetic Quality Evaluation Environment

```bash
conda create -n pyiqa python=3.10 -y
conda activate pyiqa
pip install pyiqa torch torchvision
```

## Verify Installation

```bash
conda activate osi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import spacy; print('spaCy OK')"
```
