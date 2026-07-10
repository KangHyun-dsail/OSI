# 평가 파이프라인

GenEval 및 T2I-CompBench 기반으로 생성 이미지를 정량 평가한다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `evaluate_geneval.py` | GenEval 평가 스크립트 |
| `geneval/evaluation/evaluate_images.py` | GenEval 원본 평가 스크립트 |
| `geneval/evaluation/summary_scores.py` | 평가 결과 요약 |
| `evaluate_aesthetic/pyiqa_test.py` | 이미지 품질 평가 (MUSIQ / MANIQA) |

---

## 환경 설정

### GenEval 환경 (`geneval`)

```bash
conda create -n geneval python=3.9 -y
conda activate geneval

# PyTorch 1.13.1 + CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# mmcv-full (mmdet 2.x 의존성)
pip install mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# mmdetection 2.x (로컬 소스 설치)
cd mmdetection && pip install -e . --no-build-isolation && cd ..

# 나머지 패키지
pip install open-clip-torch==2.20.0 clip-benchmark==1.4.0 pandas tqdm
pip install "numpy<2.0" "opencv-python==4.8.1.78"
```

### 필요 체크포인트

```
geneval/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

---

## GenEval 평가

### 실행

```bash
conda activate geneval
CUDA_VISIBLE_DEVICES=0 python evaluate_geneval.py \
    --name osi_alpha5.0_head300_seed42 \
    --geneval_type two_object_100 \
    --input_dir ./outputs \
    --output_dir ./results
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--name` | `osi_alpha5.0_head300_seed42` | 평가할 실험 폴더 이름 |
| `--geneval_type` | `two_object_100` | 평가 태스크 종류 |
| `--input_dir` | `./outputs` | 이미지 루트 디렉토리 |
| `--output_dir` | `./results` | 결과 저장 루트 디렉토리 |
| `--model_path` | `geneval/checkpoints` | Mask2Former 체크포인트 경로 |

### geneval_type 옵션

| 값 | 설명 |
|----|------|
| `two_object_100` | 2-object 조합 100개 |
| `three_object_100` | 3-object 조합 100개 |
| `four_object_100` | 4-object 조합 100개 |
| `five_object_100` | 5-object 조합 100개 |
| `six_object_100` | 6-object 조합 100개 |

### 결과 형식

```
./results/{geneval_type}/{name}.jsonl
```

```bash
# 정확도 확인
python geneval/evaluation/summary_scores.py ./results/two_object_100/osi_alpha5.0_head300_seed42.jsonl
```

---

## T2I-CompBench 평가

T2I-CompBench는 속성별로 다른 평가 도구를 사용한다.

### BLIP-VQA (color, texture, shape)

```bash
conda activate compbench
cd T2I-CompBench/BLIPvqa_eval
python BLIP_vqa.py \
    --out_dir ../../outputs/color_val_seen/osi_alpha5.0_head300_phrase_seed42 \
    --np_num 8
```

### CLIP Score (non-spatial)

```bash
cd T2I-CompBench
python CLIPScore_eval/CLIP_similarity.py \
    --outpath ./outputs/non_spatial_val/osi_alpha5.0_head300_seed42
```

---

## 이미지 품질 평가 (Aesthetic)

MUSIQ 및 MANIQA 지표로 생성 이미지 품질을 평가한다.

### 실행

```bash
conda activate pyiqa  # pyiqa 설치된 환경 필요
cd evaluate_aesthetic
CUDA_VISIBLE_DEVICES=0 python pyiqa_test.py \
    --geneval_type two_object_100 \
    --name osi_alpha5.0_head300_seed42 \
    --metric musiq \
    --input_dir ../outputs \
    --output_dir ../results
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--name` | `osi_alpha5.0_head300_seed42` | 평가할 실험 폴더 이름 |
| `--geneval_type` | `two_object_100` | 평가 태스크 종류 |
| `--input_dir` | `./outputs` | 이미지 루트 디렉토리 |
| `--output_dir` | `./results` | 결과 저장 루트 디렉토리 |
| `--metric` | `musiq` | 평가 지표 (`musiq` 또는 `maniqa`) |

### 결과 형식

```
./results/{geneval_type}/{name}_{metric}.txt
```

### 평가 지표

| 지표 | 설명 | 범위 |
|------|------|------|
| MUSIQ | Multi-scale image quality transformer | 0–100 (높을수록 좋음) |
| MANIQA | Multi-dimension attention network for IQA | 0–1 (높을수록 좋음) |
