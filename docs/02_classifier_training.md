# 분류기 학습 파이프라인

데이터 생성 → MM-BLIP 레이블링 → 데이터 정제 → MMS 분류기 학습의 4단계 파이프라인이다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `generate_dataset_flux.py` | 이미지 + key 벡터 생성 |
| `label_mask2former.py` | Mask2Former로 객체 존재 여부 레이블링 (flat samples) |
| `label_blip.py` | BLIP-VQA 레이블링 (flat samples) |
| `label_merge.py` | MM+BLIP 레이블 consensus 병합 |
| `feature_extract.py` | key 텐서 추출 및 train/test 분할 |
| `train_classifier_multi_object_real.py` | MMS 방향 벡터 학습 및 저장 (논문 사용) |

객체 이름은 파일명(`"a X and an Y_{idx}.png"`)에서 파싱하므로 별도 manifest가 필요 없다.

---

## 1단계: MM-BLIP 이중 레이블링

이미지당 각 객체에 대해 Mask2Former(MM)와 BLIP-VQA 두 모델로 레이블을 생성한다.
두 스크립트 모두 `data/training/samples/`의 평평한 PNG를 직접 순회한다.

### MM 레이블링 (Mask2Former)

`label_mask2former.py`로 각 이미지에서 COCO 객체를 검출한다. mmdet + Mask2Former 체크포인트만
있으면 되고 geneval 리포는 필요 없다.

```bash
conda activate geneval
python label_mask2former.py \
    --samples_dir ./data/training/samples \
    --out ./data/training/mask2former_labels.json \
    --model_path geneval/checkpoints
```

결과: `mask2former_labels.json` — `{"a dog and an elephant_000000": {"dog": 1, "elephant": 0}, ...}`

### BLIP-VQA 레이블링

`label_blip.py`로 각 객체에 대해 `"a <obj>?"`를 질의하고 raw yes-probability를 저장한다.
T2I-CompBench의 stock `BLIP/train_vqa_func.py`를 사용한다.

```bash
conda activate compbench
python label_blip.py \
    --samples_dir ./data/training/samples \
    --out ./data/training/blip_labels.json \
    --blipvqa_dir T2I-CompBench/BLIPvqa_eval
```

결과: `blip_labels.json` — `{"a dog and an elephant_000000": {"dog": 0.98, "elephant": 0.05}, ...}`

### 레이블 병합 (consensus)

MM=1이고 BLIP>present_thresh(0.9)이면 present, MM=0이고 BLIP<absent_thresh(0.3)이면 absent,
그 외에는 None(discard).

```bash
conda activate osi
python label_merge.py \
    --mask2former_labels ./data/training/mask2former_labels.json \
    --blip_labels ./data/training/blip_labels.json \
    --samples_dir ./data/training/samples \
    --output gen_text_real_two_object_512_MM_BLIP.jsonl
```

결과: `gen_text_real_two_object_512_MM_BLIP.jsonl`
```json
{"filename": "./data/training/samples/a dog and an elephant_000000.png", "matched_groups": "{\"dog\": 1, \"elephant\": 0}"}
```

---

## 2단계: 특징 추출 (`feature_extract.py`)

JSONL 파일과 pkl 파일을 읽어 분류기 학습용 텐서로 변환한다.

```bash
python feature_extract.py \
    --mm_blip_jsonl gen_text_real_two_object_512_MM_BLIP.jsonl \
    --output_dir ./data/features \
    --train_ratio 0.89
```

### 처리 로직

```python
# pkl에서 특정 timestep t, layer l, head h의 key 벡터 추출
# shape: (N, 128) — N개 샘플, 128차원
torch.save(x_train, f"{output_dir}/x_train_key_{t}_{l}_{h}.pt")
torch.save(x_test,  f"{output_dir}/x_test_key_{t}_{l}_{h}.pt")
```

결과물 경로:
```
./data/features/
├── x_train_key_{t}_{l}_{h}.pt  # (N_train, 128) float
├── x_test_key_{t}_{l}_{h}.pt   # (N_test, 128) float
├── y_train_data.pkl             # List[int] — 0: absent, 1: present
└── y_test_data.pkl
```

`t`: 타임스텝 값 (652, 814)
`l`: 레이어 인덱스 0~56
`h`: 헤드 인덱스 0~23

---

## 3단계: MMS 분류기 학습 (`train_classifier_multi_object_real.py`)

### 사용 타임스텝

논문에서는 652와 814 두 타임스텝의 key 벡터를 합쳐서 학습한다.

### MMS 방향 벡터 계산

각 `(layer, head)` 쌍마다 key 벡터의 **클래스 평균 차이(Mass Mean Shift)**를 방향 벡터로 사용한다:

```python
mu_true  = x_train[y == 1].mean(dim=0)   # present 샘플의 평균 key (128차원)
mu_false = x_train[y == 0].mean(dim=0)   # absent  샘플의 평균 key (128차원)
direction = mu_true - mu_false            # 방향 벡터 δ = E[k|y=1] - E[k|y=0]
```

### `mms_norm_sigma` 계산 (논문 사용)

```python
direction_norm = direction / direction.norm()
sigma = (x_train @ direction_norm).std()
weight[:128] = direction_norm * sigma    # 스케일링된 방향 벡터
weight[128]  = -(proj_true + proj_false) / 2  # 바이어스
```

### 실행 예시

```bash
python train_classifier_multi_object_real.py \
    --input_dir ./data/features \
    --output_dir classifier_ckpt/flux_reproduced
```

### 저장 형식

```python
# weight.pkl — shape: (57, 24, 129)
#   [:, :, :128] = 방향 벡터 (128차원)
#   [:, :, 128]  = 바이어스 (-(proj_true + proj_false) / 2)
# accuracy.pkl — [((l, h), (train_acc, test_acc)), ...]  정확도 순위 리스트
```

결과물 (inference 파이프라인이 바로 로드하는 flat 구조):
```
classifier_ckpt/flux_reproduced/
├── weight.pkl     # 방향 벡터 + 바이어스
└── accuracy.pkl   # 정확도 순위 리스트
```

> 리포에 동봉된 사전학습 분류기는 `classifier_ckpt/flux`, `classifier_ckpt/sd3`.
> 추론 시 `--classifier_dir <이 디렉토리>`로 지정하면 `weight.pkl`/`accuracy.pkl`을 읽는다.
> SD3.5 분류기는 `train_classifier_multi_object_real.py`의 `timesteps`를 `[738, 831]`로,
> `feature_extract.py` 입력을 SD3 데이터로 바꿔 재현한다.

---

## 전체 흐름 요약

```
generate_dataset_flux.py
  → ./data/training/samples/*.png
  → ./data/training/datas/*.pkl

label_mask2former.py  → ./data/training/mask2former_labels.json (MM 결과)
label_blip.py         → ./data/training/blip_labels.json        (BLIP 결과)

label_merge.py
  → gen_text_real_two_object_512_MM_BLIP.jsonl

feature_extract.py
  → ./data/features/x_{train,test}_key_{t}_{l}_{h}.pt
  → ./data/features/y_{train,test}_data.pkl

train_classifier_multi_object_real.py
  → classifier_ckpt/flux_reproduced/weight.pkl
  → classifier_ckpt/flux_reproduced/accuracy.pkl
```

> 전체 파이프라인을 스테이지 스크립트로 실행하는 순서는 최상위 [README.md](../README.md)
> "Reproducing the classifier" 섹션 참고 (`scripts/01_collect_data.sh` … `04_train_pipeline.sh`).
