# 데이터 생성 파이프라인

분류기 학습에 필요한 (이미지, key 벡터, 레이블) 데이터를 생성하는 파이프라인이다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `generate_dataset_flux.py` (SD3.5: `generate_dataset_sd3.py`) | 이미지 + key 벡터 생성 및 저장 (메인 스크립트) |
| `flux_osi_modules/pipeline.py` | `FluxPipeline_custom.__call__()` — key 벡터 수집 |
| `flux_osi_modules/models.py` | `FluxAttnProcessor_custom.__call__()` — key 벡터 추출 |

---

## 1. 프롬프트 생성 규칙

`generate_dataset_flux.py`에서 랜덤 2-object 프롬프트를 생성한다 (`--num_samples`개).

```
object_num = 2
objects 풀: assets/object_names.txt  (약 80개 MS-COCO 클래스)

프롬프트 형식: "{article} {obj1} and {article} {obj2}"
  예: "a dog and an elephant"
      "an apple and a chair"
  - article: 'an' (모음으로 시작) or 'a' (자음으로 시작)
  - 두 객체가 같지 않도록 중복 제거

파일명에 프롬프트가 포함됨: "{prompt}_{idx:06d}.png"
  예: "a dog and an elephant_000042.png"
```

**실행 예시:**
```bash
CUDA_VISIBLE_DEVICES=0 python generate_dataset_flux.py \
  --output_dir ./data/training \
  --num_samples 3000
# 또는 스테이지 스크립트로: NUM_SAMPLES=3000 GPU=0 bash scripts/01_collect_data.sh
```
(resolution 512, num_inference_steps 20, guidance_scale 3.5는 스크립트에 고정)

---

## 2. 파이프라인에서 key 벡터가 수집되는 위치

### 2-1. `models.py` — `FluxAttnProcessor_custom.__call__()`

어텐션 계산 직전, RoPE 적용 후의 key 벡터에서 target 토큰의 벡터를 추출해 `attn.target_key`에 저장한다.

```python
# models.py
for idx, target_ in enumerate(target_tokens):
    target_key[target_tokens_eng[idx]] = key[:,:,target_].mean(dim=2, keepdim=True).detach().cpu()
    # key shape: (B=1, H=24, seq_len, head_dim=128)
    # target_ : 해당 토큰의 인덱스들 (리스트, 여러 토큰 서브워드 가능)
    # 저장 형태: (1, 24, 1, 128) — 24개 헤드, head_dim=128
attn.target_key = target_key  # dict: {object_name: tensor (1,24,1,128)}
```

> key는 QK-norm + RoPE 적용 후 값이다. 어텐션 스코어 계산에 실제로 사용되는 key와 동일.

### 2-2. `pipeline.py` — `__call__()` 디노이징 루프

매 timestep마다 모든 57개 블록(dual 19 + single 38)에서 `target_key`를 수집해 딕셔너리에 쌓는다.

```python
# pipeline.py
target_head = {'key': [], 'value': []}
for block in self.transformer.transformer_blocks:       # 19개 dual block
    target_head['key'].append(block.attn.target_key)
for block in self.transformer.single_transformer_blocks:  # 38개 single block
    target_head['key'].append(block.attn.target_key)

target_latents_timestep[int(t.item())] = target_head
```

파이프라인 반환값 (collect 모드):
```python
return FluxPipelineOutput(images=image), target_latents_timestep
# 호출 시: out, hidden_states = pipe(...)
```

---

## 3. 저장되는 데이터 구조

### 이미지
```
{output_dir}/samples/{prompt}_{idx:06d}.png
```

### key 벡터 pkl 파일
```
{output_dir}/datas/{prompt}_{idx:06d}.pkl
```

pkl 파일 내용 (`hidden_states` = `target_latents_timestep`):
```python
{
  1000: {                          # 첫 번째 timestep 값 (정수)
    'key': [
      # 인덱스 0~18: dual transformer blocks (19개)
      # 인덱스 19~56: single transformer blocks (38개)
      { 'dog': tensor(1, 24, 1, 128),
        'elephant': tensor(1, 24, 1, 128) },  # block 0
      { 'dog': tensor(1, 24, 1, 128), ... },  # block 1
      ...                                      # 총 57개
    ],
    'value': []  # 비어 있음 (수집 안 함)
  },
  972: { 'key': [...], 'value': [] },  # 두 번째 timestep
  ...
  # 총 num_inference_steps (20) 개의 timestep
}
```

> **주의:** `generate_dataset_flux.py`에서는 `pipe.setting(mode='collect')`으로 호출 — 스티어링 없이 key 벡터만 수집·저장한다.

---

## 4. 저장 경로 정리

```
{output_dir}/                  # default: ./data/training
├── samples/
│   └── a dog and an elephant_000000.png
└── datas/
    └── a dog and an elephant_000000.pkl
```

---

## 5. 다음 단계

생성된 이미지에 MM-BLIP 이중 검증으로 레이블을 붙인다:
1. `label_mask2former.py` → Mask2Former 객체 감지 → `mask2former_labels.json`
2. `label_blip.py` → BLIP-VQA 검증 → `blip_labels.json`
3. `label_merge.py` → 두 결과 consensus 병합 → `gen_text_real_two_object_512_MM_BLIP.jsonl`

자세한 내용: [02_classifier_training.md](02_classifier_training.md)
