# Inference & Steering Pipeline (SD3.5)

학습된 MMS 분류기를 사용해 생성 시 T5 토큰의 key 벡터를 조작하여 누락된 개념의 생성을 유도한다.
FLUX 파이프라인(`docs/03_inference_steering.md`)과 동일한 API 구조를 따른다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `sd3_generate_and_save_real.py` | 데이터 수집용 이미지 + hidden state 생성 |
| `sd3_generate_compbench.py` | T2I-CompBench 이미지 생성 |
| `sd3_osi_modules/pipeline.py` | `StableDiffusion3Pipeline_custom` — 스티어링/수집 파이프라인 |
| `sd3_osi_modules/models.py` | `JointAttnProcessor2_0_custom` 등 커스텀 forward 함수 |

> `sd3_saving_modules/`는 통합 전 구버전이므로 미사용.

---

## FLUX와의 주요 차이

| 항목 | FLUX | SD3.5 |
|------|------|-------|
| 텍스트 인코더 | T5 전용 | T5 전용 (CLIP 포함 구조이나 T5만 사용) |
| T5 토큰 오프셋 | `target_idx` 직접 | `target_idx + num_image_tokens + 77` (CLIP 77토큰 존재) |
| head_dim | 128 | 64 |
| direction 텐서 shape | `(57, 24, 129)` | `(24, 24, 65)` |
| direction 사용 크기 | `[:128]` | `[:64]` |
| steer 대상 batch | `key[:, :, target_]` (전체) | `key[1, :, target_idx]` (conditional만) |
| transformer 블록 수 | 19 + 38 = 57 | 24 |
| 분류기 경로 | `classifier_geneval_human/…/key/` | `classifier_geneval_human_sd3_512/…/t5/` |

---

## 1. 파이프라인 설정 (`pipe.setting()`)

### Steer 모드 (OSI 추론)

```python
pipe.setting(
    mode='steer',
    direction_path='classifier_geneval_human_sd3_512/738_831_timesteps',
    model_type='mms_norm_sigma',
    num_head=100,
    sorting_name='mms_accuracy',
)
```

내부 동작:
1. `{direction_path}/t5/{model_type}_weight.pkl` → `self.direction` tensor `(24, 24, 65)` 로드
2. `{direction_path}/t5/{sorting_name}.pkl` → 정확도 순위 리스트 로드
3. 상위 `num_head`개 `(layer, head)` 쌍의 방향 벡터만 남기고 나머지는 0

### Collect 모드 (학습 데이터 수집)

```python
pipe.setting(mode='collect')
```

- `self.direction = None`으로 설정
- 스티어링 없이 T5 key 벡터만 수집
- `direction_path` 불필요

### 파라미터 요약

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `mode` | `'steer'` | `'steer'` 또는 `'collect'` |
| `direction_path` | `None` | 학습된 분류기 디렉토리. `mode='steer'`이면 필수 |
| `model_type` | `'mms_norm_sigma'` | 방향 벡터 pkl 파일명 prefix |
| `num_head` | `100` | 사용할 상위 어텐션 헤드 수 (전체 SD3.5: 24×24 = 576개) |
| `sorting_name` | `'mms_accuracy'` | 헤드 랭킹 pkl 파일명 |

> 인코더 타입(t5)과 헤드 선택 방식(top)은 고정되어 있다.

---

## 2. 토큰 인덱스 탐색 (`find_by_pieces()`)

FLUX와 동일한 T5 SentencePiece 토크나이저 기반 탐색 함수를 사용한다.

```python
prompt_tokens = self.tokenizer_3.tokenize(prompt)
find_by_pieces(prompt_tokens, 'dog')  # → [[1]]
```

SD3.5에서 T5 토큰의 실제 위치 계산:
```
target_idx = [i_ + num_image_tokens + 77 for i_ in target_]
```
- `num_image_tokens`: 패치 임베딩 후 이미지 토큰 수 (해상도에 따라 다름)
- `+77`: SD3.5는 CLIP (77토큰) + T5 순서로 텍스트 토큰이 배열됨

---

## 3. 생성 (`pipe.__call__()`)

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `prompt` | — | 생성 프롬프트 |
| `target_tokens` | `[]` | 스티어링 대상 토큰 문자열 리스트 |
| `alpha` | `7.5` | 스티어링 강도 |
| `intervention_end` | `15` | 스티어링을 멈출 디노이징 스텝 인덱스 |
| `num_inference_steps` | `28` | 전체 디노이징 스텝 수 |
| `guidance_scale` | `7.0` | CFG 스케일 |

### 반환값

- **`mode='steer'`**: `StableDiffusion3PipelineOutput(images=image)` — 이미지만 반환
- **`mode='collect'`**: `(StableDiffusion3PipelineOutput(images=image), target_latents_timestep)` — 이미지 + hidden states

`target_latents_timestep` 구조 (FLUX와 동일):
```python
{
    timestep_int: {
        'key': [
            {token_str: Tensor(batch, 24, 1, 64), ...},  # 블록별 target 토큰의 key 벡터
            ...  # transformer_blocks 24개
        ]
    },
    ...
}
```

---

## 4. 스티어링 적용 (`JointAttnProcessor2_0_custom`)

`mode`에 따라 실행 경로가 분기된다.

```python
# mode='collect': T5 key 벡터 저장만
if mode == 'collect':
    target_idx = [i_ + num_image_tokens + 77 for i_ in target_]
    target_key[token_str] = key[:, :, target_idx].mean(dim=2, keepdim=True).detach().cpu()
    attn.target_key = target_key

# mode='steer': conditional batch(index 1)의 T5 key 벡터만 수정
if mode == 'steer' and direction is not None:
    dir = direction_temp[:, :, :64].unsqueeze(2)  # (1, 24, 1, 64)
    key[1, :, target_idx] = key[1, :, target_idx] - dir * alpha
```

FLUX와 달리 steer 시 conditional batch(`[1]`)만 수정한다 (CFG uncond은 변경하지 않음).

---

## 5. 논문 기본 설정값

| 파라미터 | 값 |
|---------|----|
| `alpha` | `7.5` |
| `num_head` | `100` (전체 576개 중) |
| `intervention_end` | `15` (30 스텝 기준) |
| `num_inference_steps` | `30` |
| `guidance_scale` | `7.0` |
| 해상도 | `512×512` (데이터 수집), `1024×1024` (생성) |

---

## 6. 생성 스크립트 호출 예시

### 데이터 수집

```bash
CUDA_VISIBLE_DEVICES=0 python sd3_generate_and_save_real.py \
  --output_dir ./data/training_sd3 \
  --resolution 512 \
  --num_inference_steps 30
```

### T2I-CompBench

```bash
CUDA_VISIBLE_DEVICES=0 python sd3_generate_compbench.py \
  --alpha 7.5 \
  --num_head 100 \
  --geneval_type color_val_seen_phrase \
  --output_dir ./outputs
```

---

## 7. 데이터 수집 딕셔너리 구조 (FLUX 통일)

FLUX와 SD3.5 데이터를 merged classifier로 학습시키기 위해 딕셔너리 구조를 통일했다.

| 항목 | 구버전 (`sd3_saving_modules`) | 현재 (`sd3_osi_modules`) |
|------|-------------------------------|--------------------------|
| key 이름 | `'clip'`, `'t5'` (두 개) | `'key'` (하나) |
| 수집 인코더 | CLIP + T5 | T5만 |
| batch 처리 | `key[1,...]` (conditional만) | `key[:, ...]` (전체 평균) |
| 텐서 shape | `(1, 24, 1, 64)` | `(batch, 24, 1, 64)` |
