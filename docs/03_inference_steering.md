# Inference & Steering Pipeline (FLUX)

학습된 MMS 분류기를 사용해 생성 시 key 벡터를 조작하여 누락된 개념의 생성을 유도한다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `generate_demo.py` | 단일 이미지 데모 생성 |
| `generate_osi_main_table.py` | FLUX GenEval 이미지 생성 (논문 기본) |
| `generate_compbench.py` | FLUX T2I-CompBench 이미지 생성 |
| `flux_osi_modules/pipeline.py` | `FluxPipeline_custom` — 스티어링 파이프라인 |
| `flux_osi_modules/models.py` | `FluxAttnProcessor_custom` 등 커스텀 forward 함수 |

---

## 1. 파이프라인 설정 (`pipe.setting()`)

`__call__` 전에 반드시 한 번 호출해야 한다. `mode`에 따라 동작이 달라진다.

### Steer 모드 (OSI 추론)

```python
pipe.setting(
    mode='steer',
    direction_path='classifier_geneval_human/652_814_timesteps_gen_text_real_two_object_mm_blip',
    model_type='mms_norm_sigma',   # 방향 벡터 종류
    num_head=300,                  # 상위 헤드 수 (전체 FLUX: 1368개)
    sorting_name='mms_accuracy',   # 헤드 랭킹 기준
)
```

내부 동작:
1. `{direction_path}/key/mms_norm_sigma_weight.pkl` → `self.direction` tensor `(57, 24, 129)` 로드
2. `{direction_path}/key/mms_accuracy.pkl` → 정확도 순위 리스트 로드
3. 상위 `num_head`개 `(layer, head)` 쌍의 방향 벡터만 남기고 나머지는 0

### Collect 모드 (학습 데이터 수집)

```python
pipe.setting(mode='collect')
```

- `self.direction = None`으로 설정
- 스티어링 없이 key 벡터만 수집
- `direction_path` 불필요

### 파라미터 요약

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `mode` | `'steer'` | `'steer'` 또는 `'collect'` |
| `direction_path` | `None` | 학습된 분류기 디렉토리. `mode='steer'`이면 필수 |
| `model_type` | `'mms_norm_sigma'` | 방향 벡터 pkl 파일명 prefix |
| `num_head` | `100` | 사용할 상위 어텐션 헤드 수 |
| `sorting_name` | `'mms_accuracy'` | 헤드 랭킹 pkl 파일명 |

> key 타입(key/value)과 head selection 방식(top/bottom/random)은 key-top으로 고정되어 있다.

---

## 2. 토큰 인덱스 탐색 (`find_by_pieces()`)

T5 토크나이저의 SentencePiece 서브워드 토큰에서 target 단어의 인덱스 범위들을 반환한다.

```python
prompt_tokens = self.tokenizer_2.tokenize(prompt)
# 예: "a dog and an elephant"
# → ['▁a', '▁dog', '▁and', '▁an', '▁elephant']

find_by_pieces(prompt_tokens, 'dog')      # → [[1]]
find_by_pieces(prompt_tokens, 'elephant') # → [[4]]
```

- 단어 경계(`▁`)를 공백으로 간주해 멀티토큰 단어도 매칭 가능
- 동일 단어가 여러 위치에 있으면 모든 위치를 반환

---

## 3. 생성 (`pipe.__call__()`)

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `prompt` | — | 생성 프롬프트 |
| `target_tokens` | `[]` | 스티어링 대상 토큰 문자열 리스트 (예: `['dog', 'cat']`) |
| `alpha` | `10.0` | 스티어링 강도. 값이 클수록 해당 객체를 강하게 유도 |
| `intervention_end` | `10` | 스티어링을 멈출 디노이징 스텝 인덱스 |
| `num_inference_steps` | `28` | 전체 디노이징 스텝 수 |
| `guidance_scale` | `3.5` | CFG 스케일 |

### 반환값

- **`mode='steer'`**: `FluxPipelineOutput(images=image)` — 이미지만 반환
- **`mode='collect'`**: `(FluxPipelineOutput(images=image), target_latents_timestep)` — 이미지 + hidden states

`target_latents_timestep` 구조:
```python
{
    timestep_int: {
        'key': [
            # transformer_blocks (19개) + single_transformer_blocks (38개) 순서
            {token_str: Tensor(1, 24, 1, 128), ...},  # 블록별 target 토큰의 key 벡터
            ...
        ]
    },
    ...
}
```

---

## 4. 스티어링 적용 (`FluxAttnProcessor_custom`)

`mode`에 따라 실행 경로가 분기된다.

```python
# mode='collect': key 벡터 저장만
if mode == 'collect':
    target_key[token_str] = key[:, :, target_].mean(dim=2, keepdim=True).detach().cpu()
    attn.target_key = target_key

# mode='steer': key 벡터 수정만
if mode == 'steer' and direction is not None:
    dir = direction_temp[:, :, :128].unsqueeze(2)  # (1, 24, 1, 128)
    key[:, :, target_] = key[:, :, target_] - dir * alpha
```

`direction = mu_true - mu_false`이므로 `alpha > 0`이면 present 방향으로 key를 밀어 해당 객체 생성을 유도한다.

스티어링은 스텝 `i < intervention_end` 구간에서만 `actual_alpha = alpha`가 되고, 이후에는 `actual_alpha = 0.0`이 된다.

---

## 5. 논문 기본 설정값

| 파라미터 | 값 |
|---------|----|
| `alpha` | `5.0` |
| `num_head` | `300` (전체 1368개 중) |
| `intervention_end` | `15` (30 스텝 기준) |
| `num_inference_steps` | `30` |
| `guidance_scale` | `3.5` |
| 해상도 | `1024×1024` |

---

## 6. 생성 스크립트 호출 예시

### 데모 (단일 이미지)

```bash
CUDA_VISIBLE_DEVICES=0 python generate_demo.py \
  --prompt "a dog and a cat" \
  --target_tokens dog cat \
  --alpha 5.0 \
  --intervention_end 15
```

### GenEval

```bash
CUDA_VISIBLE_DEVICES=0 python generate_osi_main_table.py \
  --alpha 5.0 \
  --num_head 300 \
  --geneval_type two_object_100 \
  --output_dir ./outputs \
  --seed 42
```

출력 경로: `./outputs/{geneval_type}/osi_alpha5.0_head300_seed42/samples/`

`--geneval_type` 옵션: `two_object_100`, `three_object_100`, `four_object_100`, `five_object_100`, `six_object_100`

### T2I-CompBench

```bash
CUDA_VISIBLE_DEVICES=0 python generate_compbench.py \
  --alpha 5.0 \
  --num_head 300 \
  --geneval_type color_val_seen_phrase \
  --output_dir ./outputs
```

`--geneval_type` suffix 의미:
- `_phrase`: object + attribute 토큰 모두 스티어링 (논문 기본값)
- `_attribute`: attribute 토큰만 스티어링
- (없음): object 토큰만 스티어링

---

## 7. 멀티 GPU 배치 실행

`generate_osi.sh` 또는 아래처럼 직접 분산:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_osi_main_table.py --geneval_type two_object_100 &
CUDA_VISIBLE_DEVICES=1 python generate_osi_main_table.py --geneval_type three_object_100 &
CUDA_VISIBLE_DEVICES=2 python generate_osi_main_table.py --geneval_type four_object_100 &
wait
```
