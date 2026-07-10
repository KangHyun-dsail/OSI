# GitHub 공개 진행 현황

## GitHub 리포

https://github.com/KangHyun-dsail/OSI

---

## 완료된 작업

### 코드 정리
- `classifier_geneval_human/`, `classifier_geneval_human_sd3_512/` 폴더를 `classifier_ckpt/flux/`, `classifier_ckpt/sd3/`로 통합 및 평탄화 (`/key/`, `/t5/` 서브디렉토리 제거)
- `flux_osi_modules/pipeline.py`, `sd3_osi_modules/pipeline.py`에서 불필요한 `model_type`, `sorting_name` 파라미터 제거, 파일 경로를 새 구조에 맞게 수정
- 모든 생성 스크립트(`generate_demo.py`, `generate_osi_main_table.py`, `generate_compbench.py`, `sd3_generate_geneval.py`, `sd3_generate_compbench.py`)에서 `SAMPLING` 서브디렉토리 제거 및 경로 기본값 업데이트

### 신규 파일
- `generate_demo_sd3.py` — SD3.5-Medium 단일 이미지 데모
- `collect_data_demo.py` — 단일 프롬프트에 대해 이미지 + key 벡터 저장하는 데이터 수집 데모
- `figure/thumbnail.png` — 논문 thumbnail (PDF → PNG 변환)
- `.gitignore` — allowlist 방식, 지정 파일만 추적

### README
- 논문 제목: "Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers"
- ICML 2026, arXiv: https://arxiv.org/abs/2605.14270
- 썸네일, 수정된 classifier 경로, Roadmap 체크리스트 반영

### GitHub 업로드 (초기 릴리즈)
현재 업로드된 파일 목록:
```
classifier_ckpt/flux/accuracy.pkl
classifier_ckpt/flux/weight.pkl
classifier_ckpt/sd3/accuracy.pkl
classifier_ckpt/sd3/weight.pkl
figure/thumbnail.png
flux_osi_modules/models.py
flux_osi_modules/pipeline.py
sd3_osi_modules/models.py
sd3_osi_modules/pipeline.py
collect_data_demo.py
generate_demo.py
generate_demo_sd3.py
requirements.txt
README.md
.gitignore
```

---

## 완료된 작업 (2차 릴리즈: dataset/training/evaluation 파이프라인)

### classifier training pipeline
- [x] `generate_dataset_flux.py` — FLUX 학습 데이터 수집 (배치, `--num_samples` 노출)
- [x] `generate_dataset_sd3.py` — SD3 학습 데이터 수집 (배치)
- [x] MM-BLIP 라벨링 파이프라인 코드 — flux_kontext의 kv 라벨러를 포팅:
      `label_mask2former.py`(flat samples MM), `label_blip.py`(flat samples BLIP),
      `label_merge.py`(consensus 병합), `feature_extract.py`(텐서 추출)
- [x] `train_classifier_multi_object_real.py` — classifier 학습 코드
      (출력을 inference가 바로 읽는 flat `weight.pkl`/`accuracy.pkl`로 정리)

### 벤치마크 생성 & 평가
- [x] `generate_osi_main_table.py` — FLUX GenEval 배치 생성
- [x] `generate_compbench.py` — FLUX T2I-CompBench 생성
- [x] `sd3_generate_geneval.py` — SD3 GenEval 배치 생성
- [x] `sd3_generate_compbench.py` — SD3 T2I-CompBench 생성
- [x] `evaluate_geneval.py` — GenEval 평가

### 오케스트레이션 & 배포
- [x] `scripts/01_collect_data.sh` … `scripts/06_eval_geneval.sh` — 모듈형 stage 스크립트
      (env 전환은 하나의 마스터로 묶지 않고 README가 순서 안내)
- [x] 벤치마크 의존성 처리: upstream clone + `benchmark_patches/` 오버레이 방식,
      `benchmark_patches/README.md`에 문서화 (서브모듈/setup.sh 미사용)
- [x] `assets/object_names.txt` — geneval 미커밋에 따른 데이터 의존 파일 동봉
- [x] `.gitignore` allowlist 확장, `docs/01`·`docs/02` 문서 드리프트 수정
- [x] `docs/`, `INSTALL.md`, `RELEASE_PLAN.md` 추적 및 README 파이프라인 섹션 추가

---

## 남은 작업 / 확인 필요

- [x] MM 라벨링을 stock geneval `evaluate_images.py`(중첩 구조 요구) 대신 flat-samples용
      `label_mask2former.py`로 교체 → stage 2 정상화
- [x] `label_mask2former.py`/`label_blip.py`/`label_merge.py`/`feature_extract.py`를
      8샘플 스모크 테스트로 실행 검증 (GPU0, diffuser/geneval/compbench env) — 모두 정상 동작.
      단, feature_extract balancing은 클래스별 present+absent 쌍이 필요하므로 실제 weight.pkl
      산출에는 대규모(수천) 샘플 필요 (train_classifier 자체는 기존 검증된 코드).
- [ ] 대규모(예: 3000) 데이터로 weight.pkl 실제 산출 재현 (수 시간 소요, 필요 시)
- [ ] CompBench 평가(BLIP-VQA / CLIPScore / UniDet) 실행 절차 문서화 보강
