# DACON 텍스트 분류 프로젝트

AI 생성 텍스트 판별(binary, AUC 평가) 대회용 학습/제출 파이프라인입니다.
구성은 `TF-IDF(문자·단어 n-gram) + 길이 수치 피처 → XGBoost 5-fold 앙상블`.

## 프로젝트 구조

```text
├── data/raw/                    # 로컬 데이터 (직접 다운로드)
├── outputs/                     # 산출물 (제출/모델/캐시/실험로그)
├── scripts/train_tfidf_xgb.py   # CLI 엔트리포인트
└── src/dacon/
    ├── config.py                # 경로/seed + 하이퍼파라미터 dataclass
    ├── data.py                  # CSV 로드, 테스트 컬럼 정규화
    ├── features.py              # TF-IDF/수치 피처 생성
    ├── models.py                # XGBoost 모델 생성
    ├── training.py              # fold 학습/검증/예측
    ├── experiment.py            # 실행별 설정/AUC를 JSONL로 기록
    ├── logging_setup.py         # tqdm 친화 로깅
    └── submission.py            # 제출 파일 생성
```

## 데이터

[Google Drive](https://drive.google.com/drive/folders/1RVRPvS0hFV637F2cD7aNigUj9R-_b7nT?usp=sharing)에서 받아 `data/raw/`에 둡니다.

```text
train.csv: title, full_text, generated
test.csv:  ID, title, paragraph_index, paragraph_text   # paragraph_text → full_text로 처리
sample_submission.csv: ID, generated
```

## 설치

Python 3.10 이상이 필요합니다.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 검증된 버전으로 재현
pip install -e .
```

## 실행

```bash
python scripts/train_tfidf_xgb.py             # 기본 5-fold
python scripts/train_tfidf_xgb.py --no-cache  # 피처 캐시 무시하고 재계산
```

옵션: `--data-dir` `--output` `--models-dir` `--cache-dir` `--n-splits` `--no-cache`

> **기대 결과:** 기본 설정(5-fold, RTX 3090) 기준 OOF AUC ≈ 0.9987, 캐시 없을 때 ~35분.

## 동작

`StratifiedKFold`로 5-fold를 나누고, fold마다 **train 부분에만** vectorizer/scaler를 fit해
문자·단어 TF-IDF와 수치 피처를 합쳐 XGBoost를 학습합니다. test는 fold 예측의 평균으로 앙상블합니다.
(train의 title이 전부 고유해 그룹 분할은 불필요하므로 클래스 층화만 적용)

산출물:

```text
outputs/submissions/ensemble_submission.csv  # 제출 파일
outputs/models/fold_*.json                    # fold별 모델
outputs/cache/fold_*.npz                       # 피처 캐시 (--no-cache면 미생성)
outputs/experiments.jsonl                      # 실행별 설정/AUC 기록
```

## 설정 (GPU)

하이퍼파라미터는 `src/dacon/config.py`의 dataclass(`FeatureConfig`/`ModelConfig`/`TrainConfig`)에 모여 있습니다.
기본값은 RTX 3090(24GB) GPU 학습에 맞춘 값이며, 코드 수정 없이 환경변수로 바꿀 수 있습니다:

```bash
XGB_DEVICE=cpu python scripts/train_tfidf_xgb.py    # GPU 없이 CPU로 실행
XGB_MAX_BIN=48 python scripts/train_tfidf_xgb.py    # OOM이 나면 낮춤
```

`max_bin`이 GPU 메모리를 좌우합니다(넓은 sparse TF-IDF에서 매우 민감).
