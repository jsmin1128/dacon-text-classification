# DACON 텍스트 분류 프로젝트

DACON 텍스트 분류 대회를 위한 학습/제출 생성 프로젝트입니다.

기본 파이프라인은 `TF-IDF 문자/단어 n-gram + 길이 기반 수치 피처 + XGBoost` 조합입니다.

## 프로젝트 구조

```text
.
├── data/
│   └── raw/                       # 로컬 데이터 위치
├── outputs/
│   ├── cache/                     # 피처/임베딩 캐시 저장 위치
│   ├── models/                    # 모델 저장 위치
│   └── submissions/               # 제출 파일 저장 위치
├── scripts/
│   └── train_tfidf_xgb.py         # CLI 엔트리포인트
├── src/
│   └── dacon/
│       ├── config.py              # 경로/seed + 하이퍼파라미터 dataclass(Feature/Model/Train)
│       ├── data.py                # CSV 로드와 테스트 컬럼 정규화
│       ├── features.py            # TF-IDF/수치 피처 생성 (config 기반)
│       ├── models.py              # XGBoost 모델 생성 (config 기반)
│       ├── training.py            # fold 학습, 검증, 테스트 예측
│       ├── experiment.py          # 실행별 설정/AUC를 JSONL로 기록
│       └── submission.py          # 제출 파일 생성
├── tests/
│   └── test_pipeline.py           # 소규모 데이터 스모크 테스트
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 데이터

데이터는 아래 Google Drive에서 다운로드 후 `data/raw/`에 위치시킵니다.

[Google Drive 데이터 폴더](https://drive.google.com/drive/folders/1RVRPvS0hFV637F2cD7aNigUj9R-_b7nT?usp=sharing)

필수 파일:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

예상 컬럼:

```text
train.csv: title, full_text, generated
test.csv: ID, title, paragraph_index, paragraph_text
sample_submission.csv: ID, generated
```

`test.csv`의 `paragraph_text`는 코드에서 `full_text`로 맞춥니다.

## 환경 설정

Python 3.10 이상이 필요합니다.

macOS에서는 XGBoost 실행에 `libomp`가 필요합니다:

```bash
brew install libomp
```

가상환경을 만들고 의존성을 설치합니다:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

결과를 완전히 재현하려면 `requirements.txt`로 버전을 고정합니다. 최신 버전으로 설치하려면 `pip install -e .`만 실행합니다.

## 실행 방법

기본 실행:

```bash
python scripts/train_tfidf_xgb.py
```

옵션 예시:

```bash
python scripts/train_tfidf_xgb.py \
  --data-dir data/raw \
  --output outputs/submissions/ensemble_submission.csv \
  --n-splits 5
```

데이터가 바뀐 경우 캐시를 무시하고 재계산합니다:

```bash
python scripts/train_tfidf_xgb.py --no-cache
```

## 학습 흐름

1. `data.py`에서 train/test CSV를 로드합니다.
2. test의 `paragraph_text` 컬럼명을 `full_text`로 정규화합니다.
3. `features.py`에서 fold별 TF-IDF vectorizer와 scaler를 fit합니다.
4. 문자 TF-IDF, 단어 TF-IDF, 수치 피처를 합칩니다.
5. `StratifiedKFold`로 fold를 나누고 XGBoost를 학습합니다. (train의 title이 전부 고유해 그룹 분할은 무의미하므로 클래스 층화만 적용)
6. fold별 validation AUC와 전체 OOF AUC를 출력합니다.
7. 실행 설정과 AUC를 `outputs/experiments.jsonl`에 한 줄 기록합니다.
8. fold별 test prediction을 평균내어 제출 파일을 저장합니다.

사용 피처(기본값, `config.py`의 `FeatureConfig`에서 관리):

```text
char TF-IDF: title + full_text, char 3-4 gram, max_features=50000
word TF-IDF: full_text, word 1-2 gram, max_features=20000
numeric: len_char, len_word, ratio_digit, ratio_upper, ratio_punc, ratio_sym
```

> 피처 폭과 char n-gram 범위는 RTX 3090(24GB) GPU 학습이 되도록 실측으로 조정한 값입니다.
> char 5gram은 빌드 RAM/시간의 대부분을 먹으면서 정확도 기여가 거의 없어 3-4로 낮췄습니다.

## 산출물

학습 완료 후 생성되는 파일:

```text
outputs/submissions/ensemble_submission.csv   # 제출 파일
outputs/models/fold_1.json ~ fold_N.json      # fold별 학습 모델 (N = n-splits)
outputs/cache/fold_1_tr.npz ...               # fold별 TF-IDF 행렬 캐시
outputs/experiments.jsonl                      # 실행별 설정/AUC 기록 (실험 비교용)
```

캐시가 있으면 두 번째 실행부터 TF-IDF 재계산을 건너뜁니다. 데이터나 **피처 설정(char n-gram 등)** 이 바뀌면 캐시가 오래된 상태가 되므로 `--no-cache` 플래그를 쓰거나 `outputs/cache/`를 직접 비웁니다.

## 설정과 실험 관리

하이퍼파라미터는 `src/dacon/config.py`의 dataclass에 모여 있습니다:

- `FeatureConfig` — TF-IDF n-gram, max_features, min_df, hashing 토글
- `ModelConfig` — XGBoost의 device/max_depth/max_bin/learning_rate 등
- `TrainConfig` — n_splits, seed와 위 둘을 묶는 최상위 설정

실험을 바꾸려면 이 값들을 수정하면 됩니다. 실행마다 사용한 설정과 fold/OOF AUC가 `outputs/experiments.jsonl`에 한 줄씩 쌓여, 나중에 어떤 설정이 몇 점이었는지 비교할 수 있습니다.

## 테스트

파이프라인 배선을 소규모 데이터로 검증하는 스모크 테스트가 있습니다(실제 데이터·GPU 불필요, 수초 소요):

```bash
pip install -e ".[dev]"   # pytest 설치
pytest
```

## 주의사항 (GPU)

기본 설정은 RTX 3090(24GB) 기준으로 XGBoost GPU 학습(`device="cuda"`)이 되도록 조정돼 있습니다. 값을 코드 수정 없이 바꾸려면 환경변수를 씁니다:

```bash
XGB_DEVICE=cpu python scripts/train_tfidf_xgb.py      # GPU 없이 CPU로 실행
XGB_MAX_BIN=64 XGB_MAX_DEPTH=6 python scripts/train_tfidf_xgb.py
```

`max_bin`이 GPU 메모리를 좌우합니다(넓은 sparse TF-IDF에서 매우 민감). 24GB에서 OOM이 나면 `XGB_MAX_BIN`을 낮추세요.

