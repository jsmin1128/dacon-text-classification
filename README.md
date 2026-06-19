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
│       ├── config.py              # 프로젝트 경로, seed, 기본 출력 경로
│       ├── data.py                # CSV 로드와 테스트 컬럼 정규화
│       ├── features.py            # TF-IDF 및 수치 피처 생성
│       ├── models.py              # XGBoost 모델 생성
│       ├── training.py            # fold 학습, 검증, 테스트 예측
│       └── submission.py          # 제출 파일 생성
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 데이터

데이터는 아래 Google Drive에서 다운로드 후 `data/raw/`에 위치시킵니다.

[Google Drive 데이터 폴더](https://drive.google.com/drive/folders/1bep1Mjw42uRO7M1a5eMcKaf_FZK3fVBO?usp=sharing)

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
5. `StratifiedGroupKFold`로 fold를 나누고 XGBoost를 학습합니다.
6. fold별 validation AUC와 전체 OOF AUC를 출력합니다.
7. fold별 test prediction을 평균내어 제출 파일을 저장합니다.

사용 피처:

```text
char TF-IDF: title + full_text, char 3-5 gram, max_features=200000
word TF-IDF: full_text, word 1-2 gram, max_features=40000
numeric: len_char, len_word, ratio_digit, ratio_upper, ratio_punc, ratio_sym
```

## 산출물

학습 완료 후 생성되는 파일:

```text
outputs/submissions/ensemble_submission.csv   # 제출 파일
outputs/models/fold_1.json ~ fold_N.json      # fold별 학습 모델 (N = n-splits)
outputs/cache/fold_1_tr.npz ...               # fold별 TF-IDF 행렬 캐시
```

캐시가 있으면 두 번째 실행부터 TF-IDF 재계산을 건너뜁니다. 데이터가 바뀌면 `--no-cache` 플래그를 사용하거나 `outputs/cache/`를 직접 비웁니다.

## 주의사항

현재 XGBoost 설정은 GPU를 사용합니다.

```python
device="cuda"
```

GPU가 없으면 `src/dacon/models.py`에서 해당 파라미터를 제거하거나 `device="cpu"`로 바꿔야 합니다.

