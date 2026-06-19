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
├── src/
│   └── dacon/
│       ├── config.py              # 프로젝트 경로, seed, 기본 출력 경로
│       ├── data.py                # CSV 로드와 테스트 컬럼 정규화
│       ├── features.py            # TF-IDF 및 수치 피처 생성
│       ├── models.py              # XGBoost 모델 생성
│       ├── training.py            # fold 학습, 검증, 테스트 예측
│       ├── submission.py          # 제출 파일 생성
│       └── train_tfidf_xgb.py     # CLI 엔트리포인트
├── requirements.txt
└── README.md
```

## 데이터

학습과 제출에 필요한 CSV 파일은 `data/raw/`에 로컬로 둡니다.

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

가상환경을 만들고 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

필요 패키지:

```text
numpy
pandas
scipy
scikit-learn
xgboost
tqdm
```

## 실행 방법

기본 실행:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb
```

옵션 예시:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb \
  --data-dir data/raw \
  --output outputs/submissions/ensemble_submission.csv \
  --n-splits 5
```

도움말:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb --help
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

기본 제출 파일:

```text
outputs/submissions/ensemble_submission.csv
```

`outputs/` 아래는 학습 결과물 저장용입니다. 필요하면 모델 파일과 캐시 파일도 여기에 둡니다.

## 주의사항

현재 XGBoost 설정은 GPU를 사용합니다.

```python
tree_method="gpu_hist"
```

GPU가 없으면 `src/dacon/models.py`에서 CPU용 설정으로 바꿔야 합니다.

## 검증

문법 확인:

```bash
python -m compileall -q src
```

CLI 확인:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb --help
```
