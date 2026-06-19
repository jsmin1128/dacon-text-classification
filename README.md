# DACON 텍스트 분류 프로젝트

DACON 텍스트 분류 대회를 위한 재현 가능한 학습/제출 생성 프로젝트입니다.

현재 기본 파이프라인은 `TF-IDF 문자/단어 n-gram + 길이 기반 수치 피처 + XGBoost` 조합입니다. 노트북 중심 실험 구조는 제거했고, 실행 가능한 코드는 `src/dacon/` 아래에 모듈별로 분리했습니다.

## 프로젝트 구조

```text
.
├── data/
│   └── raw/
│       ├── .gitkeep               # 디렉터리 유지용 파일
│       ├── train.csv              # 로컬에 직접 배치, Git 추적 제외
│       ├── test.csv               # 로컬에 직접 배치, Git 추적 제외
│       └── sample_submission.csv  # 로컬에 직접 배치, Git 추적 제외
├── outputs/
│   ├── cache/                     # 피처/임베딩 캐시 저장용 예약 디렉터리
│   ├── models/                    # 모델 파일 저장용 예약 디렉터리
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

원본 데이터는 Git에 포함하지 않습니다. DACON에서 직접 다운로드한 뒤 `data/raw/`에 둡니다.

필수 파일은 다음 3개입니다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

현재 코드는 다음 컬럼을 전제로 합니다.

```text
train.csv: title, full_text, generated
test.csv: ID, title, paragraph_index, paragraph_text
sample_submission.csv: ID, generated
```

`test.csv`의 `paragraph_text`는 로드 단계에서 `full_text`로 이름을 맞춥니다.

`data/raw/`의 CSV 파일은 `.gitignore`로 제외됩니다. 저장소를 새로 받은 경우 직접 데이터를 내려받아 아래처럼 배치하세요.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## 환경 설정

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

필요 패키지는 [requirements.txt](requirements.txt)에 정의되어 있습니다.

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

옵션을 명시한 실행:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb \
  --data-dir data/raw \
  --output outputs/submissions/ensemble_submission.csv \
  --n-splits 5
```

도움말 확인:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb --help
```

## 학습 파이프라인

기본 학습 흐름은 다음과 같습니다.

1. `data.py`에서 train/test CSV를 로드합니다.
2. test의 `paragraph_text` 컬럼명을 `full_text`로 정규화합니다.
3. `features.py`에서 fold마다 TF-IDF vectorizer와 scaler를 학습 fold에만 fit합니다.
4. 문자 TF-IDF, 단어 TF-IDF, 수치 피처를 sparse matrix로 결합합니다.
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

기본 제출 파일 저장 위치:

```text
outputs/submissions/ensemble_submission.csv
```

`outputs/`는 Git 추적에서 제외됩니다. 제출 파일, 모델 파일, 캐시 파일은 로컬 산출물로 관리합니다.

현재 기본 학습 스크립트는 제출 파일만 저장합니다. 모델 저장이나 피처 캐싱이 필요하면 `outputs/models/`, `outputs/cache/`를 사용하도록 별도 로직을 추가하면 됩니다.

## 주의사항

현재 XGBoost 설정은 GPU 사용을 전제로 합니다.

```python
tree_method="gpu_hist"
```

CUDA/GPU 환경이 없으면 학습 실행이 실패할 수 있습니다. CPU 환경에서 실행하려면 `src/dacon/models.py`의 `tree_method` 설정을 CPU용으로 바꿔야 합니다.

또한 `requirements.txt`는 현재 버전을 고정하지 않습니다. 같은 결과를 반복적으로 재현해야 한다면 패키지 버전을 pinning하는 것이 좋습니다.

## 검증

코드 문법 확인:

```bash
python -m compileall -q src
```

CLI 진입점 확인:

```bash
PYTHONPATH=src python -m dacon.train_tfidf_xgb --help
```
