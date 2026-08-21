import os
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "submissions" / "ensemble_submission.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
DEFAULT_EXPERIMENT_LOG = PROJECT_ROOT / "outputs" / "experiments.jsonl"


@dataclass
class FeatureConfig:
    """TF-IDF 피처 설정. 기본값은 RTX 3090(24GB) GPU 학습에 맞춘 실측 최적값."""

    # 5gram은 빌드 RAM/시간을 크게 먹으면서 AUC 기여가 거의 없어 제외(0.99807->0.99804).
    char_ngram: tuple[int, int] = (3, 4)
    # 넓은 sparse를 XGBoost GPU DMatrix가 감당하도록 200K에서 축소.
    char_max_features: int = 50_000
    char_min_df: int = 3
    word_ngram: tuple[int, int] = (1, 2)
    word_max_features: int = 20_000
    word_min_df: int = 2

    # HashingVectorizer는 빌드는 빠르나 행렬 밀도가 높아져 3090에서 GPU OOM을 유발(실측).
    # 기본은 False, CPU/선형모델로 갈 때만 사용.
    use_hashing: bool = False
    char_hash_features: int = 2**16


@dataclass
class ModelConfig:
    """XGBoost 설정. max_bin=64/ max_depth=6이 3090에서 GPU OOM을 피하는 핵심."""

    device: str = "cuda"
    max_depth: int = 6
    max_bin: int = 64
    learning_rate: float = 0.05
    n_estimators: int = 1200
    subsample: float = 0.9
    colsample_bytree: float = 0.6
    reg_lambda: float = 3.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    early_stopping_rounds: int = 100

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """환경변수(XGB_DEVICE/XGB_MAX_DEPTH/XGB_MAX_BIN)로 일부 값을 재정의한다."""
        cfg = cls()
        cfg.device = os.environ.get("XGB_DEVICE", cfg.device)
        cfg.max_depth = int(os.environ.get("XGB_MAX_DEPTH", cfg.max_depth))
        cfg.max_bin = int(os.environ.get("XGB_MAX_BIN", cfg.max_bin))
        return cfg


@dataclass
class TrainConfig:
    """실험 1회를 완전히 규정하는 최상위 설정. 여기만 바꾸면 실험이 바뀐다."""

    n_splits: int = 5
    seed: int = SEED
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig.from_env)
