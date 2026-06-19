from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "submissions" / "ensemble_submission.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
