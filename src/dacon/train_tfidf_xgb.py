import argparse
import logging
import sys
import warnings
from pathlib import Path

from dacon.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + XGBoost baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dacon.submission import save_submission
    from dacon.training import seed_everything, train_and_predict

    seed_everything()
    preds = train_and_predict(args.data_dir, args.n_splits)
    save_submission(args.data_dir, args.output, preds)
    logging.info("Saved submission: %s", args.output)


if __name__ == "__main__":
    main()
