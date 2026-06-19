from pathlib import Path

import pandas as pd


def load_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig").rename(
        columns={"paragraph_text": "full_text"}
    )
    return train, test


def load_sample_submission(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
