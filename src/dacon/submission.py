from pathlib import Path

import numpy as np

from dacon.data import load_sample_submission


def save_submission(data_dir: Path, output_path: Path, preds: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submit = load_sample_submission(data_dir)
    submit["generated"] = preds
    submit.to_csv(output_path, index=False, encoding="utf-8-sig")
