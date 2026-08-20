import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dacon.config import TrainConfig


def log_experiment(
    cfg: TrainConfig,
    fold_aucs: list[float],
    oof_auc: float,
    log_path: Path,
) -> dict:
    """실험 1회의 설정과 결과를 JSONL 한 줄로 append한다.

    JSONL을 쓰는 이유: 실행마다 한 줄씩 안전하게 덧붙일 수 있고, config에 필드가
    늘어도 과거 줄과 충돌하지 않는다. 나중에 pandas.read_json(lines=True)로 비교 가능.
    """
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "oof_auc": round(float(oof_auc), 6),
        "fold_aucs": [round(float(a), 6) for a in fold_aucs],
        "n_splits": cfg.n_splits,
        "seed": cfg.seed,
        "feature": asdict(cfg.feature),
        "model": asdict(cfg.model),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
