"""파이프라인 스모크 테스트: 소규모 데이터로 배선과 회귀를 빠르게 검증한다.

실제 데이터/GPU 없이 CPU에서 수초 내로 돈다. `pytest`로 실행.
"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from dacon.config import FeatureConfig, ModelConfig, TrainConfig
from dacon.experiment import log_experiment
from dacon.features import build_fold_matrices, length_feats, make_tfidf_vectorizers
from dacon.models import make_xgb_classifier


def _toy_data(n: int = 200):
    """사람글/AI글이 뚜렷이 구분되는 장난감 데이터(분리 가능)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = i % 2
        base = "사람이 직접 쓴 자연스러운 문장 " if label == 0 else "AI generated synthetic text token "
        rows.append(
            {
                "title": f"title_{i}",
                "full_text": base * int(rng.integers(3, 9)),
                "generated": label,
            }
        )
    return pd.DataFrame(rows)


def test_config_defaults_match_tuned_values():
    fc, mc = FeatureConfig(), ModelConfig.from_env()
    assert fc.char_ngram == (3, 4)  # Phase 2에서 5gram 제거
    assert fc.char_max_features == 50_000 and fc.word_max_features == 20_000
    assert fc.use_hashing is False
    assert mc.max_depth == 6 and mc.max_bin == 64


def test_vectorizers_honor_config():
    ct, wt = make_tfidf_vectorizers(FeatureConfig(char_max_features=111, word_max_features=22))
    assert ct.max_features == 111 and wt.max_features == 22


@pytest.mark.parametrize("use_hashing", [False, True])
def test_pipeline_learns_on_toy_data(use_hashing):
    df = _toy_data(200)
    test = df.head(20).copy()
    tr_idx, val_idx = np.arange(0, 160), np.arange(160, 200)
    nft, nfte = length_feats(df), length_feats(test)
    fc = FeatureConfig(
        char_max_features=2000, word_max_features=1000,
        char_min_df=1, word_min_df=1,
        use_hashing=use_hashing, char_hash_features=2**12,
    )
    X_tr, X_va, X_te = build_fold_matrices(df, test, tr_idx, val_idx, nft, nfte, cfg=fc)
    assert X_tr.shape[0] == 160 and X_va.shape[0] == 40 and X_te.shape[0] == 20
    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1]  # 열 정합

    y = df["generated"].values
    m = make_xgb_classifier(y[tr_idx], cfg=ModelConfig(device="cpu", n_estimators=40, early_stopping_rounds=15))
    m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[val_idx])], verbose=False)
    p = m.predict_proba(X_va)[:, 1]
    assert p.shape == (40,) and p.min() >= 0.0 and p.max() <= 1.0
    assert roc_auc_score(y[val_idx], p) > 0.9  # 분리 가능한 데이터라 잘 맞아야 함


def test_experiment_log_appends_jsonl(tmp_path):
    log = tmp_path / "experiments.jsonl"
    cfg = TrainConfig()
    log_experiment(cfg, fold_aucs=[0.99, 0.98], oof_auc=0.985, log_path=log)
    log_experiment(cfg, fold_aucs=[0.97], oof_auc=0.97, log_path=log)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["oof_auc"] == 0.985
    assert rec["feature"]["char_ngram"] == [3, 4]  # asdict가 tuple을 list로 직렬화
    assert rec["model"]["max_bin"] == 64
