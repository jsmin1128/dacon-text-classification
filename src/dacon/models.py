import os

import numpy as np
from xgboost import XGBClassifier

from dacon.config import SEED

# RTX 3090(24GB)에서 GPU 학습이 가능하도록 조정한 값(실측 기반).
# TF-IDF는 넓은 sparse 데이터라 XGBoost GPU가 max_bin에 매우 민감하다.
# 실측(피처 7만, 78K행): max_bin=128은 14.9GB 단일 할당을 시도하다 OOM,
#                        max_bin=64는 GPU 메모리 ~1.2GB만 쓰고 정상 완주(AUC 동일).
# 환경변수로 재정의 가능: XGB_DEVICE(cpu/cuda), XGB_MAX_DEPTH, XGB_MAX_BIN.
DEVICE = os.environ.get("XGB_DEVICE", "cuda")
MAX_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", "6"))
MAX_BIN = int(os.environ.get("XGB_MAX_BIN", "64"))


def make_xgb_classifier(y: np.ndarray) -> XGBClassifier:
    pos_w = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=DEVICE,
        max_bin=MAX_BIN,
        learning_rate=0.05,
        n_estimators=1200,
        max_depth=MAX_DEPTH,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_lambda=3.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=SEED,
        scale_pos_weight=pos_w,
        early_stopping_rounds=100,
    )
