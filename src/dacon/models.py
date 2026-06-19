import numpy as np
from xgboost import XGBClassifier

from dacon.config import SEED


def make_xgb_classifier(y: np.ndarray) -> XGBClassifier:
    pos_w = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        device="cuda",
        learning_rate=0.05,
        n_estimators=1200,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_lambda=3.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=SEED,
        scale_pos_weight=pos_w,
        early_stopping_rounds=100,
    )
