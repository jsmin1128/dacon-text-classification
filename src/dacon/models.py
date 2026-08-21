import numpy as np
from xgboost import XGBClassifier

from dacon.config import SEED, ModelConfig


def make_xgb_classifier(y: np.ndarray, cfg: ModelConfig | None = None) -> XGBClassifier:
    cfg = cfg or ModelConfig.from_env()
    pos_w = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=cfg.device,
        max_bin=cfg.max_bin,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        min_child_weight=cfg.min_child_weight,
        random_state=SEED,
        scale_pos_weight=pos_w,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
