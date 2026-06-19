import gc
import logging
import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from dacon.config import SEED
from dacon.data import load_train_test
from dacon.features import build_fold_matrices, length_feats
from dacon.models import make_xgb_classifier


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_and_predict(data_dir, n_splits: int) -> np.ndarray:
    train, test = load_train_test(data_dir)
    num_feats_train = length_feats(train)
    num_feats_test = length_feats(test)

    y = train["generated"].values
    groups = train["title"].fillna("").values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    splits = list(sgkf.split(train, y, groups))
    tqdm_folds = tqdm(splits, desc="Folds")

    for fold, (tr_idx, val_idx) in enumerate(tqdm_folds):
        tqdm_folds.set_description(f"Fold {fold + 1}/{n_splits} prepare")
        X_tr, X_va, X_te = build_fold_matrices(
            train,
            test,
            tr_idx,
            val_idx,
            num_feats_train,
            num_feats_test,
        )

        model = make_xgb_classifier(y)
        tqdm_folds.set_description(f"Fold {fold + 1}/{n_splits} train")
        model.fit(
            X_tr,
            y[tr_idx],
            eval_set=[(X_va, y[val_idx])],
            verbose=False,
            early_stopping_rounds=100,
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y[val_idx], val_pred)
        logging.info("  Fold %s AUC = %.5f", fold + 1, fold_auc)

        oof[val_idx] = val_pred
        preds += model.predict_proba(X_te)[:, 1] / n_splits

        del X_tr, X_va, X_te, model
        gc.collect()

    logging.info("\nALL FOLD OOF AUC = %.5f", roc_auc_score(y, oof))
    return preds.clip(0, 1)
