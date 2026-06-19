import gc
import logging
import random
from pathlib import Path

import numpy as np
from scipy import sparse
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


def _load_fold_cache(cache_dir: Path, fold: int):
    paths = [cache_dir / f"fold_{fold}_{split}.npz" for split in ("tr", "va", "te")]
    if all(p.exists() for p in paths):
        return tuple(sparse.load_npz(p) for p in paths)
    return None


def _save_fold_cache(cache_dir: Path, fold: int, X_tr, X_va, X_te) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, mat in zip(("tr", "va", "te"), (X_tr, X_va, X_te)):
        sparse.save_npz(cache_dir / f"fold_{fold}_{name}.npz", mat)


def train_and_predict(
    data_dir: Path,
    n_splits: int,
    models_dir: Path,
    cache_dir: Path,
    use_cache: bool = True,
) -> np.ndarray:
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

    models_dir.mkdir(parents=True, exist_ok=True)

    for fold, (tr_idx, val_idx) in enumerate(tqdm_folds):
        fold_num = fold + 1

        cached = _load_fold_cache(cache_dir, fold_num) if use_cache else None
        if cached:
            X_tr, X_va, X_te = cached
            tqdm_folds.set_description(f"Fold {fold_num}/{n_splits} cache hit")
        else:
            tqdm_folds.set_description(f"Fold {fold_num}/{n_splits} prepare")
            X_tr, X_va, X_te = build_fold_matrices(
                train, test, tr_idx, val_idx, num_feats_train, num_feats_test,
            )
            if use_cache:
                _save_fold_cache(cache_dir, fold_num, X_tr, X_va, X_te)

        model = make_xgb_classifier(y[tr_idx])
        tqdm_folds.set_description(f"Fold {fold_num}/{n_splits} train")
        model.fit(
            X_tr,
            y[tr_idx],
            eval_set=[(X_va, y[val_idx])],
            verbose=False,
        )

        model.save_model(models_dir / f"fold_{fold_num}.json")

        val_pred = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y[val_idx], val_pred)
        logging.info("  Fold %s AUC = %.5f", fold_num, fold_auc)

        oof[val_idx] = val_pred
        preds += model.predict_proba(X_te)[:, 1] / n_splits

        del X_tr, X_va, X_te, model
        gc.collect()

    logging.info("\nALL FOLD OOF AUC = %.5f", roc_auc_score(y, oof))
    return preds.clip(0, 1)
