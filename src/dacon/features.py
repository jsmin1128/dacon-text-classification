import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.preprocessing import StandardScaler

from dacon.config import FeatureConfig


def length_feats(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["full_text"]
    return pd.DataFrame(
        {
            "len_char": txt.str.len(),
            "len_word": txt.fillna("").str.split().str.len(),
            "ratio_digit": txt.str.count(r"\d") / (txt.str.len() + 1),
            "ratio_upper": txt.str.count(r"[A-Z]") / (txt.str.len() + 1),
            "ratio_punc": txt.str.count(r"[\.\,\!\?\;]") / (txt.str.len() + 1),
            "ratio_sym": txt.str.count(r"[%\$\@\#\&]") / (txt.str.len() + 1),
        }
    )


def concat_title_text(df: pd.DataFrame) -> np.ndarray:
    return (df["title"].fillna("") + " " + df["full_text"].fillna("")).values


def make_tfidf_vectorizers(
    cfg: FeatureConfig | None = None,
) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    # 피처수/ngram/min_df는 전부 FeatureConfig에서 관리한다(GPU 메모리 근거는 config.py 참고).
    cfg = cfg or FeatureConfig()
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=cfg.char_ngram,
        min_df=cfg.char_min_df,
        max_features=cfg.char_max_features,
        dtype=np.float32,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        tokenizer=None,
        ngram_range=cfg.word_ngram,
        max_features=cfg.word_max_features,
        min_df=cfg.word_min_df,
        dtype=np.float32,
    )
    return char_tfidf, word_tfidf


def _build_char_features(
    cfg: FeatureConfig,
    tr_texts: np.ndarray,
    va_texts: np.ndarray,
    te_texts: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """char n-gram 피처. use_hashing이면 어휘 dict 없이 해싱으로 빌드(RAM/속도↓)."""
    if not cfg.use_hashing:
        char_tfidf, _ = make_tfidf_vectorizers(cfg)
        return (
            char_tfidf.fit_transform(tr_texts),
            char_tfidf.transform(va_texts),
            char_tfidf.transform(te_texts),
        )

    hasher = HashingVectorizer(
        analyzer="char",
        ngram_range=cfg.char_ngram,
        n_features=cfg.char_hash_features,
        alternate_sign=False,  # 음수 항 방지 -> 이후 IDF 가중과 XGBoost에 적합
        norm=None,  # 정규화는 TfidfTransformer가 담당
        dtype=np.float32,
    )
    tfidf = TfidfTransformer()  # IDF 가중을 유지해 기존 TfidfVectorizer에 근접
    X_char_tr = tfidf.fit_transform(hasher.transform(tr_texts))
    X_char_va = tfidf.transform(hasher.transform(va_texts))
    X_char_te = tfidf.transform(hasher.transform(te_texts))
    return X_char_tr, X_char_va, X_char_te


def build_fold_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    num_feats_train: pd.DataFrame,
    num_feats_test: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    cfg = cfg or FeatureConfig()
    tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
    _, word_tfidf = make_tfidf_vectorizers(cfg)
    scaler = StandardScaler(with_mean=False)

    X_char_tr, X_char_va, X_char_te = _build_char_features(
        cfg,
        concat_title_text(tr_df),
        concat_title_text(val_df),
        concat_title_text(test),
    )

    X_word_tr = word_tfidf.fit_transform(tr_df["full_text"])
    X_word_va = word_tfidf.transform(val_df["full_text"])
    X_word_te = word_tfidf.transform(test["full_text"])

    X_num_tr = scaler.fit_transform(num_feats_train.iloc[tr_idx])
    X_num_va = scaler.transform(num_feats_train.iloc[val_idx])
    X_num_te = scaler.transform(num_feats_test)

    X_tr = sparse.hstack([X_char_tr, X_word_tr, X_num_tr]).tocsr()
    X_va = sparse.hstack([X_char_va, X_word_va, X_num_va]).tocsr()
    X_te = sparse.hstack([X_char_te, X_word_te, X_num_te]).tocsr()
    return X_tr, X_va, X_te
