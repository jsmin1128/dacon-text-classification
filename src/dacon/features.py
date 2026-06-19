import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def length_feats(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["full_text"]
    return pd.DataFrame(
        {
            "len_char": txt.str.len(),
            "len_word": txt.str.split().apply(len),
            "ratio_digit": txt.str.count(r"\d") / (txt.str.len() + 1),
            "ratio_upper": txt.str.count(r"[A-Z]") / (txt.str.len() + 1),
            "ratio_punc": txt.str.count(r"[\.\,\!\?\;]") / (txt.str.len() + 1),
            "ratio_sym": txt.str.count(r"[%\$\@\#\&]") / (txt.str.len() + 1),
        }
    )


def concat_title_text(df: pd.DataFrame) -> np.ndarray:
    return (df["title"].fillna("") + " " + df["full_text"].fillna("")).values


def make_tfidf_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=200_000,
        dtype=np.float32,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        tokenizer=None,
        ngram_range=(1, 2),
        max_features=40_000,
        min_df=2,
        dtype=np.float32,
    )
    return char_tfidf, word_tfidf


def build_fold_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    num_feats_train: pd.DataFrame,
    num_feats_test: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
    char_tfidf, word_tfidf = make_tfidf_vectorizers()
    scaler = StandardScaler(with_mean=False)

    X_char_tr = char_tfidf.fit_transform(concat_title_text(tr_df))
    X_char_va = char_tfidf.transform(concat_title_text(val_df))
    X_char_te = char_tfidf.transform(concat_title_text(test))

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
