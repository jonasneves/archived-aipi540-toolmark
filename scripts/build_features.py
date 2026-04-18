"""Feature-engineering helpers used by the classical and deep models.

The LOTCO experiment refits TF-IDF and fine-tunes DistilBERT per fold, so
there are no precomputed feature artifacts on disk — this module provides
the factories and per-fold transforms the training orchestrator calls into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def make_tfidf_vectorizer() -> TfidfVectorizer:
    """Word + char n-grams. Char grams catch tokenizer-evasion paraphrases."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=20_000,
        min_df=2,
        sublinear_tf=True,
    )


def fit_tool_class_encoder(tool_classes: Iterable[str]) -> OneHotEncoder:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    arr = np.asarray(list(tool_classes)).reshape(-1, 1)
    enc.fit(arr)
    return enc


def tool_class_onehot(enc: OneHotEncoder, tool_classes: Iterable[str]) -> sp.csr_matrix:
    arr = np.asarray(list(tool_classes)).reshape(-1, 1)
    return enc.transform(arr).tocsr()


@dataclass(frozen=True)
class FittedFeaturizer:
    """TF-IDF + tool-class one-hot, fit on a training split."""

    vectorizer: TfidfVectorizer
    tool_encoder: OneHotEncoder
    include_tool_feature: bool

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        text_feats = self.vectorizer.transform(df["text"].tolist())
        if not self.include_tool_feature:
            return text_feats.tocsr()
        tool_feats = tool_class_onehot(self.tool_encoder, df["tool_class"])
        return sp.hstack([text_feats, tool_feats]).tocsr()


def fit_featurizer(train_df: pd.DataFrame, include_tool_feature: bool = True) -> FittedFeaturizer:
    vec = make_tfidf_vectorizer()
    vec.fit(train_df["text"].tolist())
    enc = fit_tool_class_encoder(train_df["tool_class"])
    return FittedFeaturizer(vectorizer=vec, tool_encoder=enc, include_tool_feature=include_tool_feature)


def distilbert_input_texts(df: pd.DataFrame, include_tool_feature: bool = True) -> list[str]:
    """Render rows as DistilBERT inputs.

    When `include_tool_feature` is True, prepend `[TOOL: <tool_class>] ` so
    the transformer has access to tool identity. The ablation disables it.
    """
    if not include_tool_feature:
        return df["text"].tolist()
    return [f"[TOOL: {cls}] {text}" for cls, text in zip(df["tool_class"], df["text"])]
