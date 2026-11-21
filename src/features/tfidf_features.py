"""Sparse representations: Bag-of-Words and TF-IDF (word n-grams + optional char n-grams).

* BoW (CountVectorizer): raw term counts. Every occurrence counts equally, so frequent
  but uninformative words ("issue", "please") get large weights.
* TF-IDF: term frequency x inverse document frequency. Words that appear in many tickets
  get down-weighted, distinctive words ("challan", "eduroam") get up-weighted.
  ``sublinear_tf`` uses 1+log(tf) so a word repeated 5 times is not 5x as important.
* Char n-grams (char_wb 2-5) make the model robust to typos ("pasword", "recieved") since
  misspelt words still share most character n-grams with the correct spelling.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class WordCharTfidf(BaseEstimator, TransformerMixin):
    """Word TF-IDF, optionally concatenated with char TF-IDF.

    Exposed as one estimator so a grid search can tune ``use_char`` and the nested
    vectorizer parameters (e.g. ``tfidf__word__ngram_range``).
    """

    def __init__(self, word: TfidfVectorizer | None = None, char: TfidfVectorizer | None = None, use_char: bool = False):
        self.word = word
        self.char = char
        self.use_char = use_char

    def fit(self, X, y=None):
        self.word_ = clone(self.word).fit(X)
        self.char_ = clone(self.char).fit(X) if self.use_char else None
        return self

    def transform(self, X):
        mats = [self.word_.transform(X)]
        if self.char_ is not None:
            mats.append(self.char_.transform(X))
        return sp.hstack(mats).tocsr() if len(mats) > 1 else mats[0]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.word_.get_feature_names_out())
        if self.char_ is not None:
            names += [f"char:{n}" for n in self.char_.get_feature_names_out()]
        return np.asarray(names, dtype=object)


def build_tfidf(max_features: int = 30000, ngram_range=(1, 2), min_df: int = 1, use_char: bool = False) -> WordCharTfidf:
    word = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, max_features=max_features,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=max_features, sublinear_tf=True)
    return WordCharTfidf(word=word, char=char, use_char=use_char)


def build_bow(max_features: int = 30000, ngram_range=(1, 1), min_df: int = 1) -> CountVectorizer:
    return CountVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, max_features=max_features)


def top_features_per_class(pipeline, n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Most positive coefficients per class for a linear model (LR / LinearSVC).

    Works for Pipeline([... ('tfidf', vec), ('clf', linear model)]). For calibrated SVMs the
    coefficients of the underlying LinearSVCs are averaged.
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    names = vec.get_feature_names_out()
    if hasattr(clf, "coef_"):
        coef, classes = clf.coef_, clf.classes_
    elif hasattr(clf, "calibrated_classifiers_"):
        coef = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
        classes = clf.classes_
    elif hasattr(clf, "feature_log_prob_"):  # Naive Bayes: log P(word|class) minus average
        coef = clf.feature_log_prob_ - clf.feature_log_prob_.mean(axis=0)
        classes = clf.classes_
    else:
        raise ValueError(f"Cannot extract features from {type(clf).__name__}")
    out = {}
    for i, c in enumerate(classes):
        idx = np.argsort(coef[i])[::-1][:n]
        out[str(c)] = [(str(names[j]), round(float(coef[i, j]), 4)) for j in idx]
    return out
