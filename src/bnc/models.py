"""Candidate models. Each one is a scikit-learn Pipeline: features -> classifier."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from bnc.preprocess import clean


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean,
        token_pattern=r"\S+",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=150_000,
    )


def char_tfidf() -> TfidfVectorizer:
    # Character n-grams handle Bangla word endings (inflections) without a stemmer
    return TfidfVectorizer(
        preprocessor=clean,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        sublinear_tf=True,
        max_features=150_000,
    )


def build(name: str) -> Pipeline:
    if name == "naive_bayes":
        return Pipeline([("tfidf", word_tfidf()), ("clf", ComplementNB(alpha=0.3))])
    if name == "logistic_regression":
        return Pipeline(
            [("tfidf", word_tfidf()), ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))]
        )
    if name == "linear_svm":
        return Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(C=0.5, class_weight="balanced"))])
    if name == "svm_word_char":
        features = FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])
        # Calibration turns SVM scores into probabilities for the API
        clf = CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced"), cv=3)
        return Pipeline([("tfidf", features), ("clf", clf)])
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["naive_bayes", "logistic_regression", "linear_svm", "svm_word_char"]
