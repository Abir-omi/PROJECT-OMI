"""Load the saved model and explain its predictions."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from bnc.data import LABELS_BN
from bnc.preprocess import clean

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "news_classifier.joblib"


@dataclass
class Prediction:
    label: str
    label_bn: str
    confidence: float
    probabilities: dict[str, float]
    keywords: list[str]


class NewsClassifier:
    def __init__(self, path: Path = DEFAULT_MODEL):
        bundle = joblib.load(path)
        self.pipeline = bundle["model"]
        self.name = bundle["name"]
        self.metrics = bundle.get("test", {})
        self.classes = list(self.pipeline.classes_)
        self._weights, self._word_vec = self._word_weights()

    def _word_weights(self):
        """
        Find the weight each word has for each class, used to explain a prediction.
        Works for the plain linear models and for the calibrated SVM (averaging its folds).
        """
        features = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "calibrated_classifiers_"):
            coef = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
        elif hasattr(clf, "coef_"):
            coef = clf.coef_
        elif hasattr(clf, "feature_log_prob_"):
            coef = clf.feature_log_prob_
        else:
            return None, None

        if hasattr(features, "transformer_list"):  # word + char union: word features come first
            word_vec = dict(features.transformer_list)["word"]
            coef = coef[:, : len(word_vec.vocabulary_)]
        else:
            word_vec = features
        return coef, word_vec

    def keywords(self, text: str, label: str, k: int = 6) -> list[str]:
        if self._weights is None:
            return []
        row = self._word_vec.transform([text])
        idx = row.indices
        if len(idx) == 0:
            return []
        weights = self._weights[self.classes.index(label), idx] * row.data
        order = np.argsort(weights)[::-1]
        vocab = self._word_vec.get_feature_names_out()
        return [vocab[idx[i]] for i in order[:k] if weights[i] > 0]

    def predict(self, text: str) -> Prediction:
        if not clean(text):
            raise ValueError("Text has no Bangla words")
        probs = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.classes, probs), key=lambda p: p[1], reverse=True)
        label, confidence = ranked[0]
        return Prediction(
            label=label,
            label_bn=LABELS_BN.get(label, label),
            confidence=round(float(confidence), 4),
            probabilities={c: round(float(p), 4) for c, p in ranked},
            keywords=self.keywords(text, label),
        )
