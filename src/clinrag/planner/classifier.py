"""Predict which sources a question needs from its text alone.

One logistic regression per source (relational, notes, graph) over
[text features | rule features]:

- text features: TF-IDF of word 1-2 grams (default), or BAAI/bge-base-en-v1.5
  sentence embeddings (featurizer="bge"; sentence-transformers is imported
  only when that featurizer is used, so tests never load torch).
- rule features: the binary keyword detectors below, shared with RulePlanner.

Before TF-IDF, admission ids become the token "admid" and other numbers become
"num", so the model learns the shape of a question rather than specific ids or
thresholds.

The classifier only uses question.text. It never reads required_sources at
prediction time.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec
from clinrag.workload.icd_categories import CATEGORIES
from clinrag.workload.labs import LABS

SOURCES: tuple[str, ...] = (RELATIONAL, NOTES, GRAPH)
FEATURIZERS: frozenset[str] = frozenset({"tfidf", "bge"})
BGE_MODEL = "BAAI/bge-base-en-v1.5"

# ---- rule features ----------------------------------------------------------------

_ADMISSION_ID = re.compile(r"(?<!\d)\d{7,9}(?!\d)")  # MIMIC-IV hadm_ids have 8 digits
_NUMBER = re.compile(r"(?<![\d.])\d{1,6}(?:\.\d+)?(?![\d.])")
_COMPARISON = re.compile(
    r"[<>]|\b(?:above|below|greater|less|more than|fewer than|exceed\w*|over|under"
    r"|highest|lowest|maximum|minimum|max|min)\b",
    re.IGNORECASE,
)
_NOTE_WORDS = re.compile(r"discharge summar|\bmention|\bdocumented\b|\bsummary\b", re.IGNORECASE)


def _word_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    alternatives = sorted({re.escape(t.lower()) for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


_LAB_NAMES = _word_pattern(list(LABS))
_CATEGORY_NAMES = _word_pattern(
    [c.label for c in CATEGORIES.values()] + [name.replace("_", " ") for name in CATEGORIES]
)


def mentions_lab(text: str) -> bool:
    """A known lab name (clinrag.workload.labs.LABS), e.g. "creatinine"."""
    return bool(_LAB_NAMES.search(text))


def mentions_number(text: str) -> bool:
    """A number that is not an admission id, e.g. a threshold like 2.0."""
    return bool(_NUMBER.search(_ADMISSION_ID.sub(" ", text)))


def mentions_comparison(text: str) -> bool:
    """A comparison: <, >, above, below, highest, maximum, ..."""
    return bool(_COMPARISON.search(text))


def mentions_notes(text: str) -> bool:
    """"discharge summary", "summary", "mention" or "documented"."""
    return bool(_NOTE_WORDS.search(text))


def mentions_category(text: str) -> bool:
    """A known ICD category name or label (clinrag.workload.icd_categories.CATEGORIES)."""
    return bool(_CATEGORY_NAMES.search(text))


def mentions_admission_id(text: str) -> bool:
    """A specific admission id (7-9 digit number)."""
    return bool(_ADMISSION_ID.search(text))


RULES: tuple[tuple[str, Any], ...] = (
    ("lab_name", mentions_lab),
    ("number_or_comparison", lambda t: mentions_number(t) or mentions_comparison(t)),
    ("note_words", mentions_notes),
    ("icd_category", mentions_category),
    ("admission_id", mentions_admission_id),
)


def rule_features(text: str) -> np.ndarray:
    """Binary rule feature vector, in RULES order."""
    return np.array([float(rule(text)) for _, rule in RULES])


def normalize_text(text: str) -> str:
    """Replace admission ids with "admid" and other numbers with "num"."""
    return _NUMBER.sub(" num ", _ADMISSION_ID.sub(" admid ", text))


# ---- classifier -------------------------------------------------------------------


def _text(question: QuestionSpec | str) -> str:
    return question if isinstance(question, str) else question.text


class SourceClassifier:
    """Per-source logistic regressions over text + rule features.

    featurizer: "tfidf" (default) or "bge".
    C:          inverse regularization strength of each logistic regression.
    last_latency_ms: wall time of the most recent predict_proba/predict call,
                featurization included; routers add it to planning overhead.

    A source whose training label never varies gets a constant probability
    (1.0 or 0.0) instead of a regression.
    """

    def __init__(self, featurizer: str = "tfidf", C: float = 10.0, max_iter: int = 1000) -> None:
        if featurizer not in FEATURIZERS:
            raise ValueError(f"featurizer must be one of {sorted(FEATURIZERS)}, got {featurizer!r}")
        self.featurizer = featurizer
        self.C = C
        self.max_iter = max_iter
        self.vectorizer: TfidfVectorizer | None = None
        self.models: dict[str, LogisticRegression] = {}
        self.constant: dict[str, float] = {}
        self.last_latency_ms: float = 0.0
        self._encoder: Any = None

    # -- features --

    def _encoder_model(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # lazy: loads torch

            self._encoder = SentenceTransformer(BGE_MODEL)
        return self._encoder

    def _features(self, texts: Sequence[str], fit: bool = False) -> sparse.csr_matrix:
        rules = sparse.csr_matrix(np.vstack([rule_features(t) for t in texts]))
        if self.featurizer == "tfidf":
            normalized = [normalize_text(t) for t in texts]
            if fit:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
                text_x = self.vectorizer.fit_transform(normalized)
            else:
                if self.vectorizer is None:
                    raise RuntimeError("SourceClassifier is not fitted")
                text_x = self.vectorizer.transform(normalized)
        else:
            emb = self._encoder_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            text_x = sparse.csr_matrix(np.asarray(emb))
        return sparse.hstack([text_x, rules], format="csr")

    # -- training --

    def fit(self, questions: Sequence[QuestionSpec]) -> SourceClassifier:
        """Train on labelled questions (the dev split); labels come from required_sources."""
        if not questions:
            raise ValueError("fit needs at least one question")
        x = self._features([q.text for q in questions], fit=True)
        self.models, self.constant = {}, {}
        for source in SOURCES:
            y = np.array([int(source in q.required_sources) for q in questions])
            if y.min() == y.max():
                self.constant[source] = float(y[0])
                continue
            model = LogisticRegression(C=self.C, max_iter=self.max_iter)
            model.fit(x, y)
            self.models[source] = model
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self.models or self.constant)

    # -- prediction --

    def predict_proba_batch(self, questions: Sequence[QuestionSpec | str]) -> list[dict[str, float]]:
        """predict_proba for many questions at once (featurized together). Sets last_latency_ms."""
        if not self.is_fitted:
            raise RuntimeError("SourceClassifier is not fitted")
        start = time.perf_counter()
        x = self._features([_text(q) for q in questions])
        columns: dict[str, np.ndarray] = {}
        for source in SOURCES:
            if source in self.constant:
                columns[source] = np.full(x.shape[0], self.constant[source])
            else:
                columns[source] = self.models[source].predict_proba(x)[:, 1]
        out = [{s: float(columns[s][i]) for s in SOURCES} for i in range(x.shape[0])]
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return out

    def predict_proba(self, question: QuestionSpec | str) -> dict[str, float]:
        """Probability that the question needs each source. Sets last_latency_ms."""
        return self.predict_proba_batch([question])[0]

    def predict(self, question: QuestionSpec | str, tau: float = 0.5) -> set[str]:
        """Sources with probability >= tau. May be empty. Sets last_latency_ms."""
        return threshold(self.predict_proba(question), tau)

    # -- persistence --

    def save(self, path: str | Path) -> Path:
        """Write with joblib (the bge encoder is not saved; it reloads on first use)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SourceClassifier:
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not hold a SourceClassifier")
        return obj

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state


def threshold(probs: dict[str, float], tau: float) -> set[str]:
    """Sources whose probability is at least tau."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    return {s for s, p in probs.items() if p >= tau}
