"""TF-IDF + Logistic Regression baseline (see docs/02-baselines.md §5).

Word + char n-gram TF-IDF into a multinomial logistic regression, predicting the **subtype**.
Char n-grams (`char_wb`) are the key choice: they survive truncation/concatenation (`WHOLEFDS`,
`NTFLX`) that breaks whole-word features. Wrapped to the harness :class:`Predictor` interface, with
``predict_proba`` reindexed to ``tx.SUBTYPES`` order (sklearn sorts ``classes_`` alphabetically).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from transaction_intelligence import taxonomy as tx

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "baseline.joblib"


def build_pipeline(min_df: int = 2) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class BaselinePredictor:
    """Fitted pipeline adapted to the harness Predictor interface (subtype-level)."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._classes = list(pipeline.named_steps["clf"].classes_)

    def predict(self, descriptors) -> list[str]:
        return list(self.pipeline.predict(list(descriptors)))

    def predict_proba(self, descriptors) -> np.ndarray:
        raw = self.pipeline.predict_proba(list(descriptors))  # cols in classes_ (alpha) order
        aligned = np.zeros((raw.shape[0], tx.NUM_SUBTYPES))
        for j, cls in enumerate(self._classes):
            aligned[:, tx.SUBTYPE_TO_ID[cls]] = raw[:, j]  # reindex to taxonomy order
        return aligned

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> BaselinePredictor:
        return cls(joblib.load(path))


def train_baseline(train_df: pd.DataFrame, min_df: int = 2) -> BaselinePredictor:
    """Fit the baseline pipeline on a dataframe with `descriptor` + `subtype` columns."""
    pipe = build_pipeline(min_df=min_df)
    pipe.fit(train_df["descriptor"].tolist(), train_df["subtype"].tolist())
    return BaselinePredictor(pipe)
