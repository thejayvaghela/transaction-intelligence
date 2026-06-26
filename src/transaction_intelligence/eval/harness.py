"""Model-agnostic evaluation harness (see docs/02-baselines.md §4, ADR 0005).

Any model implementing the :class:`Predictor` interface is scored identically — the rules floor,
the TF-IDF baseline, a fine-tuned transformer, or an ONNX session — so every comparison in the
project is apples-to-apples. Evaluates both the **subtype** (40-way) and derived **category**
(11-way) heads, on any dataset DataFrame with `descriptor` / `subtype` / `category` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from transaction_intelligence import taxonomy as tx

from . import metrics


@runtime_checkable
class Predictor(Protocol):
    def predict(self, descriptors: list[str]) -> list[str]:
        """Return a predicted **subtype** label per descriptor (or a sentinel to abstain)."""
        ...

    # Optional: predict_proba(descriptors) -> np.ndarray of shape (n, NUM_SUBTYPES),
    # columns aligned to tx.SUBTYPES. Enables ECE + accuracy@coverage.


@dataclass
class HeadResult:
    name: str  # "subtype" | "category"
    macro_f1: float
    accuracy: float
    per_class: pd.DataFrame
    confusion: pd.DataFrame | None = None


@dataclass
class EvalResult:
    dataset: str
    n: int
    subtype: HeadResult
    category: HeadResult
    ece: float | None = None
    coverage_curve: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        """Flat, JSON-friendly payload (logged to MLflow / checked by the CI eval gate)."""
        return {
            "dataset": self.dataset,
            "n": self.n,
            "subtype_macro_f1": self.subtype.macro_f1,
            "subtype_accuracy": self.subtype.accuracy,
            "category_macro_f1": self.category.macro_f1,
            "category_accuracy": self.category.accuracy,
            "ece": self.ece,
        }

    def summary(self) -> str:
        lines = [
            f"[{self.dataset}] n={self.n}",
            f"  subtype : macro-F1 {self.subtype.macro_f1:.3f} | acc {self.subtype.accuracy:.3f}",
            f"  category: macro-F1 {self.category.macro_f1:.3f} | acc {self.category.accuracy:.3f}",
        ]
        if self.ece is not None:
            lines.append(f"  ece     : {self.ece:.3f}")
        return "\n".join(lines)


def _head(name, y_true, y_pred, labels, confusion=False) -> HeadResult:
    return HeadResult(
        name=name,
        macro_f1=metrics.macro_f1(y_true, y_pred, labels),
        accuracy=metrics.accuracy(y_true, y_pred),
        per_class=metrics.per_class_report(y_true, y_pred, labels),
        confusion=metrics.confusion_df(y_true, y_pred, labels) if confusion else None,
    )


def _derive_category(subtype: str) -> str:
    """Map a predicted subtype to its category; pass through unknown sentinels (abstain)."""
    return tx.category_of_subtype(subtype) if subtype in tx.SUBTYPE_TO_ID else subtype


def evaluate(predictor: Predictor, df: pd.DataFrame, dataset_name: str) -> EvalResult:
    """Run the predictor over `df` and score both heads (+ calibration if proba is available)."""
    descriptors = df["descriptor"].tolist()
    y_sub_true = df["subtype"].tolist()
    y_cat_true = df["category"].tolist()

    y_sub_pred = list(predictor.predict(descriptors))
    y_cat_pred = [_derive_category(s) for s in y_sub_pred]

    subtype = _head("subtype", y_sub_true, y_sub_pred, tx.SUBTYPES)
    category = _head("category", y_cat_true, y_cat_pred, tx.CATEGORIES, confusion=True)

    ece = coverage = None
    proba_fn = getattr(predictor, "predict_proba", None)
    if callable(proba_fn):
        proba = proba_fn(descriptors)
        if proba is not None:
            proba = np.asarray(proba)
            conf = proba.max(axis=1)
            pred_idx = proba.argmax(axis=1)
            pred_lab = [tx.SUBTYPES[i] for i in pred_idx]  # proba cols aligned to tx.SUBTYPES
            correct = [p == t for p, t in zip(pred_lab, y_sub_true, strict=True)]
            ece = metrics.expected_calibration_error(conf, correct)
            coverage = metrics.accuracy_at_coverage(conf, correct)

    return EvalResult(dataset_name, len(df), subtype, category, ece, coverage)
