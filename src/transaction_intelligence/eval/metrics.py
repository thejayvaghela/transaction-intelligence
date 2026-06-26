"""Pure metric functions for the eval harness (see docs/02-baselines.md §4.3).

Kept dependency-light and side-effect-free so they're trivially unit-testable with hand-computed
golden values. The harness in :mod:`harness` composes these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Macro-averaged F1 (equal weight per class) over the given label space."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))


def per_class_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support, in `labels` order."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"label": labels, "precision": p, "recall": r, "f1": f, "support": s}
    )


def confusion_df(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows = true, cols = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def expected_calibration_error(
    confidences, correct, n_bins: int = 10
) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin population.

    confidences: max predicted probability per example. correct: bool, was the top prediction right.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    n = len(conf)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_acc = corr[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def accuracy_at_coverage(confidences, correct, steps: int = 10) -> pd.DataFrame:
    """Accuracy when answering only the most-confident `coverage` fraction of inputs."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    n = len(conf)
    if n == 0:
        return pd.DataFrame(columns=["coverage", "accuracy", "n"])
    corr_sorted = corr[np.argsort(-conf)]
    rows = []
    for c in np.linspace(1.0 / steps, 1.0, steps):
        k = max(1, int(round(c * n)))
        rows.append({"coverage": k / n, "accuracy": float(corr_sorted[:k].mean()), "n": k})
    return pd.DataFrame(rows)
