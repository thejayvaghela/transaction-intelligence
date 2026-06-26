"""Leakage-safe, group-aware, stratified splitting (see docs/01-data-pipeline.md §4.5).

The unit we assign to a split is the *merchant*, not the row. Because every row of a merchant
goes to the same split, the model can never see a merchant in training and again in test — so a
high score reflects generalization to *unseen* merchants, not memorization. We stratify by subtype
so each split contains every subtype proportionally.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .gazetteer import MerchantEntry


def split_merchants(
    entries: list[MerchantEntry], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign each merchant_id to exactly one of train/val/test.

    Returns ``{merchant_id: split}``. Stratified by subtype, grouped by merchant, deterministic.
    """
    rng = random.Random(seed)
    by_subtype: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_subtype[e.subtype].append(e.merchant_id)

    assignment: dict[str, str] = {}
    for subtype in sorted(by_subtype):
        mids = sorted(by_subtype[subtype])  # sort first for determinism, then shuffle
        rng.shuffle(mids)
        n = len(mids)

        n_test = max(1, round(n * ratios["test"]))
        n_val = max(1, round(n * ratios["val"]))
        if n_test + n_val >= n:  # guarantee a non-empty train split for tiny subtypes
            n_test = 1
            n_val = 1 if n >= 3 else 0
        n_train = n - n_val - n_test

        for i, mid in enumerate(mids):
            if i < n_train:
                assignment[mid] = "train"
            elif i < n_train + n_val:
                assignment[mid] = "val"
            else:
                assignment[mid] = "test"
    return assignment
