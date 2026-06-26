"""Weak labeling — cheap, imperfect labels from rules (gazetteer match + MCC map).

Synthetic-first means we don't need this to label the *training* set; it teaches weak supervision,
gives a "rules-only" floor, and is the tool for labeling/triaging real data (the reality-check
eval). It is deliberately brittle on noise — that brittleness motivates the learned model.
See docs/01-data-pipeline.md §4.3.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .gazetteer import MerchantEntry, load_gazetteer


@dataclass(frozen=True)
class WeakLabel:
    category: str
    subtype: str | None  # None when only the MCC fallback fired (category-only)
    confidence: float
    method: str  # "gazetteer" | "mcc"


def build_surface_index(entries: list[MerchantEntry]) -> list[tuple[re.Pattern, str, str]]:
    """Word-boundary patterns for each surface form, longest first (specific matches win)."""
    items: list[tuple[str, str, str]] = []
    for e in entries:
        for sf in e.surface_forms:
            sf_u = sf.upper()
            if len(sf_u) >= 2:
                items.append((sf_u, e.subtype, e.category))
    items.sort(key=lambda t: -len(t[0]))
    return [(re.compile(rf"\b{re.escape(sf)}\b"), sub, cat) for sf, sub, cat in items]


def build_mcc_category_map(entries: list[MerchantEntry]) -> dict[int, str]:
    """MCC -> most common category (MCC is too coarse to pin a subtype reliably)."""
    by_mcc: dict[int, Counter] = defaultdict(Counter)
    for e in entries:
        if e.mcc is not None:
            by_mcc[e.mcc][e.category] += 1
    return {mcc: counts.most_common(1)[0][0] for mcc, counts in by_mcc.items()}


class WeakLabeler:
    """Rule-based weak labeler built from the gazetteer."""

    def __init__(self, entries: list[MerchantEntry] | None = None):
        entries = entries if entries is not None else load_gazetteer()
        self.surface_index = build_surface_index(entries)
        self.mcc_map = build_mcc_category_map(entries)

    def label(self, descriptor: str, mcc: int | None = None) -> WeakLabel | None:
        """Return a WeakLabel, or None to abstain."""
        d = descriptor.upper()
        for pattern, subtype, category in self.surface_index:
            if pattern.search(d):
                return WeakLabel(category, subtype, 0.9, "gazetteer")
        if mcc is not None and mcc in self.mcc_map:
            return WeakLabel(self.mcc_map[mcc], None, 0.5, "mcc")
        return None
