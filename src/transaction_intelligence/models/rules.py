"""Adapt the rule-based weak labeler to the harness Predictor interface (the rules floor).

Abstentions (and category-only MCC hits) map to a sentinel that counts as wrong on the full set,
so the rules floor's full-set macro-F1 honestly reflects its 50% coverage.
"""

from __future__ import annotations

from transaction_intelligence.data.weak_label import WeakLabeler

ABSTAIN = "<abstain>"


class RulesPredictor:
    def __init__(self, labeler: WeakLabeler | None = None):
        self.labeler = labeler or WeakLabeler()

    def predict(self, descriptors) -> list[str]:
        out = []
        for d in descriptors:
            wl = self.labeler.label(d)
            out.append(wl.subtype if (wl is not None and wl.subtype is not None) else ABSTAIN)
        return out
