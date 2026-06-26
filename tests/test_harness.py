"""Tests for the model-agnostic eval harness."""

import numpy as np
import pandas as pd

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.eval import Predictor, evaluate

# A tiny dataset spanning two categories.
DF = pd.DataFrame(
    {
        "descriptor": ["SBUX", "MCD", "WALMART", "AMZN"],
        "subtype": [
            "food_drink/coffee",
            "food_drink/fast_food",
            "shopping/general_merch",
            "shopping/online_marketplace",
        ],
        "category": ["food_drink", "food_drink", "shopping", "shopping"],
    }
)


class PerfectPredictor:
    """Predicts every subtype correctly (no proba)."""

    def __init__(self, df):
        self._map = dict(zip(df["descriptor"], df["subtype"], strict=True))

    def predict(self, descriptors):
        return [self._map[d] for d in descriptors]


class ProbaPredictor(PerfectPredictor):
    """Perfect predictions plus aligned, confident probabilities."""

    def predict_proba(self, descriptors):
        proba = np.zeros((len(descriptors), tx.NUM_SUBTYPES))
        for i, d in enumerate(descriptors):
            proba[i, tx.SUBTYPE_TO_ID[self._map[d]]] = 1.0
        return proba


def test_perfect_predictor_is_fully_accurate():
    # macro-F1 is over the FULL 40-subtype space, so a 4-row toy scores low on macro-F1;
    # accuracy (row-wise) is the right check here.
    res = evaluate(PerfectPredictor(DF), DF, "toy")
    assert res.subtype.accuracy == 1.0
    assert res.category.accuracy == 1.0
    assert res.ece is None  # no proba -> no calibration


def test_category_is_derived_from_subtype():
    # A predictor that confuses subtype WITHIN a category keeps category correct.
    class WithinCategoryConfuser(PerfectPredictor):
        def predict(self, descriptors):
            out = super().predict(descriptors)
            return ["food_drink/restaurant" if s.startswith("food_drink") else s for s in out]

    res = evaluate(WithinCategoryConfuser(DF), DF, "toy")
    assert res.category.accuracy == 1.0  # category survives subtype confusion
    assert res.subtype.accuracy < 1.0


def test_proba_path_populates_calibration():
    res = evaluate(ProbaPredictor(DF), DF, "toy")
    assert res.ece == 0.0  # perfectly confident + correct
    assert res.coverage_curve is not None
    assert res.coverage_curve["accuracy"].iloc[-1] == 1.0


def test_runtime_protocol():
    assert isinstance(PerfectPredictor(DF), Predictor)
