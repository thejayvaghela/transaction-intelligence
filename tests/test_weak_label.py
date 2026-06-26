"""Tests for the rule-based weak labeler."""

from transaction_intelligence.data.weak_label import WeakLabeler

LABELER = WeakLabeler()


def test_known_merchant_gets_subtype():
    wl = LABELER.label("SQ *STARBUCKS STORE #123 SEATTLE WA")
    assert wl is not None
    assert wl.subtype == "food_drink/coffee"
    assert wl.method == "gazetteer"


def test_unseen_merchant_abstains_without_mcc():
    assert LABELER.label("PANERA BREAD #2231 AUSTIN TX") is None


def test_mcc_fallback_gives_category_only():
    wl = LABELER.label("UNKNOWN MERCHANT XYZ", mcc=5812)
    assert wl is not None
    assert wl.category == "food_drink"
    assert wl.subtype is None
    assert wl.method == "mcc"


def test_total_miss_abstains():
    assert LABELER.label("ZZZ QQQ NOTHING HERE") is None
