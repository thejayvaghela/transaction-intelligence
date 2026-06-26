"""Tests for the hand-verified gold eval set — the realistic, never-trained-on yardstick."""

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data.datasets import load_gold

GOLD = load_gold()
EXPECTED_COLS = {
    "descriptor", "category", "subtype", "amount", "is_debit", "merchant_unseen", "notes",
}


def test_size_and_columns():
    assert len(GOLD) >= 120
    assert EXPECTED_COLS.issubset(GOLD.columns)


def test_labels_valid_and_consistent():
    for row in GOLD.itertuples():
        assert row.category in tx.CATEGORY_TO_ID
        assert row.subtype in tx.SUBTYPE_TO_ID
        assert tx.category_of_subtype(row.subtype) == row.category


def test_all_subtypes_present():
    assert set(GOLD["subtype"]) == set(tx.SUBTYPES)


def test_meaningful_unseen_fraction():
    # The gold set must test transfer to UNSEEN merchants, not just memorization.
    assert GOLD["merchant_unseen"].mean() >= 0.3


def test_bool_columns_parsed():
    assert GOLD["is_debit"].isin([True, False]).all()
    assert GOLD["merchant_unseen"].isin([True, False]).all()
    assert GOLD["descriptor"].str.len().min() > 0
