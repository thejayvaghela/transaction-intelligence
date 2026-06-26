"""Tests for the merchant gazetteer loader and the curated gazetteer data."""

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data import gazetteer as gz

ENTRIES = gz.load_gazetteer()


def test_loads_and_ids_unique():
    assert len(ENTRIES) > 100
    ids = [e.merchant_id for e in ENTRIES]
    assert len(ids) == len(set(ids))


def test_every_subtype_covered():
    cov = gz.coverage(ENTRIES)
    assert set(cov) == set(tx.SUBTYPES)
    uncovered = [s for s, c in cov.items() if c == 0]
    assert not uncovered, f"uncovered subtypes: {uncovered}"


def test_min_merchants_per_subtype():
    gz.validate_gazetteer(ENTRIES, min_per_subtype=4)


def test_entries_well_formed():
    for e in ENTRIES:
        assert e.subtype in tx.SUBTYPE_TO_ID
        assert e.category == tx.category_of_subtype(e.subtype)
        assert e.channel in gz.VALID_CHANNELS
        assert e.canonical_name
        assert e.surface_forms[0] == e.canonical_name
        assert e.mcc is None or isinstance(e.mcc, int)
