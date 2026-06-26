"""Tests for the label taxonomy — the single source of truth for all models."""

from transaction_intelligence import taxonomy as tx


def test_expected_shape():
    assert tx.NUM_CATEGORIES == 11
    assert tx.NUM_SUBTYPES == len(tx.SUBTYPES) == sum(len(v) for v in tx.TAXONOMY.values())


def test_category_ids_contiguous_and_roundtrip():
    assert sorted(tx.ID_TO_CATEGORY) == list(range(tx.NUM_CATEGORIES))
    for i, c in enumerate(tx.CATEGORIES):
        assert tx.CATEGORY_TO_ID[c] == i
        assert tx.ID_TO_CATEGORY[i] == c


def test_subtype_ids_contiguous_and_roundtrip():
    assert sorted(tx.ID_TO_SUBTYPE) == list(range(tx.NUM_SUBTYPES))
    for i, s in enumerate(tx.SUBTYPES):
        assert tx.SUBTYPE_TO_ID[s] == i
        assert tx.ID_TO_SUBTYPE[i] == s


def test_subtypes_are_namespaced_and_unique():
    assert len(set(tx.SUBTYPES)) == tx.NUM_SUBTYPES
    for label in tx.SUBTYPES:
        assert tx.SUBTYPE_SEP in label
        assert tx.category_of_subtype(label) in tx.CATEGORY_TO_ID


def test_subtypes_of_matches_taxonomy():
    for category, subs in tx.TAXONOMY.items():
        assert tx.subtypes_of(category) == [f"{category}/{s}" for s in subs]


def test_validate_passes():
    tx.validate()  # should not raise
