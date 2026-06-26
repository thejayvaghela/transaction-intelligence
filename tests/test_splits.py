"""Tests for leakage-safe splitting — the correctness keystone of Phase 1."""

from collections import Counter, defaultdict

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data import gazetteer as gz
from transaction_intelligence.data.splits import split_merchants

ENTRIES = gz.load_gazetteer()
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _assign(seed=0):
    return split_merchants(ENTRIES, RATIOS, seed)


def test_every_merchant_assigned_once():
    assert set(_assign()) == {e.merchant_id for e in ENTRIES}


def test_no_merchant_leaks_across_splits():
    groups = defaultdict(set)
    for mid, sp in _assign().items():
        groups[sp].add(mid)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_every_subtype_in_every_split():
    sub_of = {e.merchant_id: e.subtype for e in ENTRIES}
    present = defaultdict(set)
    for mid, sp in _assign().items():
        present[sp].add(sub_of[mid])
    for sp in ("train", "val", "test"):
        assert present[sp] == set(tx.SUBTYPES), f"{sp} missing subtypes"


def test_determinism():
    assert _assign(1) == _assign(1)
    assert _assign(1) != _assign(2)


def test_ratios_reasonable():
    counts = Counter(_assign().values())
    n = sum(counts.values())
    assert 0.6 <= counts["train"] / n <= 0.75
