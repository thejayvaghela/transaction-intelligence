"""Tests for the descriptor synthesizer (the noise model)."""

import random
import statistics

from transaction_intelligence.data import gazetteer as gz
from transaction_intelligence.data import synthesize as syn

ENTRIES = gz.load_gazetteer()


def _rows(seed, noise=1.0, n=5, subset=20):
    rng = random.Random(seed)
    return syn.generate_rows(ENTRIES[:subset], n, rng, noise)


def test_determinism_same_seed():
    assert _rows(123) == _rows(123)


def test_different_seeds_differ():
    assert _rows(1) != _rows(2)


def test_row_schema_and_cleanliness():
    for r in _rows(7, n=2):
        assert set(r) == set(syn.ROW_FIELDS)
        assert r["descriptor"]
        assert r["descriptor"] == r["descriptor"].strip()
        assert "  " not in r["descriptor"]  # no double spaces
        assert isinstance(r["amount"], float)
        assert r["source"] == "synthetic"


def test_amount_signs_match_direction():
    rng = random.Random(0)
    income = next(e for e in ENTRIES if e.category == "income")
    food = next(e for e in ENTRIES if e.category == "food_drink")
    for _ in range(25):
        r = syn.synthesize_row(income, rng, 1.0)
        assert r["amount"] > 0 and r["is_debit"] is False
        r = syn.synthesize_row(food, rng, 1.0)
        assert r["amount"] < 0 and r["is_debit"] is True


def test_noise_level_adds_content():
    def mean_len(noise):
        rng = random.Random(0)
        rows = syn.generate_rows(ENTRIES, 3, rng, noise)
        return statistics.mean(len(r["descriptor"]) for r in rows)

    assert mean_len(1.0) > mean_len(0.0)
