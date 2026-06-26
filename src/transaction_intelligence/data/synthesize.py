"""Descriptor synthesizer — the noise model that turns gazetteer entries into realistic rows.

A pipeline of probabilistic, channel-conditioned transforms (see docs/01-data-pipeline.md §4.2):

    pick surface form -> normalize -> maybe abbreviate -> maybe prefix -> maybe suffix -> truncate

Determinism: every random choice draws from an injected ``random.Random`` so generation is
byte-reproducible given a seed. ``noise_level`` scales the *added-noise* probabilities (prefixes,
suffixes, truncation) — not structural choices like which surface form is picked.
"""

from __future__ import annotations

import math
import random
import re

from transaction_intelligence import taxonomy as tx

from .gazetteer import MerchantEntry

ROW_FIELDS = (
    "descriptor",
    "amount",
    "is_debit",
    "category",
    "subtype",
    "merchant_id",
    "source",
    "noise_level",
)

# Non-empty processor/transaction prefixes by channel (whether one is applied is probabilistic).
PREFIX_POOL = {
    "card_present": ["SQ *", "TST* ", "POS ", "CKE*", "IN *"],
    "online": ["PYPL *", "PP*", "SQ *", "SP *", "WWW."],
    "recurring": ["RECUR*", "AUTOPAY "],
    "financial": ["ACH ", "POS DEBIT ", "ELECTRONIC "],
}
PREFIX_PROB = {"card_present": 0.35, "online": 0.50, "recurring": 0.25, "financial": 0.40}

US_CITIES = [
    ("SAN FRANCISCO", "CA"), ("LOS ANGELES", "CA"), ("NEW YORK", "NY"), ("CHICAGO", "IL"),
    ("HOUSTON", "TX"), ("SEATTLE", "WA"), ("AUSTIN", "TX"), ("BOSTON", "MA"),
    ("DENVER", "CO"), ("MIAMI", "FL"), ("ATLANTA", "GA"), ("PORTLAND", "OR"),
    ("PHOENIX", "AZ"), ("DALLAS", "TX"), ("SAN DIEGO", "CA"), ("PHILADELPHIA", "PA"),
    ("MINNEAPOLIS", "MN"), ("NASHVILLE", "TN"), ("BROOKLYN", "NY"), ("SAN JOSE", "CA"),
]
STATES = ["CA", "NY", "TX", "FL", "WA", "IL", "MA", "CO", "OR", "GA", "AZ", "PA", "TN", "NJ"]

# Amount magnitude range ($low, $high), sampled log-uniform. Per-category, with subtype overrides.
CATEGORY_AMOUNT = {
    "food_drink": (4, 90),
    "groceries": (8, 180),
    "shopping": (10, 400),
    "transportation": (3, 90),
    "travel": (60, 900),
    "bills_utilities": (30, 350),
    "subscriptions_entertainment": (5, 60),
    "health_wellness": (10, 300),
    "financial": (2, 50),
    "transfers": (10, 1200),
    "income": (200, 5000),
}
SUBTYPE_AMOUNT = {
    "food_drink/coffee": (3, 12),
    "transportation/fuel": (20, 90),
    "travel/airline": (80, 900),
    "financial/atm_cash": (20, 400),
    "financial/investment": (50, 3000),
    "transfers/wire": (200, 5000),
}


def _phone(rng: random.Random) -> str:
    a, b, c = rng.randint(200, 999), rng.randint(200, 999), rng.randint(1000, 9999)
    return rng.choice([f"{a}{b}{c}", f"{a}-{b}-{c}"])


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"


def _store_id(rng: random.Random) -> str:
    return rng.choice([f"#{rng.randint(1000, 9999)}", f"STORE {rng.randint(10000, 99999)}"])


def _ref(rng: random.Random) -> str:
    n = rng.randint(100000, 999999)
    return rng.choice([f"REF#{n}", f"CONF {n}", _date(rng)])


def _pick_surface_form(entry: MerchantEntry, rng: random.Random) -> str:
    forms = entry.surface_forms
    weights = [2.0] + [1.0] * (len(forms) - 1)  # bias toward the canonical name
    return rng.choices(forms, weights=weights, k=1)[0]


def _normalize(text: str, rng: random.Random) -> str:
    text = text.upper()
    if rng.random() < 0.6:
        text = text.replace("'", "")
    if "&" in text:
        text = text.replace("&", rng.choice([" AND ", "", "&"]))
    return text


def _maybe_abbreviate(text: str, rng: random.Random, p) -> str:
    if rng.random() < p(0.15):
        return text.replace(" ", "")
    if rng.random() < p(0.10):
        toks = []
        for t in text.split():
            if len(t) > 5:
                t = t[0] + re.sub(r"[AEIOU]", "", t[1:])
            toks.append(t)
        text = " ".join(toks)
    return text


def _maybe_prefix(text: str, channel: str, rng: random.Random, p) -> str:
    if rng.random() < p(PREFIX_PROB[channel]):
        return rng.choice(PREFIX_POOL[channel]) + text
    return text


def _maybe_suffix(text: str, channel: str, rng: random.Random, p) -> str:
    parts = [text]
    if channel == "card_present":
        if rng.random() < p(0.6):
            city, st = rng.choice(US_CITIES)
            parts.append(f"{city} {st}")
        elif rng.random() < p(0.3):
            parts.append(_store_id(rng))
        if rng.random() < p(0.15):
            parts.append(_phone(rng))
    elif channel == "online":
        if rng.random() < p(0.4):
            parts.append(_phone(rng))
        if rng.random() < p(0.2):
            parts.append(rng.choice([".COM", rng.choice(STATES)]))
    elif channel == "recurring":
        if rng.random() < p(0.35):
            parts.append(rng.choice([_date(rng), "AUTOPAY", f"ID{rng.randint(100000, 999999)}"]))
    elif channel == "financial":
        if rng.random() < p(0.5):
            parts.append(_ref(rng))
    return " ".join(parts)


def _maybe_truncate(text: str, rng: random.Random, p) -> str:
    if rng.random() < p(0.25):
        width = rng.choice([20, 22, 25, 30])
        text = text[:width].rstrip()
    return text


def synthesize_descriptor(
    entry: MerchantEntry, rng: random.Random, noise_level: float = 1.0
) -> str:
    """Generate one noisy descriptor string for a merchant."""

    def p(base: float) -> float:
        return min(1.0, base * noise_level)

    text = _pick_surface_form(entry, rng)
    text = _normalize(text, rng)
    text = _maybe_abbreviate(text, rng, p)
    text = _maybe_prefix(text, entry.channel, rng, p)
    text = _maybe_suffix(text, entry.channel, rng, p)
    text = _maybe_truncate(text, rng, p)
    text = re.sub(r"\s+", " ", text).strip()
    return text or entry.canonical_name.upper()


def _is_debit(entry: MerchantEntry, rng: random.Random) -> bool:
    if entry.category == "income":
        return False
    if entry.category == "transfers" or entry.subtype == "financial/interest":
        return rng.random() < 0.5
    return True


def sample_amount(entry: MerchantEntry, rng: random.Random) -> tuple[float, bool]:
    """Return (signed amount, is_debit). Negative = outflow/debit; log-uniform magnitude."""
    low, high = SUBTYPE_AMOUNT.get(entry.subtype) or CATEGORY_AMOUNT[entry.category]
    mag = round(math.exp(rng.uniform(math.log(low), math.log(high))), 2)
    debit = _is_debit(entry, rng)
    return (-mag if debit else mag), debit


def synthesize_row(entry: MerchantEntry, rng: random.Random, noise_level: float = 1.0) -> dict:
    """Generate one full dataset row (descriptor + features + labels + provenance)."""
    descriptor = synthesize_descriptor(entry, rng, noise_level)
    amount, is_debit = sample_amount(entry, rng)
    return {
        "descriptor": descriptor,
        "amount": amount,
        "is_debit": is_debit,
        "category": entry.category,
        "subtype": entry.subtype,
        "merchant_id": entry.merchant_id,
        "source": "synthetic",
        "noise_level": noise_level,
    }


def generate_rows(
    entries: list[MerchantEntry],
    rows_per_merchant: int,
    rng: random.Random,
    noise_level: float = 1.0,
) -> list[dict]:
    """Generate rows_per_merchant rows for each entry, in deterministic order."""
    rows: list[dict] = []
    for entry in entries:
        for _ in range(rows_per_merchant):
            rows.append(synthesize_row(entry, rng, noise_level))
    return rows


# Sanity: ensure ROW_FIELDS matches what synthesize_row actually emits.
assert set(ROW_FIELDS) == set(
    synthesize_row(
        MerchantEntry("x", "X", (), tx.CATEGORIES[0], tx.SUBTYPES[0], None, "card_present"),
        random.Random(0),
    )
)
