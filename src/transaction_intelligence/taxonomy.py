"""Label taxonomy — the single source of truth for every model in the project.

Defines the fixed schema we classify transactions into: a two-level hierarchy of
CATEGORY -> SUBTYPE. Everything downstream (data weak-labeling, baselines, the
fine-tuned teacher, the distilled student, serving) imports its labels from here so
they can never drift out of sync.

Stability contract
------------------
Label *IDs* are derived from the ORDER of entries below and get baked into trained
model weights (output neuron ``i`` means ``CATEGORIES[i]``). Therefore:

* **Append-only** — add new categories / subtypes at the end.
* **Never reorder or delete** an existing entry without retraining every model and
  bumping ``TAXONOMY_VERSION``.

See docs/00-foundations.md and docs/decisions/0003-label-taxonomy.md for the rationale.
"""

from __future__ import annotations

TAXONOMY_VERSION = "1.0.0"

# category -> ordered subtypes. Order is significant (see the stability contract above).
TAXONOMY: dict[str, list[str]] = {
    "food_drink": ["restaurant", "coffee", "fast_food", "bar", "delivery"],
    "groceries": ["supermarket", "convenience", "specialty"],
    "shopping": ["general_merch", "clothing", "electronics", "online_marketplace"],
    "transportation": ["rideshare", "fuel", "transit", "parking_tolls"],
    "travel": ["airline", "lodging", "car_rental"],
    "bills_utilities": ["electric_gas", "water", "internet_phone", "insurance"],
    "subscriptions_entertainment": ["streaming", "gaming", "events", "software"],
    "health_wellness": ["pharmacy", "medical", "fitness"],
    "financial": ["bank_fee", "atm_cash", "interest", "investment"],
    "transfers": ["p2p", "internal", "wire"],
    "income": ["payroll", "refund", "deposit"],
}

# Full subtype label = f"{category}{SUBTYPE_SEP}{subtype}", e.g. "food_drink/coffee".
SUBTYPE_SEP = "/"

# --- Categories -----------------------------------------------------------------
CATEGORIES: list[str] = list(TAXONOMY.keys())
NUM_CATEGORIES: int = len(CATEGORIES)
CATEGORY_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
ID_TO_CATEGORY: dict[int, str] = {i: c for c, i in CATEGORY_TO_ID.items()}

# --- Subtypes (namespaced under their category to avoid collisions) -------------
SUBTYPES: list[str] = [
    f"{category}{SUBTYPE_SEP}{sub}" for category, subs in TAXONOMY.items() for sub in subs
]
NUM_SUBTYPES: int = len(SUBTYPES)
SUBTYPE_TO_ID: dict[str, int] = {s: i for i, s in enumerate(SUBTYPES)}
ID_TO_SUBTYPE: dict[int, str] = {i: s for s, i in SUBTYPE_TO_ID.items()}


def make_subtype(category: str, subtype: str) -> str:
    """('food_drink', 'coffee') -> 'food_drink/coffee'."""
    return f"{category}{SUBTYPE_SEP}{subtype}"


def category_of_subtype(subtype_label: str) -> str:
    """'food_drink/coffee' -> 'food_drink'."""
    return subtype_label.split(SUBTYPE_SEP, 1)[0]


def subtypes_of(category: str) -> list[str]:
    """Namespaced subtype labels for a category (e.g. 'groceries/supermarket', ...)."""
    return [make_subtype(category, s) for s in TAXONOMY[category]]


def validate() -> None:
    """Assert the taxonomy is well-formed. Raises AssertionError on any problem."""
    assert CATEGORIES, "taxonomy has no categories"
    assert len(CATEGORIES) == len(set(CATEGORIES)), "duplicate category names"
    for category, subs in TAXONOMY.items():
        assert subs, f"category {category!r} has no subtypes"
        assert len(subs) == len(set(subs)), f"duplicate subtypes within {category!r}"
        assert SUBTYPE_SEP not in category, f"category {category!r} contains {SUBTYPE_SEP!r}"
        for s in subs:
            assert SUBTYPE_SEP not in s, f"subtype {s!r} must not contain {SUBTYPE_SEP!r}"
    assert len(SUBTYPES) == len(set(SUBTYPES)), "duplicate namespaced subtype labels"


# Fail fast at import time if the taxonomy is ever edited into a bad state.
validate()
