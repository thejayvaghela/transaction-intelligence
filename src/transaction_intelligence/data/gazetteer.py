"""Merchant gazetteer — curated ground truth that seeds the descriptor synthesizer.

Loads ``data/gazetteer/gazetteer.yaml`` (per-subtype defaults + merchant lists) into a flat
list of :class:`MerchantEntry`. Each entry carries its surface forms (canonical name + aliases),
its taxonomy labels, a typical MCC (or ``None`` for bank-feed items), and a ``channel`` the
synthesizer uses to choose realistic descriptor formatting.

See docs/01-data-pipeline.md (§4.1) and :mod:`transaction_intelligence.taxonomy`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from transaction_intelligence import taxonomy as tx

VALID_CHANNELS = {"card_present", "online", "recurring", "financial"}

# Repo-root-relative default path (src/transaction_intelligence/data/gazetteer.py -> repo root).
DEFAULT_GAZETTEER_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "gazetteer" / "gazetteer.yaml"
)


@dataclass(frozen=True)
class MerchantEntry:
    """One merchant: its labels, surface forms, MCC, and descriptor channel."""

    merchant_id: str  # stable slug; the GROUP KEY for leakage-safe splits (not a model feature)
    canonical_name: str
    aliases: tuple[str, ...]
    category: str
    subtype: str  # namespaced, e.g. "food_drink/coffee"
    mcc: int | None
    channel: str

    @property
    def surface_forms(self) -> list[str]:
        """All textual forms this merchant can appear as (canonical first)."""
        return [self.canonical_name, *self.aliases]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "merchant"


def load_gazetteer(path: str | Path | None = None) -> list[MerchantEntry]:
    """Parse the gazetteer YAML into a flat, validated list of MerchantEntry."""
    path = Path(path) if path is not None else DEFAULT_GAZETTEER_PATH
    raw = yaml.safe_load(path.read_text())

    entries: list[MerchantEntry] = []
    seen_ids: set[str] = set()
    for subtype, block in raw.items():
        if subtype not in tx.SUBTYPE_TO_ID:
            raise ValueError(f"gazetteer subtype {subtype!r} is not in the taxonomy")
        category = tx.category_of_subtype(subtype)
        default_mcc = block.get("mcc")
        default_channel = block.get("channel", "card_present")

        for item in block["merchants"]:
            if isinstance(item, str):
                name, aliases = item, []
                mcc, channel = default_mcc, default_channel
            else:
                name = item["name"]
                aliases = list(item.get("aliases", []))
                mcc = item.get("mcc", default_mcc)
                channel = item.get("channel", default_channel)

            if channel not in VALID_CHANNELS:
                raise ValueError(f"merchant {name!r}: invalid channel {channel!r}")

            # Deterministic unique merchant_id (slug, with numeric suffix on collision).
            mid = base = _slugify(name)
            n = 2
            while mid in seen_ids:
                mid = f"{base}_{n}"
                n += 1
            seen_ids.add(mid)

            entries.append(
                MerchantEntry(
                    merchant_id=mid,
                    canonical_name=name,
                    aliases=tuple(aliases),
                    category=category,
                    subtype=subtype,
                    mcc=mcc,
                    channel=channel,
                )
            )
    return entries


def coverage(entries: list[MerchantEntry]) -> dict[str, int]:
    """Map every taxonomy subtype -> number of merchants (0 if uncovered)."""
    counts = dict.fromkeys(tx.SUBTYPES, 0)
    for e in entries:
        counts[e.subtype] += 1
    return counts


def validate_gazetteer(entries: list[MerchantEntry], min_per_subtype: int = 1) -> None:
    """Assert every subtype is covered (>= min_per_subtype) and merchant_ids are unique."""
    cov = coverage(entries)
    thin = {s: c for s, c in cov.items() if c < min_per_subtype}
    if thin:
        raise AssertionError(f"subtypes below {min_per_subtype} merchant(s): {thin}")
    ids = [e.merchant_id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate merchant_id"
