"""Loaders for the built dataset splits and the hand-verified gold eval set.

Reused by the eval harness ([02-baselines]) and all later models. Keeps the on-disk formats
(parquet splits, committed gold CSV) in one place so callers never re-parse them ad hoc.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold.csv"

_BOOL = {"true": True, "false": False}


def load_split(split: str) -> pd.DataFrame:
    """Load a built split: 'train' | 'val' | 'test' (run `make data` first)."""
    return pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")


def load_gold() -> pd.DataFrame:
    """Load the committed gold eval set, with bool columns parsed and notes filled."""
    df = pd.read_csv(GOLD_PATH)
    df["is_debit"] = df["is_debit"].astype(str).str.lower().map(_BOOL)
    df["merchant_unseen"] = df["merchant_unseen"].astype(str).str.lower().map(_BOOL)
    df["notes"] = df["notes"].fillna("")
    return df
