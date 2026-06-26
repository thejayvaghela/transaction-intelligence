"""End-to-end build smoke test — writes to a tmp dir, asserts no leakage in the written data."""

import pandas as pd

from transaction_intelligence.data.build import SPLITS, build_dataset

REQUIRED_COLS = {"descriptor", "amount", "is_debit", "category", "subtype", "merchant_id"}


def test_build_writes_splits_without_leakage(tmp_path):
    summary = build_dataset(out_dir=tmp_path, rows_per_merchant=3)
    assert summary["total_rows"] > 0
    assert sum(s["rows"] for s in summary["splits"].values()) == summary["total_rows"]

    frames = {s: pd.read_parquet(tmp_path / f"{s}.parquet") for s in SPLITS}
    ids = {s: set(df["merchant_id"]) for s, df in frames.items()}
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["val"].isdisjoint(ids["test"])
    assert REQUIRED_COLS.issubset(frames["train"].columns)
