"""Build the dataset end-to-end: gazetteer -> synthesize -> split -> write parquet.

Run via ``make data`` or ``python -m transaction_intelligence.data.build``.
Everything is driven by configs/data.yaml and seeded, so output is reproducible.
See docs/01-data-pipeline.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import yaml

from .gazetteer import load_gazetteer
from .splits import split_merchants
from .synthesize import generate_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"
SPLITS = ("train", "val", "test")


def build_dataset(
    config_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    rows_per_merchant: int | None = None,
) -> dict:
    """Generate the splits, write `{train,val,test}.parquet`, and return a summary dict."""
    cfg = yaml.safe_load(Path(config_path or DEFAULT_CONFIG).read_text())
    rpm = rows_per_merchant if rows_per_merchant is not None else cfg["rows_per_merchant"]
    out = Path(out_dir) if out_dir is not None else REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    entries = load_gazetteer(REPO_ROOT / cfg["gazetteer"])
    assignment = split_merchants(entries, cfg["split"], cfg["seed"])
    rng = random.Random(cfg["seed"])
    rows = generate_rows(entries, rpm, rng, cfg["noise_level"])

    df = pd.DataFrame(rows)
    df["split"] = df["merchant_id"].map(assignment)

    summary: dict = {
        "config": {
            "seed": cfg["seed"],
            "rows_per_merchant": rpm,
            "noise_level": cfg["noise_level"],
        },
        "merchants": len(entries),
        "total_rows": int(len(df)),
        "splits": {},
    }
    for split in SPLITS:
        sdf = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        sdf.to_parquet(out / f"{split}.parquet", index=False)
        summary["splits"][split] = {
            "rows": int(len(sdf)),
            "merchants": int(sdf["merchant_id"].nunique()),
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2))
