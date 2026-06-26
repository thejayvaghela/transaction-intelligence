# data/

| Folder | Contents | Git |
|--------|----------|-----|
| `raw/` | Original downloaded sources (Kaggle sets, MCC list, gazetteer) | **ignored** — reproducible via scripts |
| `interim/` | Intermediate transforms | **ignored** |
| `processed/` | Final model-ready datasets (train/val/test) | **ignored** |
| `gold/` | Small, hand-labeled eval slice we **never train on** | **committed** — the fixed yardstick every model is measured against |

The split between *committed* and *ignored* is deliberate: large/derived data is regenerated
from code (see [docs/01-data-pipeline.md](../docs/01-data-pipeline.md)), while the gold set is a
small, human-verified artifact that must stay stable and versioned.
