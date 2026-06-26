"""CLI: regenerate the dataset deterministically from configs/data.yaml (`make data`)."""

import json

from transaction_intelligence.data.build import build_dataset

if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2))
