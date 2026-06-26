"""Merge LLM-generated merchants (expand-gazetteer workflow output) into the gazetteer YAML.

Reads an expansion JSON ([{subtype, merchants:[{name, aliases}]}, ...]), combines it with the
existing gazetteer (preserving per-subtype mcc/channel and the curated existing merchants), dedupes
by normalized name GLOBALLY (so a name never lands in two subtypes), and rewrites
data/gazetteer/gazetteer.yaml. One-off data-build utility.

Usage: python scripts/merge_gazetteer.py <expansion.json> [cap_per_subtype]
"""

import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
GAZ = REPO / "data" / "gazetteer" / "gazetteer.yaml"

HEADER = (
    "# Merchant gazetteer — curated seed + LLM-generated expansion (workflow: expand-gazetteer).\n"
    "# Keyed by namespaced subtype. Per-subtype mcc/channel defaults; merchants are a bare name\n"
    "# or {name, aliases}. See docs/01-data-pipeline.md.\n\n"
)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def name_of(item) -> str:
    return item if isinstance(item, str) else item["name"]


def main(expansion_path: str, cap: int = 40) -> None:
    raw = yaml.safe_load(GAZ.read_text())  # subtype -> {mcc, channel, merchants}
    expansion = {d["subtype"]: d["merchants"] for d in json.loads(Path(expansion_path).read_text())}

    seen = {norm(name_of(it)) for block in raw.values() for it in block["merchants"]}

    for subtype, block in raw.items():
        merchants = list(block["merchants"])
        for m in expansion.get(subtype, []):
            if len(merchants) >= cap:
                break
            nm = (m.get("name") or "").strip()
            n = norm(nm)
            if not nm or n in seen:
                continue
            seen.add(n)
            aliases = [a.strip() for a in (m.get("aliases") or []) if a and a.strip()]
            merchants.append({"name": nm, "aliases": aliases} if aliases else nm)
        block["merchants"] = merchants

    body = yaml.safe_dump(
        raw, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )
    GAZ.write_text(HEADER + body)
    counts = {s: len(b["merchants"]) for s, b in raw.items()}
    total = sum(counts.values())
    print(f"wrote {GAZ}")
    print(f"  {total} merchants across {len(raw)} subtypes")
    print(f"  per-subtype min/max = {min(counts.values())}/{max(counts.values())}")


if __name__ == "__main__":
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(path, cap)
