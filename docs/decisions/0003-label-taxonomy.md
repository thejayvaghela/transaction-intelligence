# 0003 — Label taxonomy: single-label, 11 categories, namespaced subtypes

**Status:** Accepted · 2026-06-26

## Context

Phase 0 must lock the label schema before any data or modeling. The schema is the most expensive
thing to change later — it's baked into trained weights and every downstream artifact.

## Decisions

1. **Single-label multi-class.** Each transaction gets exactly one category. Most transactions have
   one true category; this gives clean macro-F1 / confusion matrices and simple calibration.
2. **Two-level hierarchy: 11 categories, each with namespaced subtypes** (`food_drink/coffee`).
   Modeled as a primary category target plus a secondary subtype head. Namespacing prevents
   cross-category subtype collisions (e.g. "delivery", "specialty") and keeps the subtype label
   space unambiguous.
3. **Unknown / long-tail → abstention, not an explicit `other` class.** At serving time, if the top
   calibrated probability is below a threshold, route to "needs review" rather than forcing a label.
   Keeps the label space clean and turns "I don't know" into a calibration problem (a stated
   learning goal) instead of a junk-collection class.
4. **Cross-cutting attributes are not categories.** "Recurring vs one-off" is deferred to an
   optional separate binary head; "fee/transfer" is absorbed into `financial` / `transfers`.

## Alternatives rejected

- **Multi-label** — more flexible (e.g. food + recurring) but needs per-class thresholds and a
  harder eval story; more ML surface for little gain at this scope.
- **Flat (no subtypes)** — simpler, but loses the merchant-level granularity the resume bullet
  implies ("category + subtype").
- **Explicit `other` class** — common in production, but with synthetic data we control the
  distribution; abstention is cleaner and more instructive here. Revisit if real data shows a large
  un-mappable tail.

## Consequences

- Encoded in `src/transaction_intelligence/taxonomy.py` as the single source of truth, validated at
  import time. Label IDs derive from declaration order and are **append-only** (see the stability
  contract in that module). `TAXONOMY_VERSION` starts at `1.0.0`.
- Downstream: weak-labeling ([01](../01-data-pipeline.md)), the eval harness ([02](../02-baselines.md)),
  and all model heads ([03](../03-finetuning.md), [05](../05-distillation.md)) import labels from here.
