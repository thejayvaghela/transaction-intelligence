# 00 — Foundations & Setup

**Status:** ✅ done · **Phase:** 0 · **Depends on:** nothing · **Used by:** everything

This is the conceptual + schema groundwork. No models yet. We frame the problem, map the ML
lifecycle, and **lock the label taxonomy** — the single source of truth every later phase imports.

---

## 1. The problem: transaction categorization

Banks, fintechs, and personal-finance apps (Mint/YNAB/Plaid-style) ingest raw transaction feeds
and must turn each row into a clean, queryable **category** (for budgeting, analytics, lending,
accounting, fraud triage). The hard input is the **descriptor string** — the short, noisy raw
text of the transaction — optionally plus light numeric features (amount, sign, date).

```
input :  "SQ *BLUE BOTTLE 8005551234 CA"   (+ amount = -5.75)
output:  category = food_drink, subtype = food_drink/coffee, confidence = 0.94
```

## 2. Why descriptor strings are hard

Anatomy of a real-looking descriptor:

```
SQ *BLUE BOTTLE 8005551234 CA
└┬┘ └────┬────┘ └───┬────┘ └┤
 │       │          │       └─ state code (noise)
 │       │          └───────── phone / store id (noise)
 │       └──────────────────── the ACTUAL signal (merchant name, often truncated)
 └──────────────────────────── processor prefix — tells you Square, NOT the category
```

The difficulties, with examples:

- **The prominent token is a red herring.** `SQ *`, `PYPL *`/`PP*`, `TST*` (Toast), `SP `
  (Shopify), `DD *` (DoorDash) identify the **payment processor**, which fronts merchants of every
  category. The signal you want is the buried merchant name.
- **Extreme brevity + no grammar.** 10–40 chars, ALL CAPS, abbreviations — nothing like the prose
  encoders are pretrained on.
- **Truncation / many surface forms.** `WHOLEFDS`, `AMZN MKTP US*2K4...`, `MCDONALD'S F2345`. The
  same merchant appears as `AMZN` / `AMAZON.COM` / `AMZN MKTP US` — and `AMAZON PRIME` is a
  *different subtype* (subscription) from `AMZN MKTP` (marketplace).
- **Embedded junk:** phone numbers, terminal/store IDs, reference numbers, city + 2-letter state,
  country codes, dates.
- **Genuine ambiguity:** `APPLE.COM/BILL` could be App Store (software/subscription) or hardware;
  a bare `SQ *` merchant could be any category.
- **Drift:** new merchants and new processor formats appear constantly.

**Why this breaks two naive approaches:**
- **Regex/rules** can cover the top few hundred merchants, but the tail is endless and formats
  drift per bank/processor — unmaintainable.
- **Generic NLP / off-the-shelf LLMs** were pretrained on clean text; this distribution is alien to
  them. Fine-tuning on *in-domain* data is what closes the gap — the core skill of this project.

## 3. MCC codes (Merchant Category Codes)

4-digit ISO 18245 codes card networks assign to merchants. A few well-known ones:

| MCC | Meaning | Maps toward |
|-----|---------|-------------|
| 5812 | Eating places / restaurants | `food_drink/restaurant` |
| 5814 | Fast food | `food_drink/fast_food` |
| 5813 | Drinking places (bars) | `food_drink/bar` |
| 5411 | Grocery stores / supermarkets | `groceries/supermarket` |
| 5541 | Service stations (fuel) | `transportation/fuel` |
| 4121 | Taxicabs / limousines (rideshare) | `transportation/rideshare` |
| 5912 | Drug stores / pharmacies | `health_wellness/pharmacy` |
| 4511 | Airlines | `travel/airline` |
| 7011 | Hotels / motels / resorts | `travel/lodging` |
| 7512 | Automobile rental | `travel/car_rental` |

**Strong but leaky.** MCC is *almost* a label when present, but: it's often **not in the
descriptor string** (separate feed field many noisy sources lack); **aggregators** (Square, PayPal,
Stripe) stamp **one generic MCC on all sub-merchants**; and merchants get mis-coded.

**How we use it:** as a **weak-labeling signal + feature** in [01-data-pipeline](01-data-pipeline.md)
(map MCC → our taxonomy to bootstrap labels cheaply), *not* as ground truth. The model must work
from text, because that's the realistic hard case. Our categories are MCC-*inspired* but
consolidated into a balanced, human-meaningful set.

## 4. Why own the model vs. call a hosted LLM

| | Self-hosted fine-tuned encoder | Hosted LLM per row |
|---|---|---|
| Cost / row | ~free after training | real $; explodes at millions/day |
| Latency | single-digit ms (CPU, INT8) | 100s of ms–seconds |
| Control | versioned, calibrated, stable | prompt/provider drift |
| Privacy | data stays in-house | data leaves your perimeter |
| Cold-start | needs labeled data + retraining loop | strong zero-shot on novel tail |

The honest trade: we accept "needs labeled data + a retraining loop for drift" to get
cheap/fast/controlled/private inference. The [06-optimization](06-optimization.md) benchmark turns
the "~Y× cheaper" claim into a defensible number against a measured hosted-LLM baseline.

## 5. The ML lifecycle for this project

```
[Phase 0] Problem framing + fixed taxonomy        ← you are here
[Phase 1] Data: acquire · synthesize · weak-label · leakage-safe splits
[Phase 2] Baseline (TF-IDF + LogReg) + the REUSABLE eval harness
[Phase 3] Modeling: fine-tune the transformer teacher (+ MLflow tracking)
[Phase 5] Compression: knowledge distillation → tiny student
[Phase 6] Optimization: ONNX export + INT8 quantization + Pareto report
[Phase 7] Serving: FastAPI + ONNX (batch + online)
[Phase 8] Frontend: lean Next.js demo
[Phase 9] MLOps: registry + CI eval gate + drift monitoring
[Phase 10] Deploy (one Fargate run) + model card + filled-in resume bullet
          (Phase 4 NER is optional/deferred, hanging off Phase 3)
```

Two through-lines hold it together: **(a) the fixed taxonomy** below (one schema everywhere) and
**(b) evaluation rigor** — a never-trained-on **gold set** scored by **macro-F1** via a single
reusable harness, so every model is comparable apples-to-apples.

## 6. The label taxonomy (the spec)

Decisions and rationale are recorded in
[decisions/0003-label-taxonomy.md](decisions/0003-label-taxonomy.md). Summary:

- **Single-label multi-class** — exactly one category per transaction.
- **Two-level hierarchy** — 11 categories, each with **namespaced** subtypes (`food_drink/coffee`).
  Category is the primary classification head; subtype a secondary head.
- **Unknown/long-tail → abstention**, not an explicit `other` class (route low-confidence
  predictions to "needs review"; see [glossary](glossary.md)).
- **Cross-cutting attributes are not categories** — "recurring vs one-off" is a deferred optional
  binary head; "fee/transfer" is absorbed into `financial`/`transfers`.

| Category | Subtypes |
|---|---|
| `food_drink` | restaurant, coffee, fast_food, bar, delivery |
| `groceries` | supermarket, convenience, specialty |
| `shopping` | general_merch, clothing, electronics, online_marketplace |
| `transportation` | rideshare, fuel, transit, parking_tolls |
| `travel` | airline, lodging, car_rental |
| `bills_utilities` | electric_gas, water, internet_phone, insurance |
| `subscriptions_entertainment` | streaming, gaming, events, software |
| `health_wellness` | pharmacy, medical, fitness |
| `financial` | bank_fee, atm_cash, interest, investment |
| `transfers` | p2p, internal, wire |
| `income` | payroll, refund, deposit |

→ 11 categories, 40 subtypes.

## 7. How it's implemented

`src/transaction_intelligence/taxonomy.py` is the single source of truth. It exposes the
`TAXONOMY` dict, ordered `CATEGORIES`/`SUBTYPES` lists, `*_TO_ID` / `ID_TO_*` mappings, helpers
(`make_subtype`, `category_of_subtype`, `subtypes_of`), and a `validate()` that runs **at import
time** so a malformed edit fails fast.

```bash
uv run python -c "from transaction_intelligence import taxonomy as t; print(t.NUM_CATEGORIES, t.NUM_SUBTYPES)"
uv run pytest tests/test_taxonomy.py
```

## 8. Gold eval set policy

The **gold set** is a small (~a few hundred rows), human-verified slice of descriptors with
correct category+subtype labels. Rules:
- **Never trained on.** It is the fixed yardstick; training on it would make every metric a lie.
- **Committed to git** at `data/gold/` (small + must stay stable + versioned).
- **Built in [01-data-pipeline](01-data-pipeline.md)** by hand-verifying a stratified sample of
  synthesized + real descriptors, covering every category (and the awkward edge cases).

## Results / status

Phase 0 deliverables complete: repo skeleton + docs system (Phase 0 scaffold commit), this
foundations doc, ADR 0003, and `taxonomy.py` (11 categories / 40 subtypes, validated, tested).

## Gotchas

- **Label-ID stability is a contract** — `taxonomy.py` is append-only; reorder/delete breaks saved
  models. Bump `TAXONOMY_VERSION` if the schema ever changes.
- **MCC ≠ ground truth** — aggregator merchants share one generic MCC; treat MCC as weak signal.
- **Subtype collisions** are avoided only because subtypes are namespaced under their category.

## Glossary terms introduced

[MCC](glossary.md), [single-label multi-class](glossary.md), [abstention / confidence
threshold](glossary.md), [taxonomy & descriptor string](glossary.md).
