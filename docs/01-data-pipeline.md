# 01 — Data Pipeline

**Status:** 🚧 design spec (awaiting review → then build) · **Phase:** 1
**Depends on:** [00-foundations](00-foundations.md) (the taxonomy) · **Used by:**
[02-baselines](02-baselines.md), [03-finetuning](03-finetuning.md), and every model after

> This document is the *design* for Phase 1. Sections 1–8 are the plan we agree on before writing
> code; §9 (results) and the confirmed parts of §10 (gotchas) get filled in **after** we build.

---

## 1. What this functionality is

Produce a **reproducible, labeled dataset** of transaction examples plus **leakage-safe splits**
and a **hand-verified gold set**, so Phase 2+ has something honest to train and measure on.

Each example:

```
descriptor   : "SQ *BLUE BOTTLE 8005551234 CA"   # the noisy text — the model's main input
amount       : -5.75                              # light numeric feature
is_debit     : true
category     : food_drink                         # label (primary head)
subtype      : food_drink/coffee                  # label (secondary head)
merchant_id  : blue_bottle                        # GROUP KEY for leakage-safe splitting (not a feature)
source       : synthetic                          # provenance
```

Artifacts produced (details in §5): `data/processed/{train,val,test}.parquet`,
`data/gold/gold.csv` (committed), a **dataset card**, and a thin EDA notebook.

## 2. Strategy: synthetic-first (with real data as a reality check)

We **generate** the bulk of our data from a curated **merchant gazetteer** run through a **noise
model**, rather than hunting for a large pre-labeled corpus. Rationale and rejected alternatives
are in [decisions/0004-synthetic-first-data.md](decisions/0004-synthetic-first-data.md). In short:

- No public dataset matches our exact 11-category / 40-subtype taxonomy with clean labels; real
  bank data is private; Kaggle sets are small and use different schemas.
- A synthesizer gives **abundant data, perfect labels, controllable class balance, and a difficulty
  knob** — and *building it teaches the domain* (you encode what makes descriptors hard).

**The risk, named up front:** a model can learn *the synthesizer* instead of *reality* (the
**synthetic-to-real gap**). Mitigations baked into the design:
1. **Ground every noise source in real artifacts** — real merchant names, real processor prefixes,
   real MCC codes, real US city/state lists.
2. **Build the gold set to be as real as possible** (hand-authored realistic descriptors + any real
   samples we can find), so our yardstick measures *transfer to reality*, not synthetic
   self-consistency.
3. **Optional secondary "reality-check" eval** on a small slice of real Kaggle descriptors
   weak-labeled and spot-verified.

## 3. Concepts you'll learn

- **Synthetic data & weak supervision** — generating labeled data, and labeling real data cheaply
  with rules (MCC maps, keyword/gazetteer matching) instead of manual annotation.
- **Data leakage & group splitting** — *the single most important idea in this phase.* If the same
  merchant appears in both train and test, the model memorizes `AMZN → shopping` and your test
  macro-F1 is a lie. We split by **merchant group**, so test merchants are *unseen*. The real task
  is generalizing to new merchants and new surface forms.
- **Class imbalance & stratification** — real spend is skewed; why we keep splits stratified and
  lean on **macro-F1** (equal weight per class) rather than accuracy.
- **Dataset card** — documenting provenance, generation params, distribution, and limitations as a
  first-class MLOps artifact.

## 4. Pipeline architecture

```
 gazetteer (merchant → category/subtype, aliases, MCC, channel)
      │
      ▼
 descriptor synthesizer  ──► (descriptor, amount, is_debit, labels, merchant_id, source)
   (probabilistic noise model)            │
                                          ▼
                          group-aware stratified split (by merchant_id)
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   train.parquet     val.parquet      test.parquet
                                          
 gold set (hand-verified, realistic) ──► data/gold/gold.csv   ← SEPARATE, never in the splits
```

### 4.1 Merchant gazetteer
A curated table seeding ground truth. Each row: canonical merchant, **surface-form aliases**
(`amazon` → `AMZN`, `AMAZON.COM`, `AMZN MKTP US`), `category`, `subtype`, a typical **MCC**, and a
**channel** (card-present / online / recurring) that influences the descriptor format. Seeded by
hand to cover **all 40 subtypes** (a few hundred merchants); optionally expanded once, offline, with
an LLM for breadth. Stored committed at `data/gazetteer/` (small + curated, like the gold set).

### 4.2 Descriptor synthesizer (the noise model) — the heart of Phase 1
Turns one gazetteer entry into a realistic descriptor by composing **probabilistic transforms**,
each grounded in real patterns. A `noise_level` knob scales their aggressiveness (enables a
robustness/curriculum eval). Representative transforms:

| Transform | Example effect |
|---|---|
| Pick a surface form | `Blue Bottle Coffee` → `BLUE BOTTLE` |
| Uppercase / strip punctuation | `McDonald's` → `MCDONALDS` |
| Processor/acquirer prefix | prepend `SQ *`, `TST*`, `PYPL *`, `PP*`, `DD *`, `POS`, `WWW.` (consistent with channel) |
| Truncation / abbreviation | clip to ~22–25 chars; drop vowels (`WHOLE FOODS` → `WHOLEFDS`) |
| Suffix noise | append phone (`8005551234`), store id (`#1234`), city + 2-letter state (`SAN FRANCISCO CA`), date (`03/14`) |
| Separator variety | spaces / `*` / `-` / `#` |

> The exact prefix and MCC tables will be **grounded in public references at build time** and
> recorded in the dataset card — not invented.

**Determinism:** a fixed RNG seed (in `configs/data.yaml`) → byte-identical output, so `make data`
is reproducible and a test can assert it.

### 4.3 Weak labeling (for any *real* data we ingest)
For unlabeled real descriptors: label cheaply via (a) **MCC → taxonomy map** when MCC is present,
(b) **gazetteer / keyword matching** on tokens, abstaining when confidence is low. Produces noisy
labels we spot-verify. This is the *weak-supervision* skill; we use it mainly to build the
reality-check eval, not the training set.

### 4.4 Numeric / light features
`amount` (+ sign `is_debit`) and optionally day-of-week. These disambiguate (income/transfers are
credits; coffee is small; rent is large). Generated per-subtype from plausible amount distributions.
**Text stays the primary signal**; whether/how to *fuse* numerics into the model is decided in
[03-finetuning](03-finetuning.md). Phase 1 just produces the columns.

### 4.5 Splits — leakage-safe (read this twice)
- Split by **`merchant_id` group** using scikit-learn `StratifiedGroupKFold` / `GroupShuffleSplit`
  → **no merchant in more than one split**.
- **Stratify by category** so class proportions match across train/val/test.
- Ratios ~**70 / 15 / 15**. The **test split is for development iteration**; the **gold set is the
  final yardstick** and is never in these splits.
- A unit test will **assert zero merchant overlap** across splits — the leakage guard.

### 4.6 Gold set
~**300–500 rows**, hand-verified, **stratified across all categories** (and deliberately loaded with
tricky/edge cases: aggregator `SQ *` merchants, ambiguous `APPLE.COM/BILL`, truncations). Committed
to `data/gold/` (see [data/README.md](../data/README.md)). Built by generating/collecting realistic
descriptors and **human-checking each label**. Reused by the eval harness in
[02-baselines](02-baselines.md).

## 5. Output schema & artifacts

| Artifact | Columns / contents | Git |
|---|---|---|
| `data/processed/{train,val,test}.parquet` | `descriptor, amount, is_debit, category, subtype, merchant_id, source, noise_level` | ignored (regenerated) |
| `data/gold/gold.csv` | same + a `notes` column for tricky cases | **committed** |
| `data/gazetteer/merchants.csv` | gazetteer (see §4.1) | **committed** |
| dataset card (`docs/` or `data/`) | provenance, params, distribution, limitations | committed |
| `notebooks/01_eda.ipynb` | length/token stats, class balance, sample descriptors | committed (thin) |

## 6. Planned module layout (built after approval)

```
src/transaction_intelligence/data/
  gazetteer.py     # load / represent the gazetteer
  synthesize.py    # the noise model + descriptor generator (seedable)
  weak_label.py    # MCC + keyword/gazetteer weak labeler (for real data)
  splits.py        # group-aware stratified splitting
  build.py         # orchestrate: gazetteer → synthesize → split → write
scripts/build_dataset.py        # thin CLI → `make data`
configs/data.yaml               # seed, counts, noise params, split ratios
notebooks/01_eda.ipynb
tests/test_synthesize.py        # determinism; output looks well-formed
tests/test_splits.py            # LEAKAGE GUARD: no merchant_id across splits; all classes present
tests/test_weak_label.py        # MCC/keyword mapping correctness
```

## 7. How to run / test (target)

```bash
make data                       # regenerate the dataset deterministically from configs/data.yaml
uv run pytest tests/test_splits.py      # leakage guard + class coverage
uv run pytest tests/test_synthesize.py  # determinism + sanity
```

We'll also **eyeball ~20 synthesized samples** per a few categories — the cheapest, highest-signal
check that the noise model looks real.

## 8. Key decisions + rejected alternatives

| Decision | Why | Rejected |
|---|---|---|
| Synthetic-first generation | abundant, perfectly-labeled, controllable, teaches the domain | pure real-data (none fits taxonomy); LLM-generated bulk (costly, less controllable, ironic) |
| Ground all noise in real artifacts + real gold set | shrink the synthetic-to-real gap | free-form synthetic (model learns the generator) |
| Split by **merchant group** | measure generalization to unseen merchants; avoid leakage | random row split (inflates metrics — memorization) |
| Stratify + headline **macro-F1** | fairness to rare classes under imbalance | accuracy (hides rare-class failure) |
| Gold set separate & committed | a stable, honest, versioned yardstick | reusing the test split as the final metric |

## 9. Results / metrics

_TBD after build: dataset sizes, # merchants, class distribution, descriptor length stats, gold-set
size + per-class counts, sample descriptors._

## 10. Gotchas

- **Synthetic-to-real gap** — watch for a model that aces `test` but stumbles on `gold`; gold is the
  truth-teller.
- **Gazetteer coverage bias** — whatever merchants we seed is what the model sees; keep breadth per
  subtype.
- **Over-noising** — too much corruption makes data unlearnable; the `noise_level` knob must be
  tuned, not maxed.
- **Aggregator MCC ambiguity** — `SQ *` merchants share a generic MCC; don't trust MCC alone (see
  [00-foundations §3](00-foundations.md)).
- **Leakage creeps back** — any new join/dedup step can re-mix merchants across splits; the leakage
  test is the backstop.

## 11. Glossary terms introduced

[synthetic data](glossary.md), [weak supervision / weak labeling](glossary.md),
[data leakage](glossary.md), [group split](glossary.md), [stratification](glossary.md),
[dataset card](glossary.md), [gazetteer](glossary.md), [synthetic-to-real gap](glossary.md).
