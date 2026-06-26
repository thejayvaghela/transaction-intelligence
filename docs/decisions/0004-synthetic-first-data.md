# 0004 — Synthetic-first data generation (real data as reality check)

**Status:** Accepted · 2026-06-26

## Context

Phase 1 needs a labeled dataset matching our 11-category / 40-subtype taxonomy
([0003](0003-label-taxonomy.md)). No public dataset matches it cleanly: real bank data is private,
and public Kaggle transaction sets are small, inconsistently labeled, and use different taxonomies.

## Decision

Generate the **bulk of training data synthetically** — a curated merchant **gazetteer** run through
a probabilistic **noise model** that emulates real descriptor corruption — and use **real data only
as a reality check** (a small weak-labeled, spot-verified eval slice) and to **ground** the
synthesizer (real merchant names, processor prefixes, MCC codes, city/state lists). The **gold set**
is built to be as real as possible.

## Why / consequences

- Abundant data with **perfect labels**, **controllable class balance**, and a **difficulty knob**.
- Building the synthesizer **teaches the domain** — you encode exactly what makes descriptors hard.
- **Main risk: the synthetic-to-real gap** — a model can learn the generator, not reality. Countered
  by grounding all noise in real artifacts and by measuring on a realistic, hand-verified gold set
  (and an optional real-data reality-check eval). If `gold` and `test` diverge sharply, the gap is
  real and we enrich the gazetteer / noise model.

## Alternatives rejected

- **Pure real-data** — would spend the phase cleaning/relabeling mismatched public sets, with too
  few examples per subtype.
- **LLM-generated bulk data** — costly at scale, less controllable, and ironically depends on the
  hosted-LLM approach we're replacing. Acceptable *sparingly + offline* to widen the gazetteer.
- **Free-form synthetic (ungrounded)** — easy but maximizes the synthetic-to-real gap.
