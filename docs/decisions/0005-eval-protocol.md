# 0005 — Evaluation protocol: one shared harness, macro-F1 on test + gold

**Status:** Accepted · 2026-06-26

## Context

Every model in this project (rules floor, baseline, transformer teacher, distilled student,
quantized student) must be comparable, and the comparison must be CI-gateable later
([09](../09-mlops.md)). That requires a single, fixed measurement protocol decided up front.

## Decisions

1. **One model-agnostic eval harness** is THE measurement contract. Any model adapted to a tiny
   `Predictor` interface is scored identically. No per-model bespoke metrics.
2. **Headline metric = macro-F1** (equal weight per class), reported for both the **subtype** (40-way)
   and **category** (11-way) heads.
3. **Predict the finest level (subtype); derive category** via `category_of_subtype`. One model,
   two metric views.
4. **Evaluate on both `test` (in-distribution) and `gold` (transfer).** The test→gold gap is a
   first-class reported number (the synthetic-to-real gap).
5. **Report calibration (ECE) and accuracy@coverage**, not just top-1 accuracy — calibration gates
   the abstention design ([00](../00-foundations.md)), and accuracy@coverage is the fair way to
   compare against the abstaining rules floor.

## Alternatives rejected

- **Per-model bespoke metrics** — incomparable; the classic way teams flatter a new model.
- **Accuracy as headline** — hides rare-class failure under class imbalance.
- **Category-only evaluation** — loses the subtype granularity the product promises.
- **Test-only evaluation** — overstates real-world skill; hides the synthetic-to-real gap.

## Consequences

- Built in `src/transaction_intelligence/eval/` and reused unchanged by every later model phase.
- `EvalResult.to_dict()` is the payload logged to MLflow and checked by the CI eval gate.
