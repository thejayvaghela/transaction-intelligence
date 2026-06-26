# 02 — Baselines & the Reusable Eval Harness

**Status:** 🚧 design spec (awaiting review → then build) · **Phase:** 2
**Depends on:** [01-data-pipeline](01-data-pipeline.md) (splits + gold) · **Used by:**
[03-finetuning](03-finetuning.md), [05-distillation](05-distillation.md),
[06-optimization](06-optimization.md), [09-mlops](09-mlops.md) (the harness is reused everywhere)

> Sections 1–8 are the plan to agree on before code; §9 (results) is filled after we build.

---

## 1. What this functionality is

Two deliverables:
1. A **model-agnostic evaluation harness** — the single source of truth for "how good is a model."
   Every later model (baseline, teacher, student, quantized) is scored through it, so comparisons
   are apples-to-apples.
2. A **TF-IDF + Logistic Regression baseline** — the honest bar the transformer must beat to justify
   its cost.

We then compare the baseline against the **rules floor** (the weak labeler from
[01](01-data-pipeline.md)) on the same harness.

## 2. Why baselines (and a *shared* harness) come first

- A baseline answers "is the fancy model worth it?" If TF-IDF+LogReg hits 0.85 macro-F1 and a
  fine-tuned transformer hits 0.86 at 100× the cost/latency, that's a finding, not a failure.
- Building the harness **once, correctly** means the teacher/student/quantized numbers are directly
  comparable and CI-gateable later ([09](09-mlops.md)). Bespoke per-model metrics are how teams
  fool themselves. (Rationale: [decisions/0005-eval-protocol.md](decisions/0005-eval-protocol.md).)

## 3. Concepts you'll learn

- **TF-IDF** and **n-grams** — turning text into sparse vectors; **char n-grams** vs word n-grams,
  and why char n-grams shine on truncated/abbreviated descriptors (`WHOLEFDS`, `NTFLX`).
- **Logistic regression** for multi-class text; `class_weight="balanced"` for imbalance.
- **Evaluation rigor** — macro vs micro F1, per-class precision/recall, the **confusion matrix**,
  and reading them to find which subtypes collide.
- **Calibration** (reliability, **ECE**) — a model can be accurate yet overconfident; calibration
  is what makes the abstention threshold ([00](00-foundations.md)) trustworthy.
- **Accuracy@coverage** — the curve you get by abstaining below a confidence threshold; the fair way
  to compare against the rules floor (which abstains on 50%).

## 4. The eval harness (the heart)

### 4.1 Design goals
- **Model-agnostic:** anything implementing a tiny `Predictor` interface can be scored — sklearn
  pipeline, the weak labeler, later a transformer, later an ONNX session.
- **Two heads:** report **subtype** (40-way) *and* **category** (11-way) metrics. The baseline
  predicts the finest level (subtype); category is *derived* via `category_of_subtype`.
- **Two eval sets:** **test** (synthetic, in-distribution) *and* **gold** (realistic, transfer). The
  test→gold gap quantifies the synthetic-to-real gap ([ADR 0004](decisions/0004-synthetic-first-data.md)).
- **Serializable:** results → dict/JSON for logging to MLflow and for the CI eval gate.

### 4.2 The `Predictor` interface
```python
class Predictor(Protocol):
    def predict(self, descriptors: list[str]) -> list[str]: ...        # returns subtype labels
    def predict_proba(self, descriptors: list[str]) -> np.ndarray | None: ...  # optional, for calibration/coverage
```
The weak labeler is adapted to this interface (abstain → a sentinel that counts as wrong on the
full set but is excluded above a coverage threshold).

### 4.3 What it computes
| Metric | Why |
|---|---|
| **macro-F1** (subtype + category) | headline; equal weight per class under imbalance |
| accuracy, micro-F1, weighted-F1 | secondary context |
| per-class precision/recall/F1 | which subtypes fail |
| confusion matrix (category-level) | which categories collide (fast_food↔restaurant) |
| **ECE** + reliability bins | is confidence trustworthy? (gates abstention) |
| **accuracy@coverage** curve | fair comparison vs the abstaining rules floor |
| test vs gold side-by-side | the synthetic-to-real gap |

Confusion matrices/curves are emitted as structured data (DataFrame/dict) + a text report — no
plotting dependency yet (kept for an optional notebook later; flagged to avoid premature tooling).

## 5. The baseline model

- **Pipeline:** `FeatureUnion(word TF-IDF (1,2) + char_wb TF-IDF (2,5)) → LogisticRegression
  (multinomial, class_weight="balanced")`. Trained on the **train** split to predict **subtype**.
- **Why char n-grams:** descriptors are truncated/concatenated (`WHOLEFDS`, `AMZN MKTP`); subword
  character patterns survive that corruption far better than whole-word tokens.
- **Text-only first.** The numeric `amount`/`is_debit` features are held for an *ablation* (does
  adding them help income/transfers?) — kept out of the first baseline for a clean text bar.
- Wrapped as a `Predictor`; persisted with joblib to `models/` (gitignored).

## 6. Planned module layout (built after approval)

```
src/transaction_intelligence/eval/
  __init__.py
  metrics.py      # pure: macro_f1, per-class report, confusion df, ECE, accuracy@coverage
  harness.py      # Predictor protocol, EvalResult dataclass, evaluate(predictor, df) (both heads)
src/transaction_intelligence/models/
  __init__.py
  baseline.py     # build/train/save/load the TF-IDF+LogReg pipeline; Predictor wrapper
  rules.py        # adapt the weak labeler to the Predictor interface
scripts/train_baseline.py    # -> `make baseline`: train, eval on test+gold, print report, save
tests/test_metrics.py        # toy y_true/y_pred -> known macro-F1, ECE, coverage curve
tests/test_baseline.py       # trains on a small subset, beats a trivial floor; Predictor conformance
```

## 7. How to run / test (target)

```bash
make data        # ensure splits exist
make baseline    # train TF-IDF+LogReg, evaluate on test + gold, print the comparison report
uv run pytest tests/test_metrics.py tests/test_baseline.py
```

## 8. Key decisions + rejected alternatives

| Decision | Why | Rejected |
|---|---|---|
| One shared, model-agnostic harness | comparability across all models; CI-gateable | per-model bespoke metrics (incomparable) |
| headline = **macro-F1** | fair to rare classes under imbalance | accuracy (hides rare-class failure) |
| predict subtype, derive category | one model, both metric views; natural hierarchy | two separate models (more complexity for a baseline) |
| char + word n-grams | robust to truncation/abbreviation | word-only (brittle on `WHOLEFDS`) |
| evaluate on test **and** gold | exposes synthetic-to-real gap | test-only (overstates real-world skill) |
| report ECE + accuracy@coverage | calibration gates the abstention design | top-1 accuracy only |

## 9. Results / metrics

_TBD after build: baseline macro-F1 (subtype + category) on test and gold; vs the rules floor
(50% coverage @ 97%); ECE; accuracy@coverage curve; top confused subtype pairs; the test→gold gap._

## 10. Gotchas (to confirm after build)

- **The test→gold drop is the real story** — a high test macro-F1 with a big gold drop = the model
  learned the synthesizer.
- **Calibration ≠ accuracy** — `class_weight="balanced"` and LogReg can be miscalibrated; ECE checks.
- **Abstention fairness** — when comparing to the rules floor, compare at matched coverage, not just
  full-set macro-F1.

## 11. Glossary terms introduced

[TF-IDF](glossary.md), [n-gram (char vs word)](glossary.md), [logistic regression](glossary.md),
[precision / recall / F1](glossary.md), [confusion matrix](glossary.md),
[calibration / ECE](glossary.md), [accuracy@coverage](glossary.md).
