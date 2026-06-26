# 03 — Transformer Fine-Tuning (the teacher)

**Status:** 🚧 design spec (awaiting review → then build) · **Phase:** 3
**Depends on:** [01-data-pipeline](01-data-pipeline.md) (data), [02-baselines](02-baselines.md)
(the eval harness + the bar) · **Used by:** [05-distillation](05-distillation.md) (this is the
teacher), [06-optimization](06-optimization.md), [07-serving](07-serving.md), [09-mlops](09-mlops.md)

> Sections 1–11 are the plan to agree on before code; §12 (results) is filled after we build.

---

## 1. What this functionality is

Fine-tune a pretrained transformer encoder to classify descriptors into **subtype** (deriving
category), evaluate it on the **same harness** ([02](02-baselines.md)) against test + gold, and beat
the baseline bar (**gold subtype macro-F1 0.77 / category 0.85**). The winner becomes the **teacher**
distilled in [05](05-distillation.md). Experiment tracking via **MLflow starts here**.

## 2. Why fine-tuning (transfer learning) — the "why"

A pretrained encoder (DistilBERT) already "knows" language from massive pretraining. **Fine-tuning**
nudges those weights to our task with relatively little labeled data — vs. training from scratch
(needs far more data) or the TF-IDF baseline (no notion of token *meaning* or context, only surface
n-grams). The bet: the encoder's learned representations let it **generalize to unseen merchants**
better than n-grams — e.g. recognizing that "ROASTERS"/"CAFE"/"ESPRESSO" all signal coffee even for
a brand it never saw. That generalization is exactly where the baseline is weakest
([02 §9](02-baselines.md)).

Rejected framings: **zero-shot hosted LLM** (the thing we're replacing — too slow/costly at scale);
**from-scratch transformer** (data-hungry, pointless when pretraining exists).

## 3. Concepts you'll learn

Tokenization (subword/WordPiece), transfer learning vs feature extraction, the HF `Trainer` loop,
key hyperparameters (learning rate, warmup, weight decay, epochs, batch size, max sequence length),
logits → softmax → cross-entropy, class-weighted loss for imbalance, **temperature scaling** for
calibration, and MLflow experiment tracking.

## 4. Architecture & approach

### 4.1 Model choice
- **Primary: `distilbert-base-uncased`** — 66M params, ~2× faster/smaller than BERT-base at ~97% of
  its quality; CPU-friendly for the cheap serving target. Uncased is fine (descriptors are
  effectively case-insensitive).
- **Stretch: `microsoft/deberta-v3-small`** — stronger (disentangled attention), SentencePiece
  tokenizer; heavier. We compare it as a *better teacher* candidate for [05](05-distillation.md).
- **Teacher framing:** Phase 3 produces the best fine-tuned classifier; [05](05-distillation.md)
  decides the teacher→student pairing with numbers (DistilBERT is itself already distilled, so a
  DeBERTa-v3 teacher → small student may distill more cleanly — settled there).

Rejected: BERT-base / RoBERTa (bigger, marginal gain), large models (can't serve cheaply on CPU).

### 4.2 Tokenization
Descriptors are short (median ~20 chars → ~10–15 subword tokens), so **`max_length=48`** with
truncation/padding — long enough to never clip a real descriptor, short enough to keep training/
inference fast. Subword tokenization also helps noisy text (`WHOLEFDS` splits into known pieces).

### 4.3 Head: single subtype, derive category
One classification head over the **40 subtypes**; category is derived via `category_of_subtype`
(same contract as the baseline, so the harness plugs in unchanged). **Critical:** the model's
`id2label`/`label2id` MUST be built from `tx.SUBTYPES` order so output index *i* means
`tx.SUBTYPES[i]` — the same label-order discipline that bit us in the baseline's proba alignment.

Rejected: two heads (category + subtype) — more complex for little gain; revisit only if category
errors are dominated by cross-category subtype confusion.

### 4.4 Loss & class imbalance
Cross-entropy with **class weights** (inverse-frequency, like the baseline's `class_weight="balanced"`)
so rare subtypes aren't ignored — macro-F1 is the target. Label smoothing optional.

### 4.5 Calibration & abstention
After training, fit **temperature scaling** on the val split (one scalar that softens/sharpens
logits) to lower ECE, then pick an **abstention threshold** on val via the harness accuracy@coverage
curve ([00 abstention design](00-foundations.md)). Cheap, principled, and teaches calibration.

### 4.6 Numeric features
Text-only first (hold `amount`/`is_debit` for an **ablation**), matching the baseline's clean text
bar. If added later, fuse via a small MLP on the pooled embedding.

## 5. Training (HF `Trainer`)
Starting hyperparameters (tuned on val, tracked in MLflow): lr **2e-5–5e-5**, batch **32–64**,
**3–5 epochs**, weight decay 0.01, linear warmup ~10%, fp16 on GPU, seed fixed. Early-stop / select
the checkpoint by **val subtype macro-F1**, never gold (gold is touched only for the final number).

## 6. The Colab workflow (train on Colab, serve/dev locally)
Realizes [ADR 0002](decisions/0002-colab-local-split.md). The training logic lives in
`src/.../models/finetune.py` as importable functions; the Colab notebook is a thin driver:
1. `git clone` the repo, `pip install -e .` → identical code to local.
2. Mount Google Drive (checkpoints) and configure S3 creds (promoted artifacts).
3. Call `train_transformer(config)`; checkpoints stream to Drive (survive disconnects).
4. Log run to MLflow (§7); push the best checkpoint to S3.
Locally we run a **tiny CPU smoke** (1–2 steps, a few rows) to prove the pipeline before burning GPU.

## 7. MLflow tracking (starts now, not Phase 9)
- `MLFLOW_TRACKING_URI` → a **SQLite file on mounted Drive**; artifacts → S3 (or Drive locally).
- Per run we log: params (model, lr, bs, epochs, max_len, seed, **git SHA**), metrics
  (`EvalResult.to_dict()` for test + gold, ECE), and the model artifact.
- The **registry + promotion gate** is deferred to [09-mlops](09-mlops.md); here we just get
  durable, comparable run history.

## 8. Predictor wrapper (harness integration)
A `TransformerPredictor` implementing the harness `Predictor` interface: `predict` (argmax →
`tx.SUBTYPES`) and `predict_proba` (softmax, already in taxonomy order via `id2label`). Loads a saved
checkpoint and runs on CPU for evaluation — so the *same* `evaluate(...)` from [02](02-baselines.md)
scores it.

## 9. Planned module layout (built after approval)
```
src/transaction_intelligence/models/finetune.py   # build/train/eval; TransformerPredictor
src/transaction_intelligence/tracking.py           # MLflow setup + log_eval(EvalResult) helper
configs/finetune.yaml                              # model id, lr, bs, epochs, max_len, seed
scripts/train_transformer.py                       # CLI (tiny local run | full)
notebooks/03_finetune_colab.ipynb                  # thin Colab driver
tests/test_finetune.py                             # tiny CPU smoke; Predictor + proba-alignment
```
New deps (added this phase): `torch`, `transformers`, `datasets`, `mlflow`, `accelerate`
(+ `sentencepiece` for DeBERTa).

## 10. How to run / test (target)
```bash
# local: tiny smoke (CPU) to prove the pipeline
uv run pytest tests/test_finetune.py
uv run python scripts/train_transformer.py --smoke
# full run: the Colab notebook (GPU) -> checkpoint to Drive/S3 -> MLflow run
# then evaluate via the harness on test + gold
```

## 11. Key decisions + rejected alternatives
| Decision | Why | Rejected |
|---|---|---|
| DistilBERT primary | small/fast/CPU-servable, strong on short text | BERT/RoBERTa (heavier), large LLMs (uncheap) |
| single subtype head, derive category | reuses the harness unchanged; simple | 2-head model (complexity) |
| `max_length=48` | descriptors are short; fast | 128/512 (wasteful) |
| class-weighted CE | macro-F1 target, rare subtypes | plain CE (rare-class neglect) |
| temperature scaling on val | calibrated confidence for abstention | uncalibrated softmax |
| MLflow from Phase 3 | never lose a run; comparable history | tracking only in Phase 9 (re-runs) |
| text-only first | clean comparison to the baseline | fuse amount now (confounds the bar) |

## 12. Results / metrics
_TBD after build: DistilBERT (and DeBERTa-v3-small) subtype/category macro-F1 on test + gold vs the
baseline 0.77/0.85; ECE pre/post temperature scaling; accuracy@coverage; training curves; per-class
gains over the baseline; chosen teacher._

## 13. Gotchas (to confirm after build)
- **Label-order contract** — build `id2label` from `tx.SUBTYPES`; a mismatch silently scrambles
  every prediction.
- **Colab ephemerality** — checkpoint to Drive/S3 each epoch; never rely on Colab local disk.
- **Overfitting synthetic** — select on val macro-F1; the test→gold gap is the truth-teller.
- **`max_length` too small** clips descriptors; too big wastes compute.
- **DeBERTa-v3** needs `sentencepiece`; tokenizer differs from DistilBERT.

## 14. Glossary terms introduced
[tokenization](glossary.md), [subword tokenization](glossary.md), [transfer learning /
fine-tuning](glossary.md), [logits](glossary.md), [softmax](glossary.md), [cross-entropy](glossary.md),
[learning rate / warmup](glossary.md), [epoch / batch size](glossary.md), [temperature scaling](glossary.md),
[MLflow run / experiment](glossary.md).
