# 0006 — Fine-tuning approach: DistilBERT, single subtype head, MLflow from now

**Status:** Accepted · 2026-06-26

## Context

Phase 3 fine-tunes a transformer to beat the baseline (gold subtype macro-F1 0.77 / category 0.85)
on the shared harness, and to become the teacher for distillation ([05](../05-distillation.md)).
Training runs on Colab ([ADR 0002](0002-colab-local-split.md)).

## Decisions

1. **Primary model = `distilbert-base-uncased`** (DeBERTa-v3-small as a stretch/teacher candidate) —
   small, fast, CPU-servable for the cheap-inference target; strong on short text.
2. **Single classification head over the 40 subtypes; derive category** via `category_of_subtype`.
   Reuses the eval harness unchanged. `id2label`/`label2id` built from `tx.SUBTYPES` order (label
   contract).
3. **Class-weighted cross-entropy** (inverse-frequency) — macro-F1 is the target; `max_length=48`
   (descriptors are short).
4. **Calibrate with temperature scaling on val**, then set an abstention threshold via the harness
   accuracy@coverage curve.
5. **MLflow tracking starts in Phase 3** (SQLite on Drive + S3 artifacts); the registry + CI
   promotion gate stay in [09](../09-mlops.md).
6. **Text-only first**; numeric `amount`/`is_debit` fusion is a later ablation.

## Alternatives rejected

- **BERT-base / RoBERTa / large LLMs** — heavier, costlier to serve, marginal gain at this scope.
- **Two heads (category + subtype)** — added complexity for little expected gain; revisit only if
  category errors are dominated by cross-category subtype confusion.
- **Tracking only in Phase 9** — would mean re-running experiments; logging from the first run is
  cheap and realistic.

## Consequences

- `src/.../models/finetune.py` (importable training fns + `TransformerPredictor`), `tracking.py`
  (MLflow helper), `configs/finetune.yaml`, a thin Colab notebook driver. New deps: torch,
  transformers, datasets, mlflow, accelerate (+ sentencepiece for DeBERTa).
