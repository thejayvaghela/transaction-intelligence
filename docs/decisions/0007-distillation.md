# 0007 — Distillation: bert-tiny student, KD loss, distilled-vs-scratch ablation

**Status:** Accepted · 2026-06-29

## Context

The DistilBERT teacher ([03](../03-finetuning.md)) ties the TF-IDF baseline (n-gram-friendly task).
Distillation is still the project's #1 learning goal and a prerequisite for INT8/ONNX
([06](../06-optimization.md)), so we do it properly and benchmark honestly.

## Decisions

1. **Student = `prajjwal1/bert-tiny`** (2 layers / 128 hidden / ~4.4M params; ~15× smaller than the
   66M teacher). Off-the-shelf, tokenizer-compatible, gives a real size/speed delta.
2. **KD loss** `L = α·CE(z_s, y) + (1−α)·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T))`, start `T=3, α=0.5`.
3. **Precompute teacher logits** once over train+val (teacher is frozen) and train the student
   against the cache — fast/cheap on Colab.
4. **Run the distilled-vs-from-scratch ablation** — the distillation *lift* is the headline result,
   independent of beating the baseline.
5. **Reuse `TransformerPredictor` + the eval harness**; temperature-scale the student.

## Alternatives rejected

- **Custom student architecture / BiLSTM / MLP** — more effort and loses the transformer→ONNX path.
- **KL-only or CE-only loss** — KL-only drifts from ground truth; CE-only isn't distillation.
- **Live teacher forward each step** — wasteful; the teacher is frozen, so cache its logits.

## Consequences

- `src/.../models/distill.py`, `configs/distill.yaml`, `scripts/distill.py`, a Colab distill cell,
  tests. No new deps. Output student feeds [06-optimization](../06-optimization.md).
