# 05 — Knowledge Distillation (teacher → tiny student)

**Status:** 🚧 design spec (awaiting review → then build) · **Phase:** 5
**Depends on:** [03-finetuning](03-finetuning.md) (the DistilBERT teacher), [02-baselines](02-baselines.md)
(the harness) · **Used by:** [06-optimization](06-optimization.md) (quantize the student),
[07-serving](07-serving.md)

> Sections 1–7 are the plan; §8 (results) is filled after we build.

---

## 1. What this functionality is

Train a **tiny student** to mimic the **DistilBERT teacher** from [03](03-finetuning.md), and quantify
**accuracy retained vs. size/speed gained**, on the same harness. The headline lesson is the
**ablation**: distilled-student vs. same-size-student-trained-from-scratch — that isolates what
distillation actually buys.

**Honest framing:** the teacher ties the baseline ([03 §12](03-finetuning.md)), so the student will
too — for *this* task the baseline wins the product Pareto. Phase 5 is therefore a **skills +
benchmark** deliverable: you learn distillation properly and complete the accuracy/latency/cost
picture. Distillation also *requires* a neural net to be meaningful (a prerequisite for INT8/ONNX in
[06](06-optimization.md)).

## 2. Why distillation — the "why"

A big model's **soft predictions carry more than the hard label**. When the teacher says
`coffee 0.7, fast_food 0.2, restaurant 0.08, …`, those relative probabilities encode *learned
similarity structure* ("coffee is closer to fast_food than to wire_transfer") — Hinton's **"dark
knowledge."** A small student trained to match that full distribution learns that structure and
typically beats the same small model trained only on one-hot labels. That gap is the point of the
phase.

## 3. Concepts you'll learn

Knowledge distillation; **soft targets / dark knowledge**; **distillation temperature** (softening
logits to expose inter-class structure); the **KD loss** (a blend of KL-divergence on soft targets +
cross-entropy on hard labels, with the `T²` correction); student-architecture choices; and the
**distillation lift** (distilled vs. from-scratch).

## 4. Approach

### 4.1 Student model
**`prajjwal1/bert-tiny`** — 2 layers, hidden 128, ~4.4M params (vs DistilBERT's 66M → **~15× smaller**).
Uses bert-base WordPiece (tokenizer-compatible). Off-the-shelf, clean, gives a real size/speed delta
to measure. *Rejected:* hand-built custom architectures (more work, no benefit); BiLSTM/MLP (loses
the transformer-distillation lesson and the ONNX path).

### 4.2 Distillation loss
For student logits `z_s`, teacher logits `z_t`, temperature `T`, weight `α`:

```
L = α · CE(z_s, hard_label)  +  (1 − α) · T² · KL( softmax(z_s/T) ‖ softmax(z_t/T) )
```
- The **soft term** (KL on T-softened distributions) transfers dark knowledge; the **`T²`** factor
  keeps gradient magnitudes comparable as `T` changes.
- The **hard term** (plain CE) anchors the student to ground truth.
- Starting point: `T = 3`, `α = 0.5` (tuned on val, logged to MLflow).

### 4.3 Precomputed teacher logits (efficiency)
Run the teacher **once** over train+val, cache its logits (aligned to `tx.SUBTYPES`), then train the
student against the cache — no teacher forward pass per step (much faster/cheaper on Colab). The
teacher checkpoint comes from [03](03-finetuning.md) (`models/teacher`).

### 4.4 The ablation (the real lesson)
Train bert-tiny **twice**: (a) with the KD loss (distilled), (b) with hard labels only (from
scratch), same everything else. **Distilled − from-scratch macro-F1 = the distillation lift.** This
is what we actually want to demonstrate, regardless of the baseline.

### 4.5 Calibration + harness reuse
The student is wrapped in the existing `TransformerPredictor` ([03](03-finetuning.md)) — it's
model-agnostic, so the **same `evaluate(...)`** scores it on test + gold. Temperature-scale the
student on val for trustworthy confidence.

## 5. Planned module layout (built after approval)
```
src/transaction_intelligence/models/distill.py   # precompute_teacher_logits, KD loss, train_student
configs/distill.yaml                              # student_id, T, alpha, lr, epochs, seed
scripts/distill.py                                # CLI: load teacher -> distill -> eval (test+gold)
notebooks/03_finetune_colab.ipynb                 # add a distillation cell (after the teacher cell)
tests/test_distill.py                             # KD-loss math (toy) + tiny-model distill smoke
```
Reuses `TransformerPredictor` + the eval harness; no new heavy deps (torch/transformers already in
the `train` extra).

## 6. How to run / test (target)
```bash
# local: tiny smoke (CPU)
uv run pytest tests/test_distill.py
# Colab: train teacher (cell 4) -> then distill cell:
#   python scripts/distill.py            # distilled student
#   python scripts/distill.py --no-kd    # from-scratch ablation
```

## 7. Key decisions + rejected alternatives
| Decision | Why | Rejected |
|---|---|---|
| `bert-tiny` student | ~15× smaller, off-the-shelf, real delta to measure | custom arch (effort), BiLSTM/MLP (loses ONNX path) |
| KD loss = α·CE + (1−α)·T²·KL | standard, blends ground truth + dark knowledge | KL-only (drifts from labels), CE-only (no distillation) |
| precompute teacher logits | fast/cheap; teacher frozen anyway | live teacher forward each step (slow) |
| distilled-vs-from-scratch ablation | isolates distillation's actual value | reporting distilled alone (no causal claim) |
| reuse TransformerPredictor + harness | apples-to-apples vs teacher/baseline | bespoke eval (incomparable) |

## 8. Results / metrics
_TBD after build: student macro-F1 (test+gold) vs teacher vs baseline; size (66M→4.4M) and CPU-latency
deltas; **% accuracy retained**; **distillation lift** (distilled − from-scratch); calibrated ECE._

## 9. Gotchas (to confirm after build)
- **Label-order contract** again — student `id2label` from `tx.SUBTYPES`.
- **Logit alignment** — teacher-logit cache must be in `tx.SUBTYPES` order before computing KL.
- **`T²` factor** — forgetting it makes the soft-loss gradient scale with `T` and destabilizes tuning.
- Student may merely match the baseline — expected; the *lift* and the *pipeline* are the deliverables.

## 10. Glossary terms introduced
[knowledge distillation](glossary.md), [soft targets / dark knowledge](glossary.md),
[distillation temperature](glossary.md), [KD loss](glossary.md), [distillation lift](glossary.md).
