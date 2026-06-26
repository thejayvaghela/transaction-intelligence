# Transaction Intelligence — Documentation Index

Documentation is a **first-class deliverable**. Each functionality gets its own doc containing:
what it does, the concepts/theory learned, key decisions + rejected alternatives, how it's
implemented, how to run/test it, results/metrics, and gotchas.

**Doc lifecycle (how we work):** we *draft each doc as a design spec before building*, then
*finalize it with real results after*. Docs lead the build and also close it out.
Cross-reference, don't duplicate — link to other docs with relative links. The
[glossary](glossary.md) collects every ML term as we introduce it.

## Functionality docs

| #  | Doc | Summary | Status |
|----|-----|---------|--------|
| 00 | [00-foundations.md](00-foundations.md) | Problem framing, MCC codes, why descriptors are hard, ML-lifecycle map, label taxonomy | 🚧 next |
| 01 | 01-data-pipeline.md | Acquire + synthesize descriptors, weak labels, leakage-safe splits, EDA | ⬜ planned |
| 02 | 02-baselines.md | TF-IDF + logistic regression baseline + the **reusable eval harness** | ⬜ planned |
| 03 | 03-finetuning.md | Fine-tune DistilBERT / DeBERTa-v3 teacher; MLflow tracking | ⬜ planned |
| 04 | 04-ner.md | *(Optional, deferred)* token-level merchant/location NER | ⬜ deferred |
| 05 | 05-distillation.md | Distill teacher → tiny student | ⬜ planned |
| 06 | 06-optimization.md | ONNX export + INT8 quantization + accuracy/latency Pareto report | ⬜ planned |
| 07 | 07-serving.md | FastAPI + ONNX service (batch + online paths) | ⬜ planned |
| 08 | 08-frontend.md | Lean Next.js demo page | ⬜ planned |
| 09 | 09-mlops.md | MLflow registry + CI eval gate + Evidently drift monitoring | ⬜ planned |
| 10 | 10-deploy.md | Docker + one Fargate deploy + model card + filled-in resume bullet | ⬜ planned |

## Dependency map

```
00 foundations
   └─ 01 data ─ 02 baselines + eval-harness ─ 03 finetune ─┬─ 05 distill ─ 06 optimize ─ 07 serve ─ 08 frontend
                                                           └─ 09 mlops ─ 10 deploy
   (04 NER hangs off 03 — optional, deferred)
```

- The **eval harness** built in [02](02-baselines.md) is reused by 03 / 05 / 06 / 09.
- The **taxonomy** defined in [00](00-foundations.md) (`src/transaction_intelligence/taxonomy.py`)
  is the single source of truth everywhere downstream.

## Decisions (ADRs)

- [decisions/0001-dependency-manager.md](decisions/0001-dependency-manager.md) — `uv` over Poetry
- [decisions/0002-colab-local-split.md](decisions/0002-colab-local-split.md) — train on Colab, develop/serve locally
