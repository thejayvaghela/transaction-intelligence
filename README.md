# Transaction Intelligence

A fine-tuned + distilled transformer that labels short, noisy bank/card **transaction
descriptor strings** (e.g. `SQ *BLUE BOTTLE 8005551234 CA`) into a **category + subtype +
confidence** in single-digit milliseconds on cheap CPU hardware.

> **Why own the model?** Calling a hosted LLM per transaction is too slow and far too expensive
> at millions of rows/day, and generic models misfire on domain-specific descriptors. So we own
> the weights end-to-end: **fine-tune → distill → quantize → self-host**.

## Status

🚧 **Phase 0 — Foundations & setup.** The build is documented phase by phase; start at
[`docs/README.md`](docs/README.md).

## Stack

Python 3.11 · `uv` · PyTorch + Hugging Face · ONNX Runtime · FastAPI · MLflow · Docker ·
Next.js (demo) · AWS Fargate + S3 · GitHub Actions

## Quickstart

```bash
uv sync          # create the environment (dev tooling for now)
uv run pytest    # run the test suite
```

## Documentation

Docs are a **first-class deliverable** — every functionality has its own design-then-results
doc, cross-linked, with a running [glossary](docs/glossary.md). Index: [docs/README.md](docs/README.md).
