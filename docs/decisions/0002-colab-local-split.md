# 0002 — Train on Colab, develop / serve locally

**Status:** Accepted · 2026-06-26

## Context

GPU training runs on **Google Colab** (free/Pro). Colab sessions are ephemeral — local disk is
wiped and runtimes cap at ~12h. The local machine (Apple Silicon-class Mac) handles data work,
serving, the frontend, and git.

## Decision

- The `src/` package is **pip-installable**; Colab notebooks `git clone` the repo and
  `pip install -e .`, then call the **same** training functions used locally — no notebook ↔ repo
  code drift.
- **Artifacts escape Colab:** live checkpoints → mounted Google Drive; promoted models + final
  benchmark artifacts → **S3** (which doubles as the Phase 10 artifact store).
- **MLflow without a server:** `MLFLOW_TRACKING_URI` → a SQLite file on mounted Drive; artifact
  store → S3. Persists across Colab sessions and is browsable from the local machine.

## Why / consequences

- Reproducibility: training is a versioned function call, not copy-pasted notebook cells.
- Durability: nothing important lives only on Colab's ephemeral disk.
- The S3 + SQLite-MLflow setup becomes the real registry backbone reused in Phase 09.

## Alternatives rejected

- **Hosted MLflow tracking server** — overkill and adds cost for a solo project.
- **Logging to Colab local disk** — runs vanish when the session ends.
- **Training locally on Apple MPS** — workable for DistilBERT, but slower and ties up the dev
  machine; Colab gives faster iteration and access to bigger free GPUs.
