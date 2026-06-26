# 0001 — Dependency management with `uv`

**Status:** Accepted · 2026-06-26

## Context

We need reproducible Python environments for a project that starts with a light dev phase and
later pulls in a heavy ML stack (PyTorch, Transformers, ONNX Runtime). Main candidates: `uv` and
Poetry.

## Decision

Use **`uv`** (Astral) for Python version pinning, env creation, dependency resolution, and running
commands, with a committed `uv.lock`.

## Why / consequences

- 10–100× faster installs than Poetry (Rust resolver) — matters when iterating on the ML stack.
- One tool covers `.python-version` pinning, venv creation, locking, and `uv run`.
- Standard `pyproject.toml` + PEP 735 dependency groups; no proprietary lockfile semantics.
- `uv sync` installs our `src/`-layout package in editable mode automatically.

## Alternatives rejected

- **Poetry** — mature and popular, but slower/heavier and historically quirky with PyTorch/CUDA
  index resolution. Maturity was its only real edge, and `uv` has closed that gap.
- **pip + venv + requirements.txt** — simplest, but no real lock/resolution and manual Python
  pinning. Fine for throwaway scripts, not a reproducible portfolio pipeline.
