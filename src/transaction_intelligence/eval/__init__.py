"""Evaluation — the reusable, model-agnostic harness reused by every model (see docs/02)."""

from .harness import EvalResult, HeadResult, Predictor, evaluate

__all__ = ["EvalResult", "HeadResult", "Predictor", "evaluate"]
