"""MLflow tracking helpers (see docs/03-finetuning.md §7).

Configures a tracking URI (a SQLite file — on mounted Drive in Colab) and logs an EvalResult's
metrics. The model registry + CI promotion gate live in Phase 9; here we just get durable,
comparable run history from the first fine-tune onward.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import mlflow

DEFAULT_EXPERIMENT = "transaction-intelligence"
_METRIC_KEYS = (
    "subtype_macro_f1",
    "subtype_accuracy",
    "category_macro_f1",
    "category_accuracy",
    "ece",
)


def setup(tracking_uri: str | None = None, experiment: str = DEFAULT_EXPERIMENT) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def log_eval(result, prefix: str) -> None:
    """Log an EvalResult's metrics under a prefix (e.g. 'test', 'gold')."""
    d = result.to_dict()
    for k in _METRIC_KEYS:
        v = d.get(k)
        if v is not None:
            mlflow.log_metric(f"{prefix}_{k}", float(v))


@contextmanager
def run(params: dict | None = None, run_name: str | None = None):
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield
