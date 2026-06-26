"""MLflow tracking helpers (see docs/03-finetuning.md §7).

Tracking is **optional and non-blocking**: if mlflow isn't installed or fails, these become
no-ops so training/eval always proceed (the harness numbers print regardless). We use
``mlflow-skinny`` (lightweight client) + a **file store** (``./mlruns`` by default, or
``MLFLOW_TRACKING_URI``) — both resolve cleanly on Colab. The registry + CI promotion gate live
in Phase 9.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

_METRIC_KEYS = (
    "subtype_macro_f1",
    "subtype_accuracy",
    "category_macro_f1",
    "category_accuracy",
    "ece",
)
DEFAULT_EXPERIMENT = "transaction-intelligence"


def _mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def setup(tracking_uri: str | None = None, experiment: str = DEFAULT_EXPERIMENT) -> str | None:
    mlflow = _mlflow()
    if mlflow is None:
        print("[tracking] mlflow not available; continuing without experiment tracking")
        return None
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "file:./mlruns"
    # mlflow 3.x gates the (lightweight, skinny-compatible) file store behind this opt-out flag.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        return uri
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[tracking] mlflow setup failed ({exc}); continuing without tracking")
        return None


def log_eval(result, prefix: str) -> None:
    mlflow = _mlflow()
    if mlflow is None:
        return
    d = result.to_dict()
    for k in _METRIC_KEYS:
        v = d.get(k)
        if v is not None:
            try:
                mlflow.log_metric(f"{prefix}_{k}", float(v))
            except Exception:  # pragma: no cover
                pass


@contextmanager
def run(params: dict | None = None, run_name: str | None = None):
    mlflow = _mlflow()
    if mlflow is None:
        yield
        return
    try:
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            yield
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[tracking] mlflow run disabled ({exc}); continuing without tracking")
        yield
