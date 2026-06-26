"""Tiny CPU smoke for the transformer fine-tuning pipeline + Predictor conformance.

Uses a tiny random model so it runs fast and offline-ish; this verifies the wiring (datasets,
weighted loss, taxonomy-aligned proba, save/load), NOT model quality.
"""

import pytest

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data.datasets import load_split
from transaction_intelligence.eval import Predictor

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transaction_intelligence.models.finetune import (  # noqa: E402
    TransformerPredictor,
    train_transformer,
)

TINY_MODEL = "hf-internal-testing/tiny-random-distilbert"


def test_smoke_train_predict_save_load(tmp_path):
    train = load_split("train").sample(96, random_state=0)
    val = load_split("val").sample(48, random_state=0)
    cfg = {
        "model_id": TINY_MODEL,
        "max_length": 32,
        "batch_size": 16,
        "output_dir": str(tmp_path / "ckpt"),
    }

    model = train_transformer(train, val, cfg, smoke=True)
    assert isinstance(model, Predictor)

    rows = train["descriptor"].head(8).tolist()
    proba = model.predict_proba(rows)
    assert proba.shape == (8, tx.NUM_SUBTYPES)
    assert [tx.SUBTYPES[i] for i in proba.argmax(1)] == model.predict(rows)

    saved = model.save(tmp_path / "saved")
    reloaded = TransformerPredictor.load(saved, max_length=32)
    assert reloaded.predict(rows[:3]) == model.predict(rows[:3])
