"""Tests for distillation: KD-loss math (golden) + a tiny end-to-end distill smoke."""

import pytest
import torch
import torch.nn.functional as F

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data.datasets import load_split
from transaction_intelligence.eval import Predictor

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transaction_intelligence.models.distill import kd_loss, train_student  # noqa: E402
from transaction_intelligence.models.finetune import train_transformer  # noqa: E402

TINY = "hf-internal-testing/tiny-random-distilbert"


def test_kd_loss_zero_kl_when_student_matches_teacher():
    z = torch.randn(4, tx.NUM_SUBTYPES)
    labels = torch.tensor([0, 1, 2, 3])
    expected = 0.5 * F.cross_entropy(z, labels)  # KL(z||z)=0 -> loss = alpha*CE
    got = kd_loss(z, z.clone(), labels, temperature=3.0, alpha=0.5).item()
    assert abs(got - expected.item()) < 1e-4


def test_kd_loss_alpha_one_is_pure_ce():
    z_s, z_t = torch.randn(4, tx.NUM_SUBTYPES), torch.randn(4, tx.NUM_SUBTYPES)
    labels = torch.tensor([0, 1, 2, 3])
    got = kd_loss(z_s, z_t, labels, 3.0, 1.0).item()
    assert abs(got - F.cross_entropy(z_s, labels).item()) < 1e-6


def test_distill_smoke(tmp_path):
    train = load_split("train").sample(64, random_state=0)
    val = load_split("val").sample(32, random_state=0)
    teacher = train_transformer(
        train, val, {"model_id": TINY, "max_length": 32, "batch_size": 16}, smoke=True
    )
    student = train_student(
        train,
        val,
        teacher,
        {"student_id": TINY, "max_length": 32, "batch_size": 16, "temperature": 3.0, "alpha": 0.5},
        smoke=True,
    )
    assert isinstance(student, Predictor)
    proba = student.predict_proba(train["descriptor"].head(5).tolist())
    assert proba.shape == (5, tx.NUM_SUBTYPES)
