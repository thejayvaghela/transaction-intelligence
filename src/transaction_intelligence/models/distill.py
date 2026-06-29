"""Knowledge distillation: DistilBERT teacher -> tiny student (see docs/05-distillation.md).

Trains a small student to match the (frozen) teacher's softened logits + the hard labels. The
teacher's logits are precomputed once over the train set and carried through the batch. Reuses the
fine-tuning predictor + calibration + eval harness so the student is scored apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from transaction_intelligence import taxonomy as tx

from .finetune import TransformerPredictor, _compute_metrics, _make_dataset, fit_temperature

STUDENT_ID = "prajjwal1/bert-tiny"


def kd_loss(z_s, z_t, labels, temperature: float, alpha: float):
    """KD loss = alpha*CE(student, hard) + (1-alpha)*T^2*KL(soft student || soft teacher)."""
    ce = F.cross_entropy(z_s, labels)
    if alpha >= 1.0:
        return ce
    kl = F.kl_div(
        F.log_softmax(z_s / temperature, dim=1),
        F.softmax(z_t / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    return alpha * ce + (1.0 - alpha) * kl


class _KDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, teacher_logits):
        self.encodings = encodings
        self.labels = labels
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["teacher_logits"] = self.teacher_logits[idx]
        return item


class _KDCollator:
    """Pads tokenizer fields via the standard collator; stacks teacher_logits if present."""

    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        tl = None
        if "teacher_logits" in features[0]:
            tl = torch.tensor(
                np.array([f.pop("teacher_logits") for f in features]), dtype=torch.float
            )
        batch = self.base(features)
        if tl is not None:
            batch["teacher_logits"] = tl
        return batch


class _DistillTrainer(Trainer):
    def __init__(self, *args, temperature=3.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_T = temperature
        self.kd_alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits = inputs.pop("teacher_logits", None)  # absent during eval
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        z_s = outputs.logits.float()
        if teacher_logits is None:
            loss = F.cross_entropy(z_s, labels)
        else:
            z_t = teacher_logits.float().to(z_s.device)
            loss = kd_loss(z_s, z_t, labels, self.kd_T, self.kd_alpha)
        return (loss, outputs) if return_outputs else loss


def train_student(train_df, val_df, teacher: TransformerPredictor, config: dict, *, smoke=False):
    """Distill `teacher` into a tiny student; returns a calibrated TransformerPredictor."""
    student_id = config.get("student_id", STUDENT_ID)
    max_length = config.get("max_length", 48)

    tokenizer = AutoTokenizer.from_pretrained(student_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_id,
        num_labels=tx.NUM_SUBTYPES,
        id2label=dict(enumerate(tx.SUBTYPES)),
        label2id=dict(tx.SUBTYPE_TO_ID),
        ignore_mismatched_sizes=True,
    ).float()

    teacher_logits = teacher.predict_logits(train_df["descriptor"].tolist())  # (n, 40), tx order
    enc = tokenizer(train_df["descriptor"].tolist(), truncation=True, max_length=max_length)
    labels = [tx.SUBTYPE_TO_ID[s] for s in train_df["subtype"]]
    train_ds = _KDDataset(enc, labels, teacher_logits)
    val_ds = _make_dataset(val_df, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=config.get("output_dir", "models/student"),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        per_device_train_batch_size=config.get("batch_size", 64),
        per_device_eval_batch_size=64,
        num_train_epochs=1 if smoke else config.get("epochs", 6),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.get("seed", 20260626),
        max_steps=2 if smoke else -1,
        logging_steps=50,
        report_to=[],
        fp16=config.get("fp16", True) and torch.cuda.is_available(),
    )

    trainer = _DistillTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_KDCollator(tokenizer),
        compute_metrics=_compute_metrics,
        temperature=config.get("temperature", 3.0),
        alpha=config.get("alpha", 0.5),
    )
    trainer.train()

    predictor = TransformerPredictor(model, tokenizer, max_length=max_length)
    predictor.temperature = fit_temperature(model, val_df, tokenizer, max_length, predictor.device)
    return predictor
