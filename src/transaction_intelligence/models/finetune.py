"""Fine-tune a transformer for subtype classification (see docs/03-finetuning.md).

Importable so the Colab notebook and the local smoke run the SAME code. Label space = the 40
subtypes (category derived). ``id2label`` is built from ``tx.SUBTYPES`` so output index i means
``tx.SUBTYPES[i]`` — which makes ``predict_proba`` already taxonomy-aligned (no reindex needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.eval import metrics

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ID = "distilbert-base-uncased"


class _DescriptorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _make_dataset(df, tokenizer, max_length):
    enc = tokenizer(df["descriptor"].tolist(), truncation=True, max_length=max_length)
    labels = [tx.SUBTYPE_TO_ID[s] for s in df["subtype"]]
    return _DescriptorDataset(enc, labels)


def _class_weights(df) -> torch.Tensor:
    counts = np.bincount([tx.SUBTYPE_TO_ID[s] for s in df["subtype"]], minlength=tx.NUM_SUBTYPES)
    inv = np.zeros(tx.NUM_SUBTYPES, dtype=float)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    inv *= tx.NUM_SUBTYPES / inv.sum()  # normalize to mean ~1
    return torch.tensor(inv, dtype=torch.float)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    y_true = [tx.SUBTYPES[i] for i in labels]
    y_pred = [tx.SUBTYPES[i] for i in preds]
    return {"macro_f1": metrics.macro_f1(y_true, y_pred, tx.SUBTYPES)}


class _WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (rare subtypes matter for macro-F1)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        cw = self._class_weights
        weight = cw.to(outputs.logits.device) if cw is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


class TransformerPredictor:
    """Fine-tuned transformer adapted to the harness Predictor interface (subtype-level)."""

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 48,
        batch_size: int = 64,
        device=None,
        temperature: float = 1.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature  # post-hoc calibration scalar (1.0 = none)

    @torch.no_grad()
    def predict_proba(self, descriptors) -> np.ndarray:
        descriptors = list(descriptors)
        if not descriptors:
            return np.zeros((0, tx.NUM_SUBTYPES))
        out = []
        for i in range(0, len(descriptors), self.batch_size):
            batch = descriptors[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits / self.temperature  # temperature scaling
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)  # (n, NUM_SUBTYPES), aligned to tx.SUBTYPES via id2label

    def predict(self, descriptors) -> list[str]:
        return [tx.SUBTYPES[i] for i in self.predict_proba(descriptors).argmax(axis=1)]

    def save(self, path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "temperature.json").write_text(json.dumps({"temperature": self.temperature}))
        return path

    @classmethod
    def load(cls, path, max_length: int = 48) -> TransformerPredictor:
        path = Path(path)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tpath = path / "temperature.json"
        temperature = json.loads(tpath.read_text())["temperature"] if tpath.exists() else 1.0
        return cls(model, tok, max_length=max_length, temperature=temperature)


@torch.no_grad()
def _collect_logits(model, df, tokenizer, max_length, device, batch_size=64):
    descs = df["descriptor"].tolist()
    labels = torch.tensor([tx.SUBTYPE_TO_ID[s] for s in df["subtype"]])
    chunks = []
    for i in range(0, len(descs), batch_size):
        enc = tokenizer(
            descs[i : i + batch_size],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        chunks.append(model(**enc).logits.detach().cpu())
    return torch.cat(chunks), labels


def fit_temperature(model, val_df, tokenizer, max_length, device) -> float:
    """Fit a single temperature T on val by minimizing NLL of softmax(logits / T).

    Optimizes log(T) for positivity. Lowers ECE (over-confident logits get softened) without
    changing predictions (dividing by a positive scalar preserves argmax).
    """
    logits, labels = _collect_logits(model, val_df, tokenizer, max_length, device)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def train_transformer(
    train_df, val_df, config: dict, *, smoke: bool = False
) -> TransformerPredictor:
    """Fine-tune a transformer on (descriptor -> subtype) and return a TransformerPredictor."""
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    max_length = config.get("max_length", 48)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=tx.NUM_SUBTYPES,
        id2label=dict(enumerate(tx.SUBTYPES)),
        label2id=dict(tx.SUBTYPE_TO_ID),
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=config.get("output_dir", "models/teacher"),
        learning_rate=float(config.get("learning_rate", 3e-5)),
        per_device_train_batch_size=config.get("batch_size", 32),
        per_device_eval_batch_size=64,
        num_train_epochs=1 if smoke else config.get("epochs", 4),
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
        fp16=config.get("fp16", True) and torch.cuda.is_available(),  # DeBERTa-v3 needs fp16 off
    )

    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_make_dataset(train_df, tokenizer, max_length),
        eval_dataset=_make_dataset(val_df, tokenizer, max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        class_weights=_class_weights(train_df),
    )
    trainer.train()

    predictor = TransformerPredictor(model, tokenizer, max_length=max_length)
    predictor.temperature = fit_temperature(model, val_df, tokenizer, max_length, predictor.device)
    return predictor
