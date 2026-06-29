# Glossary

Plain-English definitions of ML terms, added as each phase introduces them. Each entry notes
where the term first appears so you can jump to the full explanation.

- **Taxonomy** — the fixed schema of categories + subtypes we classify into; the single source of
  truth for every model. _(see [00-foundations](00-foundations.md))_
- **Descriptor string** — the short, noisy raw text of a transaction (e.g.
  `SQ *BLUE BOTTLE 8005551234 CA`) that we classify. _(see [00-foundations](00-foundations.md))_
- **Macro-F1** — F1 computed per class and averaged with **equal weight per class**, so rare
  categories count as much as common ones. Our headline metric. _(coming in [02-baselines](02-baselines.md))_
- **MCC (Merchant Category Code)** — a 4-digit ISO 18245 code card networks assign to a merchant
  (e.g. `5812` = restaurants, `5411` = supermarkets). A strong but leaky category signal; we use it
  for **weak labeling**, not as ground truth. _(see [00-foundations](00-foundations.md))_
- **Single-label multi-class classification** — each input is assigned **exactly one** class from a
  fixed set (here: one category per transaction). Contrast with multi-label (several at once).
  _(see [00-foundations](00-foundations.md))_
- **Abstention / confidence threshold** — letting the model **decline to commit** when its top
  probability is below a calibrated threshold, routing the case to "needs review" instead of an
  explicit `other` class. _(see [00-foundations](00-foundations.md))_
- **Synthetic data** — training examples we *generate* (here: descriptors from a gazetteer + noise
  model) rather than collect, giving abundant, perfectly-labeled, controllable data.
  _(see [01-data-pipeline](01-data-pipeline.md))_
- **Weak supervision / weak labeling** — labeling data cheaply with imperfect rules (MCC maps,
  keyword/gazetteer matching) instead of manual annotation. _(see [01-data-pipeline](01-data-pipeline.md))_
- **Data leakage** — when information from the test set sneaks into training (e.g. the same merchant
  in both), inflating metrics and hiding true generalization. _(see [01-data-pipeline](01-data-pipeline.md))_
- **Group split** — splitting train/val/test by a **group key** (here `merchant_id`) so a group
  never spans splits; the fix for merchant-memorization leakage. _(see [01-data-pipeline](01-data-pipeline.md))_
- **Stratification** — splitting so each class keeps the same proportion across train/val/test.
  _(see [01-data-pipeline](01-data-pipeline.md))_
- **Dataset card** — a document recording a dataset's provenance, generation params, distribution,
  and known limitations. _(see [01-data-pipeline](01-data-pipeline.md))_
- **Gazetteer** — a curated lookup of merchants → category/subtype (+ aliases, MCC) that seeds the
  synthesizer's ground truth. _(see [01-data-pipeline](01-data-pipeline.md))_
- **Synthetic-to-real gap** — the risk that a model learns the data *generator* instead of reality;
  measured by comparing performance on synthetic `test` vs the realistic `gold` set.
  _(see [01-data-pipeline](01-data-pipeline.md))_

- **TF-IDF** — Term Frequency × Inverse Document Frequency: turns text into sparse vectors where
  tokens common in one example but rare overall score high. _(see [02-baselines](02-baselines.md))_
- **n-gram (char vs word)** — contiguous runs of N tokens. *Word* n-grams use whole words; *char*
  n-grams use character sequences and survive truncation (`WHOLEFDS`). _(see [02-baselines](02-baselines.md))_
- **Logistic regression** — a linear classifier outputting class probabilities; a strong, fast text
  baseline. _(see [02-baselines](02-baselines.md))_
- **Precision / recall / F1** — precision = of predicted-X, how many were X; recall = of actual-X,
  how many we caught; F1 = their harmonic mean. _(see [02-baselines](02-baselines.md))_
- **Confusion matrix** — a grid of true vs predicted class counts; reveals which classes get mixed
  up. _(see [02-baselines](02-baselines.md))_
- **Calibration / ECE** — whether predicted confidence matches actual accuracy; Expected Calibration
  Error summarizes the gap. Gates the abstention threshold. _(see [02-baselines](02-baselines.md))_
- **Accuracy@coverage** — accuracy when you only answer on the most-confident fraction (coverage) of
  inputs; the fair way to compare against an abstaining baseline. _(see [02-baselines](02-baselines.md))_

- **Tokenization** — splitting text into the units a model consumes. _(see [03-finetuning](03-finetuning.md))_
- **Subword tokenization** — splitting words into frequent sub-word pieces (WordPiece/SentencePiece),
  so rare/garbled tokens (`WHOLEFDS`) become known pieces. _(see [03-finetuning](03-finetuning.md))_
- **Transfer learning / fine-tuning** — start from a model pretrained on huge text, then nudge its
  weights on your task with little data. _(see [03-finetuning](03-finetuning.md))_
- **Logits** — a model's raw, unnormalized output scores (pre-softmax). _(see [03-finetuning](03-finetuning.md))_
- **Softmax** — turns logits into a probability distribution over classes. _(see [03-finetuning](03-finetuning.md))_
- **Cross-entropy loss** — the standard classification loss: penalizes low probability on the true
  class. _(see [03-finetuning](03-finetuning.md))_
- **Learning rate / warmup** — step size for weight updates; warmup ramps it up early for stability.
  _(see [03-finetuning](03-finetuning.md))_
- **Epoch / batch size** — one epoch = one pass over the data; batch size = examples per update step.
  _(see [03-finetuning](03-finetuning.md))_
- **Temperature scaling** — a post-hoc calibration: divide logits by a scalar T (fit on val) to make
  confidence match accuracy (lower ECE). _(see [03-finetuning](03-finetuning.md))_
- **MLflow run / experiment** — a logged training run (params + metrics + artifacts) grouped under a
  named experiment. _(see [03-finetuning](03-finetuning.md))_

- **Knowledge distillation** — training a small "student" model to mimic a larger "teacher", so the
  student inherits the teacher's behavior at a fraction of the size. _(see [05-distillation](05-distillation.md))_
- **Soft targets / dark knowledge** — the teacher's full probability distribution (not just the top
  label); the relative probabilities encode learned class-similarity structure. _(see [05-distillation](05-distillation.md))_
- **Distillation temperature** — a `T` that softens the teacher's logits (`softmax(z/T)`) to expose
  that inter-class structure for the student to learn. _(see [05-distillation](05-distillation.md))_
- **KD loss** — distillation objective: `α·CE(hard) + (1−α)·T²·KL(soft student ‖ soft teacher)`.
  _(see [05-distillation](05-distillation.md))_
- **Distillation lift** — the macro-F1 gap between a distilled student and the same model trained
  from scratch on hard labels; isolates distillation's value. _(see [05-distillation](05-distillation.md))_

_More to come as we go: INT8 quantization, ONNX, latency/throughput, drift / PSI, ..._
