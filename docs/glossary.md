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

_More to come as we go: tokenization, logits, transfer learning, distillation temperature, soft
labels, INT8 quantization, ONNX, calibration, drift / PSI, ..._
