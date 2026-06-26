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

_More to come as we go: tokenization, logits, transfer learning, distillation temperature, soft
labels, INT8 quantization, ONNX, calibration, drift / PSI, ..._
