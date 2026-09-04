# Improvement Roadmap

This folder records planned architecture upgrades without changing the current
operational pipeline. It distinguishes an implemented result from a proposed
experiment so that presentation and engineering claims remain traceable.

## Current baseline (V1)

```text
7 handcrafted features at 10 Hz
-> 5 s mean-pooled windows with 1 s hop
-> Linear(7, 32) -> Tanh -> Linear(32, 1)
-> score per window -> NMS -> optional LLM rerank -> render
```

V1 is a seven-feature, window-independent MLP scorer trained with pairwise
LTR loss. Its adjacent-score smoothness loss is not a temporal encoder. The
class name `AdditiveAttentionScorer` is historical: the model has no temporal
attention.

The controlled 10-video in-domain pilot is documented in
[`../in_domain_ltr_5fold_report.md`](../in_domain_ltr_5fold_report.md). It is
not evidence of cross-domain generalization.

## Vocabulary

- **LTR:** a learning objective that orders higher-importance windows above
  lower-importance ones; it is not a particular neural architecture.
- **TCN:** a temporal convolutional encoder that lets each window use nearby
  and longer-range video context before the LTR head scores it.
- **Operational checkpoint:** the all-data checkpoint used by the UI. It is
  separate from held-out evaluation checkpoints.
- **Promotion:** replacing the UI's current MLP default only after the V2 gate
  in [`roadmap.md`](roadmap.md) is met.

## Status

| Version | Scope | Status |
| --- | --- | --- |
| V1 | 7 handcrafted features + MLP LTR | Implemented; current UI default |
| V2 | 7 handcrafted features + non-causal TCN + LTR | Planned experiment |
| V3 | V2 plus SBERT text embeddings | Deferred pending V2 |
| V4 | V3 plus CLIP visual embeddings | Deferred pending V3 |

Only V2 is authorized for the next implementation cycle. V3 and V4 are
roadmap items, not implemented functionality.
