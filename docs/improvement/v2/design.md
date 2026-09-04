# V2 Design: TCN-LTR Over Window Tokens

## Scope

V2 adds temporal context to V1 while retaining the existing seven handcrafted
features and all post-scoring behavior. It does not introduce SBERT, CLIP,
CLAP, a Transformer, semantic deduplication, or LLM changes.

```text
feature timeline: 7 x T at 10 Hz
        -> mean pool each 5 s window, hop 1 s
        -> X = [x_1, ..., x_W], x_i in R^7
        -> non-causal TCN
        -> H = [h_1, ..., h_W], h_i in R^32
        -> scalar LTR head
        -> scores [s_1, ..., s_W]
        -> existing NMS / optional LLM / boundary refinement / render
```

The TCN operates on an entire video sequence. It must never receive a sequence
that concatenates different videos, since that would create invalid temporal
context and leakage.

## Proposed network

```text
Linear(7, 32) -> Tanh
-> four residual non-causal Conv1D blocks
   channels=32, kernel=3, dilations=[1, 2, 4, 8]
   activation=GELU, dropout=0.1
-> Linear(32, 1)
```

Symmetric padding gives each token both past and future context, which is
valid because highlight creation is offline. With kernel size 3 and dilations
1/2/4/8, the receptive field is approximately 31 window tokens, or roughly
31 seconds with the current one-second hop.

## Training behavior

- Train one complete chronological sequence per video; do not chunk unless a
  later scaling need makes it necessary.
- Form positive/negative pairs only within a video.
- Retain the comparable pairwise hinge margin, source weights, maximum pairs
  per video, and adjacent-score smoothness term (`lambda_smooth=0.01`) from
  V1 unless validation experiments justify a documented change.
- Pretrain a V2 TCN checkpoint on TVSum + SumMe, then initialize each custom
  fold from that V2 checkpoint. V1 MLP weights cannot be cleanly loaded into
  the new TCN architecture.

## Interfaces to preserve

- Feature schema `1.1`: `rms`, `pitch`, `silence`, `text_score`,
  `scene_change`, `gesture`, `turn_rate`.
- Input sampling/window contract: 10 Hz, 5-second window, 1-second hop.
- Existing checkpoint preflight and metadata validation, extended with an
  explicit `model_type: tcn_ltr_v2` field so MLP and TCN files cannot be
  confused.
- Existing NMS, optional LLM assessment, validator, and rendering behavior.

## Risks

The TCN has more capacity than V1 and may overfit a 10-video pilot. A metric
gain smaller than fold variance is not enough to promote it. Do not claim that
the TCN learns temporal patterns until the locked evaluation and qualitative
failure review support that statement.
