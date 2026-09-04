# Version Roadmap And Gates

## V2: Temporal context for the current seven features

**Question:** Does contextualizing the existing window sequence improve
held-out ranking beyond the V1 MLP, without making ordinal ranking worse?

V2 keeps the extractor, feature contract, labels, window geometry, NMS, and
optional LLM flow fixed. It replaces only the independent-window scorer with a
small non-causal TCN followed by the same pairwise LTR scoring task.

Promotion requires all of the following on the locked 10-video,
video-disjoint five-fold protocol:

1. Macro held-out AP improves by at least `+0.01` over a freshly rerun V1
   baseline on the identical splits and seeds.
2. Macro Kendall tau is not lower than V1.
3. AP improves on at least 6 of 10 held-out videos.
4. Fold variance and the train-validation gap are not materially worse.
5. Hyperparameters are selected using validation only, never held-out tests.

If any gate fails, retain V1 as the UI default and publish the negative result.
If V2 passes, train an all-data V2 operational checkpoint for demo only and
switch the default in a separate, reviewed change.

## V3: Add text semantics only after V2

```text
7 handcrafted features + SBERT text embedding -> TCN -> LTR head
```

SBERT supplies a dense representation of transcript content, whereas V1/V2
`text_score` is an interpretable TF-IDF density scalar. It is not a cosine
similarity retrieval step: this project has no user query to compare against.
V3 requires timestamped transcript chunks, cached embeddings, missing-text
masks, projection/normalization, and an ablation against V2.

## V4: Add visual semantics only after V3

```text
7 handcrafted features + SBERT + CLIP visual embedding -> TCN -> LTR head
```

V4 requires shot detection, deterministic keyframe selection, cached CLIP
embeddings, visual-validity masks, and a V3-vs-V4 ablation. `scene_change`
remains a distinct handcrafted transition signal; CLIP represents visual
content, not merely cuts. Do not add query-conditioned retrieval components
such as TRAKE: this is a query-free highlight-ranking task.

## Common safeguards

- Preserve video-disjoint splits and keep temporal context within one video.
- Version every cache/manifest/feature contract; never silently reuse an
  incompatible cache.
- Report mean +/- standard deviation, per-video metrics, parameter counts,
  hashes, configs, histories, and failure cases.
- Keep LLM assessment disabled for primary LTR metric evaluation.
