# V3: SBERT Text Semantics (Deferred)

V3 is planned only after V2 has a documented outcome. It adds timestamped,
cached SBERT text embeddings to the seven handcrafted features before the TCN
and LTR head. This is an embedding input for a query-free highlight scorer, not
cosine-similarity retrieval.

Prerequisites: V2 result ledger, transcript alignment policy, missing-text
masks, embedding cache/versioning, projection and normalization design, and a
V2-vs-V3 ablation on locked video-disjoint folds.
