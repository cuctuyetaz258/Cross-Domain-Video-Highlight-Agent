# V4: CLIP Visual Semantics (Deferred)

V4 is planned only after V3 has a documented outcome. It adds cached CLIP
visual embeddings using deterministic shot/keyframe selection, projection,
normalization, and missing-visual masks before the temporal encoder.

`scene_change` should remain as a handcrafted cut/transition signal. CLIP
captures visual content and is not a replacement for scene-change detection.
V4 must be evaluated as a V3-vs-V4 ablation on the same locked protocol.
