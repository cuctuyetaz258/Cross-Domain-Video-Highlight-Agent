# V2 Implementation Checklist

## Before coding

- [ ] Reconcile local `main` with `origin/main` without overwriting existing
      user changes.
- [ ] Inventory available media, transcripts, labels, and caches.
- [ ] Freeze and commit the `v2_10video` manifest/splits plus fingerprints.
- [ ] Rerun and archive the V1 baseline on precisely those splits.

## Model and training

- [ ] Implement a dedicated `TemporalConvLTRScorer`; do not overload the V1
      MLP class or mislabel it as attention.
- [ ] Add non-causal residual TCN blocks with the V2 configuration in
      [`design.md`](design.md).
- [ ] Add sequence-per-video batching, valid-length masking, and
      within-video pair sampling.
- [ ] Keep V1 training behavior reproducible behind a model selection flag.
- [ ] Implement V2 pretraining on TVSum + SumMe and save architecture-specific
      metadata/checkpoint version.
- [ ] Add unit tests for tensor shapes, padding behavior, no cross-video
      context, checkpoint rejection, and deterministic seeded sampling.

## Evaluation and integration

- [ ] Train all five custom folds; use validation-only checkpoint selection.
- [ ] Produce raw metrics, histories, curves, hashes, parameter counts, and a
      consolidated V2 report.
- [ ] Review failures and apply the promotion gate unchanged.
- [ ] Leave the MLP UI default unchanged unless V2 passes.
- [ ] If promoted, train a separately marked all-data operational checkpoint,
      then add a controlled UI/default update and smoke test.

## Non-goals for this version

- [ ] Do not add SBERT, CLIP, CLAP, Transformer attention, keyframe filtering,
      semantic deduplication, or LLM fusion experiments to V2.

## Current developer entry points

```bash
# Verify the existing five folds and their schema-1.1 caches are unchanged.
python scripts/lock_v2_manifest.py

# Before pretraining, validate that all benchmark records have cached inputs.
python scripts/validate_ltr_cache_manifest.py \
  --manifest data/manifests/tvsum_summe.jsonl --split train

# Pretrain the V2 architecture on the benchmark manifest, then use that V2
# checkpoint as --init-checkpoint for each custom fold. This requires a
# schema-compatible cache for every TVSum and SumMe record first.
python -m highlight_agent.models.train_tcn_ltr \
  --manifest data/manifests/tvsum_summe.jsonl \
  --cache-dir data/features_cache \
  --output data/models/tcn_ltr_pretrained_tvsum_summe.pt \
  --max-epochs 50 --patience 15 --lr 1e-4

# Fine-tune/evaluate one custom fold. Repeat only after the V2 pretrain run.
python -m highlight_agent.models.train_tcn_ltr \
  --manifest data/manifests/custom_fold0.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/tcn_ltr_pretrained_tvsum_summe.pt \
  --output data/models/tcn_ltr_custom_fold0.pt \
  --max-epochs 50 --patience 15 --lr 1e-4
python -m evaluation.evaluate_tcn_ltr \
  --manifest data/manifests/custom_fold0.jsonl \
  --cache-dir data/features_cache \
  --checkpoint data/models/tcn_ltr_custom_fold0.pt \
  --split test --output-json data/reports/tcn_ltr_custom_fold0_test.json
```

The first CPU smoke run is deliberately not a pretraining or evaluation result:
it uses a tiny pair budget and only verifies that cached feature inputs, TCN
training, checkpoint validation, and held-out scoring execute end to end.
