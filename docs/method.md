# Method and communication

## One parallel split-learning update

1. **Clients tokenize locally.** Each client retains its tokenizer graph and sends detached cut activations.
2. **The server predicts.** A shared encoder processes the representations using early or late fusion and returns detached predictions.
3. **Clients compute losses.** Each sends `(n_client / n_batch) × dLoss/dPrediction` and scalar loss telemetry.
4. **The server backpropagates once.** Sample-weighted gradients update the shared encoder; cut gradients return to trainable client tokenizers.

Retrieval keeps client-local negatives and sums caption losses with the same sample weighting, including incomplete batches. At evaluation, client tokenizers are aggregated and paired with the shared encoder.

## Communication and deployment

The runner is a **single-process simulator** with explicit detached tensor boundaries; tests also exercise serialization. Counters include activations, predictions, prediction gradients, scalar losses and cut gradients, measured as tensor elements × dtype size. Serialization headers, control messages and network framing are excluded.

Paper §3.2 describes a scalar client-loss upload. Across a detached boundary, the server also needs the prediction gradient; this implementation includes those bytes. Its counters therefore differ in scope from scalar-only paper figures.

Labels remain in client loss calculations, but prediction gradients can reveal them. Local labels alone do not provide a privacy guarantee. Distributed deployment requires a network transport and its own privacy controls.

## Models and evaluation

- **Paper configuration:** pretrained Meta-Transformer ViT-B/16, with the last six encoder blocks trained for classification and all twelve for retrieval.
- **Published size ablations:** official ViT-Ti/16, ViT-S/16, ViT-L/14 and ViT-H/14 architectures. Text uses OpenCLIP and audio uses AST patch embeddings.
- **Initialization:** pretrained loading errors propagate. The explicit Python option `pretrained=False` initializes the same architecture from scratch for offline checks.
- **Classification:** deterministic evaluation transforms, example-weighted loss and accuracy.
- **Retrieval:** image/caption groups are preserved; recall is reported in both directions.

COCO-QA uses frequency-ordered answers with alphabetical ties; MELD retains its numeric emotion map. Check label order before reusing a checkpoint. Action audio windows align with video frames. MELD's `3.88s` compatibility key selects a 2.56-second clip padded to 766 filterbank frames. See [dataset details](../DATASETS.md).

## Run records

Research JSON files retain the configuration, seed, prepared-manifest hash, class map, split counts, metric history and communication counters. Keep source manifests and checkpoint hashes with each run. Tests cover serialized gradient equivalence, sample weighting, partition coverage, data validation and checkpoint round trips; they do not replace full dataset experiments.
