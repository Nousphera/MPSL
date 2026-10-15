# Method and execution protocol

MPSL fine-tunes multimodal transformers using client-side tokenizers and a shared server encoder. Early and late fusion support paired multimodal classification and image–text retrieval. The implementation runs in one process with explicit detached tensor exchanges between client and server computations.

Reference: [Fine-Tuning Multimodal Transformers on Edge](https://arxiv.org/abs/2502.06355).

## Explicit gradient exchange

1. Each client retains its tokenizer graph and sends detached cut activations.
2. The server retains its encoder graph and returns detached predictions.
3. A client with batch size `n` sends `g = (n / B) * dL_client/dprediction`, where `B` is the combined batch size. Scalar loss telemetry is sent for metrics.
4. The server assembles gradients in batch order, performs one backward pass, and returns cut gradients to the trainable client tokenizers.

For retrieval, clients return gradients for both image and text embeddings. Contrastive negatives remain client-local; losses sum over captions and use the same client sample weights as classification. Short batches retain their actual sample counts.

Trainers accept a `transport` callable. The default clones detached tensors locally; serialization tests exchange values through `torch.save`/`torch.load(weights_only=True)` and compare parameter updates with a joint weighted-batch computation.

### Communication accounting

Counters include cut activations, predictions, prediction gradients, scalar loss telemetry and cut gradients. They sum logical tensor elements multiplied by dtype size, excluding serializer headers, metadata, control messages and network framing.

Paper §3.2 describes a scalar client-loss upload. Across a detached boundary, backpropagation also requires its prediction gradient. The implementation counts this additional upload: `n × number_of_classes × dtype_bytes` per classification client step, or both output-embedding tensors for retrieval. Use these counters consistently when comparing runs; they have a different scope from scalar-only communication figures.

Labels are used in each client's loss calculation. Prediction gradients can reveal labels, so label-local computation is not a privacy guarantee. Network deployment requires a separate transport and privacy design.

## Data and model configuration

- **Prepared manifests:** all seven multimodal datasets use read-only references to acquired source files. Results retain the manifest hash, class map and split counts. See [DATASETS.md](../DATASETS.md) for provider access and input layouts.
- **Partitions:** `--dirichlet_alpha 0.1` selects the paper's class concentration; use `0.5` to compare a different concentration. Retrieval partitioning uses the experiment seed and keeps complete image/caption groups together.
- **Evaluation:** deterministic image/video frames and audio padding, with example-weighted classification loss and accuracy. Model heads and retrieval evaluation size follow the manifest.
- **Label maps:** COCO-QA uses frequency-ranked answers with alphabetical ties; MELD uses the shipped numeric emotion map. Match a checkpoint's label order before reuse.
- **Audio:** action-recognition windows stay aligned with the selected frame. MELD uses a 2.56-second clip padded to 766 filterbank frames; its compatibility option is named `3.88s`. The dataset guide specifies this input contract.

The [experiment guide](experiments.md) lists Meta-Transformer, image, text and audio checkpoints, cache paths and baseline commands. External text encoders are cached by architecture and device outside federated model state. Token-ID input skips encoder loading. OpenCLIP supplies the CLIP tokenizer and pretrained text encoder; PyAV decodes media and torchaudio computes Kaldi filterbanks.

## Environment and saved records

Follow [Setup](../README.md#requirements) for Python 3.11/3.12 and the pinned dependency stack. Linux CPU installation selects PyTorch, torchvision and torchaudio CPU wheels together. For CUDA, install a matching set of wheels in a separate environment. The [macOS environment snapshot](../requirements-tested-macos-py311.txt) records exact package versions.

Save the complete command, source manifests, pretrained checkpoint hashes, seed and dependency freeze with each run. Research outputs include configuration, metric history, communication counters and checkpoints. Choose unique save names.

## Tests

```bash
python -m pytest
python -m mpsl.smoke --output runs/smoke
```

The suite checks all seven source importers and loaders, media decoding, tokenization, deterministic evaluation, partition coverage, checkpoint round trips, tensor-byte accounting and classification/retrieval updates across serialized boundaries. Small trainable fixtures exercise early and late fusion without downloading model checkpoints or full datasets.

## Attribution

Based on the authors' [MPSL source at `5fa2c84`](https://github.com/Nousphera/MPSL/tree/5fa2c84142102555d84448fd3dfcb0f756798b83). Original authorship, history and the [MIT license](../LICENSE) are retained. Third-party models, datasets and code retain their own terms.
