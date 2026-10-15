# Reproducibility and source correspondence

## Source

This revision derives from the authors' public MPSL source at commit `5fa2c84142102555d84448fd3dfcb0f756798b83`. Original history, dataset implementations, model variants and licensing remain available. Entry points moved into the installable `mpsl` package; historical environment pins are preserved in `requirements-original.txt`.

Reference: [arXiv:2502.06355](https://arxiv.org/abs/2502.06355). The paper evaluates paired multimodal supervised tasks with a shared server encoder. This release does not extend those results to unsupervised domain adaptation, arbitrary missing modalities, asynchronous workers or a real distributed privacy protocol.

## Method and execution

Client tokenizers generate intermediate activations. The server processes the concatenated batch (per modality for late fusion) and returns detached predictions. Each client computes its loss and sends a sample-count-weighted prediction gradient. The server performs one backward pass, then returns cut-layer gradients to each client's original graph. The simulator runs in one Python process, but no client loss graph is used by server backpropagation.

A dedicated regression checks that the resulting client and server updates equal a monolithic weighted-batch update. Both early and late fusion paths train in the offline check. Retrieval tests compare against the original client-local contrastive negatives and sum over captions, rather than silently substituting global negatives. Early/late fusion uses small synthetic networks there; full Meta-Transformer loading requires separately acquired weights.

### Explicit gradient exchange

Paper §3.2 and Eq. (4) describe clients sending local losses for a sample-weighted server objective. The original code passes live client loss graphs into server autograd while counting only a scalar loss upload. A scalar loss value alone cannot carry its derivative through a serialized boundary. This release corrects the executable exchange, while preserving the objective and one server backward pass:

1. A client retains its tokenizer graph and sends detached cut activations.
2. The server retains its own graph and sends detached predictions to each client.
3. For client batch size `n` and actual combined batch size `B`, the client returns `g = (n / B) * dL_client/dprediction`. A detached scalar loss is also sent for metrics only.
4. The server inserts each gradient into the matching batch slice and backpropagates once. It returns detached cut gradients for client updates.

For retrieval, both image and text output-embedding gradients are returned. Local contrastive negatives, caption summation and client sample weights are unchanged. The default transport clones detached tensors locally. Trainers accept a `transport` callable to exchange tensor values; tests serialize and deserialize every exchange using `torch.save`/`torch.load(weights_only=True)` and verify parameter updates and a single server backward. This is a boundary check, not a deployed multi-process or network service.

The counters include cut activations, predictions, prediction gradients, scalar loss telemetry, and cut gradients (when clients are trainable). They sum `numel * element_size`; they exclude serializer headers, shape/dtype metadata, control messages and network framing. Compared with the scalar-only source accounting, each classification client upload adds `n * number_of_classes * dtype_bytes` per participating step. Retrieval adds the size of its image and text output tensors. Published communication numbers are therefore not interchangeable with these corrected counters.

This protocol does **not** establish label privacy. In ordinary single-label cross-entropy, a prediction gradient can reveal the target label. The paper's loss-only privacy statement is not a guarantee provided by this implementation. No differential privacy, secure aggregation or cryptographic transport is added here.

## Corrections in this revision

- Count each tensor's logical elements and dtype, rather than its backing storage. A small view must not count the entire combined batch.
- Weight split, centralized and federated classification accuracy and loss by examples, including a short final batch.
- Keep metric histories on each experiment instance rather than class-level shared lists.
- Aggregate and restore checkpoints using actual participating client IDs; reject invalid weight sets.
- Partition retrieval images into exactly the requested client IDs with full unique coverage; the low-level partitioner permits empty partitions, while training loaders enforce their minimum size.
- Replace the implicit client loss graph with explicit weighted prediction-gradient uploads in classification and retrieval; count those tensor payloads.
- Load only the chosen dataset/model implementation and install within a dedicated namespace.
- Expose Dirichlet concentration explicitly. The paper reports 0.1; the original source hard-coded 0.5. The new default is 0.1 and historical behavior is available explicitly.

## Environments and records

The maintained stack uses Python 3.11/3.12, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.19 and OpenCLIP 3.3. The original experiment pins are archival, not the recommended install. OpenCLIP loads the same named OpenAI CLIP checkpoints; the tokenizer reproduces the original BPE IDs, context length and truncation. Frozen external text encoders are cached by architecture and device, outside the federated model state. Token-only runs do not download an unused encoder. PyAV decodes media because current torchvision no longer provides `read_video`; audio filterbanks retain torchaudio Kaldi processing. HTTPS certificate verification is preserved. NumPy 2 is used for compatibility with current PyArrow. The [resolved macOS CPU environment](../requirements-tested-macos-py311.txt) records the executed stack; Linux CPU CI explicitly installs the CPU PyTorch, torchvision and torchaudio wheels together. CUDA execution is not established by the CPU checks.

Save the complete command, dataset version/split, checkpoint hash, seed and dependency freeze with every experiment. The CPU smoke writes exact indices and metrics. Research loaders may need access to large datasets or provider credentials; those resources are not redistributed here.

## Validated versus unverified

Validated locally: all 77 package modules import with experiment dependencies; generated video/audio decode and Kaldi filterbanks execute; original CLIP token IDs and truncation match; package installation, regression tests, offline early/late fusion training and deterministic repeat, classification/retrieval update equivalence with detached and serialized exchanges, exact tensor payload accounting, checkpoint round trips, and dataset partition coverage. Full benchmark accuracy, multi-seed confidence intervals, H100 runtime, real network communication and formal privacy are not re-established by these checks.

The historical centralized/federated/retrieval experiment variants are retained. Their full dataset trajectories have not been rerun. Do not combine older paper numbers with new implementation timings as if they were measured together.

## Dataset preparation follow-up

All seven multimodal datasets now use a manifest produced by
`scripts/prepare_data.py`, with read-only references to acquired source files.
COCO images retain original names; retrieval follows Karpathy filepaths; Flickr
uses filenames instead of assuming image IDs equal an external dataset row index.
T4SA uses exact string IDs and per-preparation text, with no pickle/global cache.
UCF, Kinetics and MELD retain shipped sample selections and fixed global label maps;
all required media is checked. Constructor-side downloads, unsafe shell commands,
archive extraction, raw-data moves and cleanup were removed. The old Kinetics
last-file completeness test and UCF always-truthy path check are gone.

Evaluation uses deterministic video frames/crops and audio padding. Model heads
and retrieval evaluation sizes follow the prepared manifest, and research result
JSON records its hash and configuration. COCO-QA's closed answer vocabulary covers
both splits, with explicit alphabetical tie ordering. This may change label IDs
relative to old NumPy sorting; old checkpoints require label-map verification.
MELD's 2.56-second clip and 766-frame padded frontend are preserved explicitly,
including the historical misnomer `3.88s`. These corrections and the source-format
contracts are documented in [DATASETS.md](../DATASETS.md).

Generated image/text/audio/video fixtures pass through every importer and its
public dataset loader, then real classification and retrieval split trainers.
These checks use small trainable test models, not downloaded Meta-Transformer
weights. Full corpus acquisition and the published benchmark trajectories remain
unverified. Provider access is external; no restricted images, tweets or videos
are distributed.
