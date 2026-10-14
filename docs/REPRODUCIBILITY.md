# Reproducibility and source correspondence

## Source

This revision derives from the authors' public MPSL source at commit `5fa2c84142102555d84448fd3dfcb0f756798b83`. Original history, dataset implementations, model variants and licensing remain available. Entry points moved into the installable `mpsl` package; historical environment pins are preserved in `requirements-original.txt`.

Reference: [arXiv:2502.06355](https://arxiv.org/abs/2502.06355). The paper evaluates paired multimodal supervised tasks with a shared server encoder. This release does not extend those results to unsupervised domain adaptation, arbitrary missing modalities, asynchronous workers or a real distributed privacy protocol.

## Method and execution

Client tokenizers generate intermediate activations. The server processes the concatenated batch (per modality for late fusion), returns predictions, and computes a sample-count-weighted aggregate of client cross-entropy losses. Cut-layer gradients are sent back to the corresponding client graph. The simulator retains labels and client loss graphs in the same Python process; it does not serialize an autograd graph across machines.

A dedicated regression checks that the resulting client and server updates equal a monolithic weighted-batch update. Both early and late fusion paths train in the offline check. Early/late fusion uses small synthetic networks there; full Meta-Transformer loading requires separately acquired weights.

## Corrections in this revision

- Count each tensor's logical elements and dtype, rather than its backing storage. A small view must not count the entire combined batch.
- Weight split, centralized and federated classification accuracy and loss by examples, including a short final batch.
- Keep metric histories on each experiment instance rather than class-level shared lists.
- Aggregate the actual participating client IDs; reject invalid weight sets.
- Load only the chosen dataset/model implementation and install within a dedicated namespace.
- Expose Dirichlet concentration explicitly. The paper reports 0.1; the original source hard-coded 0.5. The new default is 0.1 and historical behavior is available explicitly.

## Environments and records

The maintained stack uses Python 3.11/3.12, PyTorch 2.5.1 and torchvision 0.20.1. Full experiment extras pin compatible tokenizer/model/data dependencies; `setuptools` is pinned because OpenAI CLIP 1.0.1 still imports `pkg_resources`. The original pins are archival, not the recommended install.

Save the complete command, dataset version/split, checkpoint hash, seed and dependency freeze with every experiment. The CPU smoke writes exact indices and metrics. Research loaders may need access to large datasets or provider credentials; those resources are not redistributed here.

## Validated versus unverified

Validated locally: package installation, regression tests, offline early/late fusion training and deterministic repeat, split-gradient equivalence, and dataset partition coverage. Full benchmark accuracy, multi-seed confidence intervals, H100 runtime, real network communication and formal privacy are not re-established by these checks.

The historical centralized/federated/retrieval experiment variants are retained. Their full dataset trajectories have not been rerun. Do not combine older paper numbers with new implementation timings as if they were measured together.
