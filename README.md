# MPSL

**Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach**

[Paper](https://arxiv.org/abs/2502.06355) · [Project](https://nousphera.github.io/MPSL/) · [Experiment guide](docs/experiments.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

Lightweight client tokenizers and a shared server encoder support **image–text, audio–text and audio–video learning**. Clients compute their losses; the server combines their gradients. The repository includes MPSL, centralized and federated baselines, and early/late fusion implementations.

This is a maintained revision of the authors' research code. The implementation is a **single-process research simulator**: tensors and autograd connections model client/server exchanges. It does not implement a private distributed transport or establish a privacy guarantee.

## Start with a CPU check

Python 3.11–3.12. No datasets, checkpoints or GPU are needed for this check.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python -m mpsl.smoke --output runs/smoke
```

The smoke run trains **two clients, two synthetic modalities and both fusion paths**, using the real classification split trainer. It records losses, accuracy, tensor payload bytes and partition indices in `runs/smoke/metrics.json`. These are execution checks, not paper results. Use a new output directory to preserve earlier runs.

## Run research experiments

```bash
python -m pip install -e '.[experiments,test]'
python -m mpsl.main_mpsl --help
python -m mpsl.main_centralized --help
python -m mpsl.main_fl --help
```

Acquire the pretrained **Meta-Transformer** weights and datasets described in the [experiment guide](docs/experiments.md). Full multimodal training is substantially more expensive than the smoke check; the original experiments used an NVIDIA H100.

```bash
python -m mpsl.main_mpsl \
  --dataset coco-qa --dataset_split_type noniid --dirichlet_alpha 0.1 \
  --nr_of_clients 25 --batch_size 125 --nr_of_epochs 15 \
  --nr_of_last_encoder_blocks_to_finetune 6 --start_lr 1e-4 \
  --random_seed 2024 --save_file_name cocoqa-mpsl \
  --torch_data_dir data --pre_processors_cache_dir cache/preprocessors \
  --tokenizer_weights_cache_dir cache/tokenizers --model_weights_dir checkpoints
```

Create these data/cache/checkpoint directories first. Supply a unique save name. Set the server batch size to a multiple of the client count. `--dirichlet_alpha 0.1` follows the paper; `0.5` recovers the historical code's concentration. Dataset acquisition and access restrictions are documented per dataset.

## What is included

| Component | Location |
| --- | --- |
| Shared transformer and modality tokenizers | `src/mpsl/models/` |
| Split, centralized and federated trainers | `src/mpsl/trainers/` |
| Dataset loaders and seeded client partitions | `src/mpsl/available_datasets/` |
| Tensor communication accounting and aggregation | `src/mpsl/utils/` |
| Offline early/late fusion execution check | `src/mpsl/smoke.py` |
| Regression and gradient-equivalence checks | `tests/` |

The maintained release fixes partial-batch accuracy weighting in split and baseline trainers, shared experiment-history state, storage-based communication counts and non-contiguous participating-client IDs. Dataset/model imports are lazy, and the code installs under the `mpsl` namespace rather than generic top-level `models` or `utils` packages.

## Validation boundaries

Tests check split-gradient equivalence to a joint weighted batch, deterministic early/late fusion training, isolated metrics, dtype-aware payload sizes and partition coverage. Full published dataset results, accelerator performance and all retrieval benchmarks have **not** been rerun for this revision. Tensor byte counts exclude network framing and do not measure encrypted or authenticated traffic.

See [reproducibility notes](docs/REPRODUCIBILITY.md) for source provenance and the precise scope of validation. Dataset, model and third-party code terms continue to apply; the repository retains its MIT license.
