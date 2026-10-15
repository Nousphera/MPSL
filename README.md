# MPSL

**Fine-Tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach**

[Paper](https://arxiv.org/abs/2502.06355) · [Project](https://nousphera.github.io/MPSL/) · [Datasets](DATASETS.md) · [Experiments](docs/experiments.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Abstract

Multimodal transformers are expensive to fine-tune on resource-constrained edge
devices. MPSL distributes this work between lightweight client tokenizers and a
shared server encoder. Clients tokenize images, audio and text, while the server
processes their representations and supports early or late modality fusion.
Client objectives are combined into a sample-weighted update with one server
backward pass. The framework supports visual question answering, sentiment and
emotion recognition, action recognition, and image–text retrieval, with
centralized and federated baselines. See the [paper](https://arxiv.org/abs/2502.06355)
for the method and reported experiments.

This maintained implementation is a single-process research simulator with
explicit detached tensor exchanges. Correct server backpropagation requires
prediction-gradient uploads; these are included in its communication counters
and can reveal labels. It does not provide a privacy guarantee or reproduce the
paper's scalar-only communication accounting. The [method correction](docs/REPRODUCIBILITY.md#explicit-gradient-exchange)
explains the distinction.

## Setup

Use **Python 3.11 or 3.12** on macOS or Linux. The maintained CPU environment uses
PyTorch 2.14.1, torchvision 0.29.1, torchaudio 2.11.0, Transformers 5.19 and
OpenCLIP 3.3. Install Python's `venv` support (`python3.11-venv` on distributions
that package it separately). Media decoding uses PyAV wheels; no system FFmpeg
command is required on the tested macOS wheel installation.

```bash
git clone https://github.com/Nousphera/MPSL.git
cd MPSL
bash scripts/setup.sh
source .venv/bin/activate
python -m pytest
```

The setup script creates an isolated environment and installs all experiment and
test dependencies. Set `PYTHON_BIN=python3.12` or `VENV_DIR=/path/to/venv` to change
them. On Linux it explicitly selects the CPU PyTorch/vision/audio wheels together.
For CUDA, create a separate environment, install compatible wheels from the
[PyTorch installer](https://pytorch.org/get-started/locally/), then run
`python -m pip install -e '.[experiments,test]'`. CUDA execution is not revalidated
by the CPU tests. Avoid mixing CPU PyTorch with a CUDA-only torchaudio wheel.

For a smaller trainer-only environment, use
`SETUP_PROFILE=smoke bash scripts/setup.sh`. Optional media/model integration tests
will be skipped when those dependencies are absent. The
[resolved macOS Python 3.11 environment](requirements-tested-macos-py311.txt)
records the tested full installation; it is a platform-specific snapshot.

## Datasets

All seven paper datasets have a source importer and use the same prepared
manifest interface in the actual training loaders. Follow [DATASETS.md](DATASETS.md)
for provider access, exact extracted layouts and commands for COCO-QA, B-T4SA,
COCO retrieval, Flickr30K, UCF101's audio subset, Kinetics-Sounds and MELD.

```bash
python scripts/prepare_data.py --dataset coco-qa --source /data/raw --root data
python scripts/prepare_data.py --dataset coco-qa --root data --validate
```

Preparation references original files in place and records splits, labels,
annotation hashes and media sizes. `--hash-media` adds content hashes;
`--check-media` fully decodes audio/video. Keep the source files available.
Restricted or large datasets must be acquired explicitly from their providers.
No constructor downloads, renames or deletes them.

## Run

First exercise both fusion paths without datasets or checkpoints:

```bash
python -m mpsl.smoke --output runs/smoke
```

This trains two clients with two synthetic modalities through the real
classification split trainer. Use a new output directory for another run.

For research training, obtain the **Meta-Transformer base encoder** from its
[model zoo](https://github.com/invictus717/MetaTransformer#-model-zoo) and save it as
`checkpoints/Meta-Transformer_base_patch16_encoder.pth`. The
[experiment guide](docs/experiments.md) lists tokenizer checkpoints, cache paths,
audio options and baseline commands. Then run on prepared data:

```bash
mkdir -p cache/preprocessors cache/tokenizers checkpoints
python -m mpsl.main_mpsl \
  --dataset coco-qa --dataset_split_type noniid --dirichlet_alpha 0.1 \
  --nr_of_clients 25 --batch_size 125 --nr_of_epochs 15 \
  --nr_of_last_encoder_blocks_to_finetune 6 --start_lr 1e-4 \
  --random_seed 2024 --save_file_name cocoqa-mpsl \
  --torch_data_dir data --pre_processors_cache_dir cache/preprocessors \
  --tokenizer_weights_cache_dir cache/tokenizers --model_weights_dir checkpoints
```

The server batch size must be divisible by the client count, with enough examples
for each client. The paper's Dirichlet concentration is `0.1`; `0.5` selects the
historical code setting. `python -m mpsl.main_mpsl --help`,
`python -m mpsl.main_centralized --help` and `python -m mpsl.main_fl --help` expose
all options. Full pretrained experiments need substantially more memory and time
than the CPU check; the original experiments used an NVIDIA H100.

## Outputs and reproduction limits

The smoke writes `runs/smoke/metrics.json`, including losses, accuracy, payload
bytes and client indices. Research runs write `<save_file_name>.json` and
`<save_file_name>.pth` under `--model_weights_dir`. JSON includes configuration,
prepared-manifest hash, class map, split counts, metric history and communication
counters. Use a unique save name to preserve earlier results.

Tests cover source-format preparation for all seven datasets, unchanged raw files,
actual classification/retrieval split-training updates, deterministic media
evaluation, gradient equivalence across serialized tensor boundaries, partition
coverage and checkpoint round trips. Generated fixtures establish execution;
full benchmark accuracy, H100 performance and pretrained dataset trajectories have
not been rerun. Tensor byte counts exclude serialization/network framing.
MELD preserves the recovered 766-frame audio shape despite its misleading
historical `3.88s` option; [details and dataset conventions](DATASETS.md) stay
explicit. See [reproducibility notes](docs/REPRODUCIBILITY.md) for provenance.

## Citation

```bibtex
@misc{fudala2025finetuningmultimodaltransformersedge,
  title={Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach},
  author={Timo Fudala and Vasileios Tsouvalas and Nirvana Meratnia},
  year={2025},
  eprint={2502.06355},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2502.06355}
}
```

## License

[MIT](LICENSE). Dataset, pretrained model and third-party code terms apply
separately; those resources are not redistributed by this repository.
