# Experiment guide

## Installation

The [setup script](../scripts/setup.sh) creates `.venv` and installs the single [requirements.txt](../requirements.txt). Override `PYTHON_BIN=python3.12` or `VENV_DIR=/path/to/venv` when needed. Python 3.11 and 3.12 are supported.

For a manual installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0
fi
python -m pip install -r requirements.txt
python -m pip check
```

For **CUDA on Linux**, select a compatible wheel index from the [PyTorch installer](https://pytorch.org/get-started/locally/) and pass it as `TORCH_INDEX_URL` to `scripts/setup.sh` with a fresh `VENV_DIR=.venv-cuda`. An existing CPU environment is not converted automatically. Torch, torchvision and torchaudio must use matching CPU/CUDA variants. The paper experiments used an NVIDIA H100; the offline check still requires memory for a full ViT-B/16. PyAV wheels handle media decoding.

## Checkpoints and caches

Acquire the encoder from the [Meta-Transformer model zoo](https://github.com/invictus717/MetaTransformer#-model-zoo). Place `Meta-Transformer_base_patch16_encoder.pth` in `checkpoints/`; the large variant uses `Meta-Transformer_large_patch14_encoder.pth`. These are encoder state dictionaries, not saved MPSL training runs.

| Component | Pretrained source |
| :--- | :--- |
| Image processor | `facebook/vit-mae-base` |
| Image tokenizer | OpenCLIP `ViT-B/16` · `laion2b_s34b_b88k` |
| Text features | OpenAI CLIP via OpenCLIP |
| Audio tokenizer | `MIT/ast-finetuned-audioset-10-10-0.4593` |

Tokenizers load on first use or from populated caches. Set `HF_HOME` for shared Hugging Face assets. `--use_clip_encoder_for_text_embeddings False` selects token IDs instead of pretrained text features and changes the experiment.

| CLI option | Default path |
| :--- | :--- |
| `--torch_data_dir` | `data` |
| `--pre_processors_cache_dir` | `cache/preprocessors` |
| `--tokenizer_weights_cache_dir` | `cache/tokenizers` |
| `--model_weights_dir` | `checkpoints` |

Paths are relative to the working directory and can be overridden. Retain the encoder SHA256 with each run. Acquire and prepare datasets separately using [DATASETS.md](../DATASETS.md).

## Training and baselines

The [README command](../README.md#train-and-evaluate) runs COCO-QA with 100 clients, a server batch of 500 and Dirichlet concentration 0.1. The default learning rate is `1e-4`, seed is `2024`, and training lasts 15 epochs.

Reuse that command's dataset, learning-rate, model and path options for comparisons:

| Run | Module | Changes to the MPSL command |
| :--- | :--- | :--- |
| **MPSL** | `mpsl.main_mpsl` | None |
| Centralized | `mpsl.main_centralized` | Omit `--nr_of_clients`, `--dataset_split_type`, `--dirichlet_alpha`, `--test_num_workers` |
| FedAvg | `mpsl.main_fl` | Change module and save name; omit `--test_num_workers` |
| FedCLIP | `mpsl.main_fl` | Omit `--test_num_workers`; add `--use_adapter_approach True --include_image_adapter True --include_text_adapter True` |

Use a unique `--save_file_name` per run and the same prepared data, class map, seed and client partitions across comparisons. All modules expose `--help`.

For action recognition, use `--audio_time_dimension 1s` or `10.24s`. MELD automatically selects its `3.88s` compatibility key; the actual clip and filterbank dimensions are documented in [DATASETS.md](../DATASETS.md#meld-meld).

## Outputs

Each epoch evaluates the prepared test split. Research runs save `<save_file_name>.json` and `.pth` to `--model_weights_dir`; JSON records metrics, configuration, data metadata and logical tensor communication. Inspect a run with:

```bash
python -m json.tool checkpoints/cocoqa-mpsl.json
```

The offline check writes `metrics.json` under its `--output` directory and refuses to overwrite an existing run. See [method details](method.md) for gradient exchange and counter definitions.

## Compute profiling

```bash
python -m mpsl.calculate_computation_overhead \
  --dataset coco-qa --save_file_name cocoqa-overhead \
  --nr_of_last_encoder_blocks_to_finetune 6
```

The profiler measures centralized model computation; client/server allocations follow the split boundary. Keep the same model, fusion and trainable-layer settings when comparing with the [published results](https://nousphera.github.io/MPSL/#findings).
