# Dataset and experiment guide

## Environment
The original experiments used Python 3.11.3 and a single NVIDIA H100 GPU. For the maintained Python 3.11/3.12 environment, follow [Setup](../README.md#setup), including the Linux CPU audio wheel selection.

Create a new Python environment (using e.g. virtualenv or anaconda) and install all required packages via:
```console
python -m pip install -e '.[experiments]'
```

## Model
Throughout our experiments, we use pre-trained weights from [Meta-Transformer](https://github.com/invictus717/MetaTransformer) based on ViT-B/16 (Meta-Transformer's base scale). In order to execute our experiments, ensure the model weights have been downloaded.

## Datasets

Use the source importers documented in [DATASETS.md](../DATASETS.md). All seven
paper datasets require explicit acquisition, preparation and validation before
training. Set `--torch_data_dir` to the prepared root containing the dataset
manifest directories. Constructors no longer download, rename, move or delete
source files. The guide specifies provider access, exact source layouts,
recovered versus maintained split conventions, and deterministic evaluation.

## Checkpoints and caches

Obtain `Meta-Transformer_base_patch16_encoder.pth` from the upstream
[Meta-Transformer model zoo](https://github.com/invictus717/MetaTransformer#-model-zoo)
and place it directly in `--model_weights_dir`. The optional large encoder uses
`Meta-Transformer_large_patch14_encoder.pth` in that directory. These are encoder
state dictionaries, not full training checkpoints. Record the checkpoint SHA256
with your run; it is not bundled or downloaded by the setup script.

The default image processor loads `facebook/vit-mae-base`; the image tokenizer
loads OpenCLIP `ViT-B/16` with `laion2b_s34b_b88k` weights. Default text embeddings
load the named OpenAI CLIP encoder through OpenCLIP. Audio tokenizers load
`MIT/ast-finetuned-audioset-10-10-0.4593`. These initial model loads require network
access or a populated Hugging Face/OpenCLIP cache, separate from dataset access.
`--pre_processors_cache_dir` and `--tokenizer_weights_cache_dir` control the
explicit processor/image-tokenizer caches; `HF_HOME` controls other Hugging Face
caches. `--use_clip_encoder_for_text_embeddings False` selects token IDs instead
of pretrained text features and changes the experiment.

For action recognition use `--audio_time_dimension 1s` or `10.24s`. MELD selects
the historical `3.88s` key by default; its actual recovered frontend produces
766 filterbank frames as explained in [DATASETS.md](../DATASETS.md#meld-meld).

## Executing experiments
To run any of the experiments, execute one of the following installed module entry points:

- `main_mpsl.py`: Runs fine-tuning using MPSL — **our proposed framework**.
- `main_centralized.py`: Runs centralized fine-tuning.
- `main_fl.py`: Runs fine-tuning using Federated Learning.
  - To reproduce the adapter-based approach from [FedCLIP](https://github.com/microsoft/PersonalizedFL), use `main_fl.py` with the `--use_adapter_approach` flag set to `True`.

Each script supports a set of configurable parameters passed as command-line arguments. To view all available options for a specific script, use the `--help` flag. For example:
```console
python -u -m mpsl.main_mpsl --help
```

### Example usage
Please refer to the examples below on how to execute experiments with the COCO-QA dataset.

**Fine-tuning the model on the COCO-QA dataset using our proposed framework (MPSL):**
```console
python -u -m mpsl.main_mpsl --model meta_transformer \
                                  --dataset coco-qa \
                                  --dataset_split_type noniid --dirichlet_alpha 0.1 \
                                  --nr_of_clients 100 \
                                  --batch_size 500 \
                                  --nr_of_epochs 15 \
                                  --nr_of_last_encoder_blocks_to_finetune 6 \
                                  --start_lr 1e-4 \
                                  --random_seed 2024 \
                                  --save_file_name my_trained_model \
                                  --torch_data_dir <path_to_datasets_folder> \
                                  --pre_processors_cache_dir <path_to_preprocessors_cache_folder> \
                                  --tokenizer_weights_cache_dir <path_to_tokenizer_weights_folder> \
                                  --model_weights_dir <path_to_model_weights_folder>
```

**Centralized fine-tuning:**
```console
# Centralized
python -u -m mpsl.main_centralized --model meta_transformer \
                                         --dataset coco-qa \
                                         --batch_size 500 \
                                         --nr_of_epochs 15 \
                                         --nr_of_last_encoder_blocks_to_finetune 6 \
                                         --start_lr 1e-4 \
                                         --random_seed 2024 \
                                         --save_file_name my_trained_model \
                                         --torch_data_dir <path_to_datasets_folder> \
                                         --pre_processors_cache_dir <path_to_preprocessors_cache_folder> \
                                         --tokenizer_weights_cache_dir <path_to_tokenizer_weights_folder> \
                                         --model_weights_dir <path_to_model_weights_folder>
```

**Federated Learning:**
```console
python -u -m mpsl.main_fl --model meta_transformer \
                                --dataset coco-qa \
                                --dataset_split_type noniid --dirichlet_alpha 0.1 \
                                --nr_of_clients 100 \
                                --batch_size 500 \
                                --nr_of_epochs 15 \
                                --nr_of_last_encoder_blocks_to_finetune 6 \
                                --start_lr 1e-4 \
                                --random_seed 2024 \
                                --save_file_name my_trained_model \
                                --torch_data_dir <path_to_datasets_folder> \
                                --pre_processors_cache_dir <path_to_preprocessors_cache_folder> \
                                --tokenizer_weights_cache_dir <path_to_tokenizer_weights_folder> \
                                --model_weights_dir <path_to_model_weights_folder>
```

**Federated Learning + Adapter (FedCLIP) approach:**
```console
python -u -m mpsl.main_fl --use_adapter_approach True \
                                --include_image_adapter True \
                                --include_text_adapter True \
                                --model meta_transformer \
                                --dataset coco-qa \
                                --dataset_split_type noniid --dirichlet_alpha 0.1 \
                                --nr_of_clients 100 \
                                --batch_size 500 \
                                --nr_of_epochs 15 \
                                --nr_of_last_encoder_blocks_to_finetune 6 \
                                --start_lr 1e-4 \
                                --random_seed 2024 \
                                --save_file_name my_trained_model \
                                --torch_data_dir <path_to_datasets_folder> \
                                --pre_processors_cache_dir <path_to_preprocessors_cache_folder> \
                                --tokenizer_weights_cache_dir <path_to_tokenizer_weights_folder> \
                                --model_weights_dir <path_to_model_weights_folder>
```

### Calculating computation overhead
To calculate the computation overhead, refer to `src/mpsl/calculate_computation_overhead.py`. The computation overhead is initially calculated for the centralized models. From this, we deduce the overhead for the Split Learning counterpart by reasoning about which layers are executed on the client-side and server-side.

Below is an example of how to calculate the computation overhead for the model used with the COCO-QA dataset:

```console
python -u -m mpsl.calculate_computation_overhead --dataset coco-qa \
                                                       --save_file_name cocoqa-overhead \
                                                       --model meta_transformer \
                                                       --use_pre_layer_norm True \
                                                       --use_post_layer_norm False \
                                                       --nr_of_last_encoder_blocks_to_finetune 6 \
                                                       --torch_data_dir <path_to_datasets_folder> \
                                                       --pre_processors_cache_dir <path_to_preprocessors_cache_folder> \
                                                       --tokenizer_weights_cache_dir <path_to_tokenizer_weights_folder> \
                                                       --model_weights_dir <path_to_model_weights_folder>
```

## Citations
If you use this repository or find our work helpful, please consider citing:

<pre>@misc{fudala2025finetuningmultimodaltransformersedge,
      title={Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach},
      author={Timo Fudala and Vasileios Tsouvalas and Nirvana Meratnia},
      year={2025},
      eprint={2502.06355},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2502.06355},
}
</pre>
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/MPSL/) · [Research overview](https://nousphera.github.io/projects/mpsl/) · [A0 poster](https://nousphera.github.io/MPSL/poster.pdf) · [Paper](https://arxiv.org/abs/2502.06355)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
