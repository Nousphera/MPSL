# Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach

Multimodal transformers integrate diverse data types like images, audio, and text, advancing tasks such as audio-visual understanding and image-text retrieval; yet their high parameterization limits deployment on resource-constrained edge devices. Split Learning (SL), which partitions models at a designated cut-layer to offload compute-intensive operations to the server, offers a promising approach for distributed training of multimodal transformers, though its application remains underexplored. We present MPSL, a parallel SL approach for computational efficient fine-tuning of multimodal transformers in a distributed manner, while eliminating label sharing, client synchronization, and per-client sub-model management. MPSL employs lightweight client-side tokenizers and a unified modality-agnostic encoder, allowing flexible adaptation to task-specific needs. Our evaluation across 7 multimodal datasets demonstrates that MPSL matches or outperforms Federated Learning, reduces client-side computations by 250x, and achieves superior scalability in communication cost with model growth. Through extensive analysis, we highlight task suitability, trade-offs, and scenarios where MPSL excels, inspiring further exploration.

[[paper](https://arxiv.org/pdf/2502.06355)]

## Environment
We conduct our experiments with Python 3.11.3 using a single NVIDIA H100 GPU.

Create a new Python environment (using e.g. virtualenv or anaconda) and install all required packages via:
```console
foo@bar:~$ pip install -r requirements.txt
```

## Model
Throughout our experiments, we use pre-trained weights from [Meta-Transformer](https://github.com/invictus717/MetaTransformer) based on ViT-B/16 (Meta-Transformer's base scale). In order to execute our experiments, ensure the model weights have been downloaded.

## Datasets
The following datasets are used in our experiments:

- [COCO-QA](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/)
- [Flickr30K](https://huggingface.co/datasets/nlphuji/flickr30k)
- [Kinetics-Sounds](https://github.com/weiguoPian/AV-CIL_ICCV2023)
- [MELD](https://affective-meld.github.io/)
- [MS-COCO](https://cocodataset.org/)
- [T4SA](http://www.t4sa.it/)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

All datasets, except for T4SA and Kinetics-Sounds, are downloaded automatically when the experiments are initialized. For instructions on how to include these two datasets, please refer to the sections on [T4SA](#t4sa) and [Kinetics-Sounds](#kinetics-sounds) below.

### T4SA
Access to the T4SA dataset requires authentication. Please visit [the official website](http://www.t4sa.it/) to request access and download the dataset. We use the **B-T4SA** version in our experiments.

Once downloaded, place the dataset in a sub-folder named `b-t4sa`, in the datasets parent folder.

### Kinetics-Sounds
For including the Kinetics-Sounds dataset, we follow the instructions in [this](https://github.com/weiguoPian/AV-CIL_ICCV2023) repository. Since Kinetics-Sounds is a subset of Kinetics-400, the original data has to be downloaded using the [official](https://github.com/cvdfoundation/kinetics-dataset) repository.
To ensure the data is in the expected format for our codebase, please refer to the steps below.

1. Download and extract the Kinetics-400 dataset into your `datasets` folder following the instructions in the [official](https://github.com/cvdfoundation/kinetics-dataset) repository.
2. Execute `python arrange_by_classes.py k400` to restructure the raw Kinetics-400 data. The python file is included in the official repository.
3. Optional: If you wish to minimize used space, delete the `k400_targz` folder by executing `rm -r k400_targz`.
4. Optional: The Kinetics-Sounds data that we use for our experiments is roughly 24 GB worth of videos, whereas Kinetics-400 in its entirety is roughly 441 GB. Upon initialization, the codebase will copy the relevant video files into a `kinetics-sounds` folder in your `dataset` folder. Once finished, the `kinetics-dataset` folder that contains all Kinetics-400 data can be deleted. 

## Executing experiments
To run any of the experiments, execute one of the following scripts located in the `src` folder:

- `main_mpsl.py`: Runs fine-tuning using MPSL — **our proposed framework**.
- `main_centralized.py`: Runs centralized fine-tuning.
- `main_fl.py`: Runs fine-tuning using Federated Learning. 
  - To reproduce the adapter-based approach from [FedCLIP](https://github.com/microsoft/PersonalizedFL), use `main_fl.py` with the `--use_adapter_approach` flag set to `True`.

Each script supports a set of configurable parameters passed as command-line arguments. To view all available options for a specific script, use the `--help` flag. For example:
```console
foo@bar:~$ python -u main_mpsl.py --help
```

### Example usage
Please refer to the examples below on how to execute experiments with the COCO-QA dataset.

**Fine-tuning the model on the COCO-QA dataset using our proposed framework (MPSL):**
```console
foo@bar:~$ python -u main_mpsl.py --model meta_transformer \
                                  --dataset coco-qa \
                                  --dataset_split_type noniid \
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
foo@bar:~$ python -u main_centralized.py --model meta_transformer \
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
foo@bar:~$ python -u main_fl.py --model meta_transformer \
                                --dataset coco-qa \
                                --dataset_split_type noniid \
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
foo@bar:~$ python -u main_fl.py --use_adapter_approach True \
                                --include_image_adapter True \
                                --include_text_adapter True \
                                --model meta_transformer \
                                --dataset coco-qa \
                                --dataset_split_type noniid \
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
To calculate the computation overhead, refer to `src/calculate_computation_overhead.py`. The computation overhead is initially calculated for the centralized models. From this, we deduce the overhead for the Split Learning counterpart by reasoning about which layers are executed on the client-side and server-side.

Below is an example of how to calculate the computation overhead for the model used with the COCO-QA dataset:

```console
foo@bar:~$ python -u calculate_computation_overhead.py --dataset coco-qa \
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
