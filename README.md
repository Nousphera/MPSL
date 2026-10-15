<h1 align="center">MPSL</h1>
<p align="center"><strong>Fine-Tuning Multimodal Transformers on Edge:<br>A Parallel Split Learning Approach</strong></p>
<p align="center">Timo Fudala · <a href="https://vtsouval.github.io/"><strong>Vasileios Tsouvalas</strong></a> · Nirvana Meratnia</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.06355" title="Read the paper"><img src="docs/assets/icons/resource-paper.svg" width="48" height="48" alt="Paper"></a>
  <picture><img src="docs/assets/icons/resource-separator.svg" width="20" height="48" alt=""></picture>
  <a href="https://nousphera.github.io/MPSL/" title="Explore the project"><img src="docs/assets/icons/resource-project.svg" width="48" height="48" alt="Project"></a>
  <picture><img src="docs/assets/icons/resource-separator.svg" width="20" height="48" alt=""></picture>
  <a href="https://nousphera.github.io/assets/posters/mpsl-a0-landscape.pdf" title="View the poster"><img src="docs/assets/icons/resource-poster.svg" width="48" height="48" alt="Poster"></a>
</p>

## Abstract

**Multimodal Parallel Split Learning (MPSL)** moves transformer fine-tuning from resource-constrained clients to a *shared server*. Clients retain *lightweight modality tokenizers* and send representations for *parallel server processing*. Local objectives drive a **single, sample-weighted server update**, with gradients returned to clients. MPSL supports *early and late fusion* across **vision, language and audio**, covering question answering, sentiment and emotion recognition, action recognition, and image–text retrieval. It **substantially reduces client computation**, with *task-dependent accuracy trade-offs*.

<p align="center">
  <a href="docs/assets/readme/method-light.mp4" title="Watch the method video">
    <picture>
      <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="docs/assets/readme/method-dark-mobile.gif">
      <source media="(max-width: 600px)" srcset="docs/assets/readme/method-light-mobile.gif">
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/method-dark.gif">
      <img src="docs/assets/readme/method-light.gif" alt="MPSL: client tokenizers send representations to a shared server; local losses drive one server update and gradients return to clients." width="960">
    </picture>
  </a>
</p>

## Setup

Create a `.venv` and install all required packages:

```bash
git clone https://github.com/Nousphera/MPSL.git
cd MPSL
bash scripts/setup.sh
source .venv/bin/activate
```

For an existing environment, install the requirements with:

```bash
python -m pip install -r requirements.txt
```

## Data and weights

Download, extract and validate a dataset, including the pretrained base encoder:

```bash
# One dataset
python scripts/prepare_data.py --dataset coco-qa

# All datasets
python scripts/prepare_data.py
```

**Datasets:** `coco-qa` · `coco-retrieval` · `flickr30k` · `t4sa` · `ucf101` · `kinetics-sounds` · `meld`.

For datasets requiring access, download the following files before running the same command:

| Dataset | Download and place in `raw/` |
| :--- | :--- |
| [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) | Images at `flickr30k-images/<image>.jpg`, or the archive as `flickr30k-images.tar.gz`. Annotations download automatically. |
| [T4SA](http://www.t4sa.it/) | `b-t4sa_imgs.tar` and `raw_tweets_text.csv`. Alternatively, extract under `b-t4sa/` with `data/`, `b-t4sa_train.txt`, `b-t4sa_test.txt` and the CSV. |

Sources stay in `raw/`, manifests in `data/`, and the [Meta-Transformer base encoder](https://github.com/invictus717/MetaTransformer#-model-zoo) in `checkpoints/`. Rerun to resume interrupted downloads or check prepared data.

> [!NOTE]
> Kinetics-Sounds scans several hundred GB of official K400 archives, retaining only the paper's clips. Image/text and AST audio tokenizers load on first use.

## Run Experiments

From the repository root, set the shared options and run your chosen method:

```bash
# Shared COCO-QA settings
common=(--dataset coco-qa --batch_size 500 --nr_of_epochs 15
        --nr_of_last_encoder_blocks_to_finetune 6 --num_workers 0)
clients=(--nr_of_clients 100 --dataset_split_type noniid --dirichlet_alpha 0.1)

# MPSL (ours)
python -m mpsl.main_mpsl "${common[@]}" "${clients[@]}" \
  --test_num_workers 0 --save_file_name cocoqa-mpsl

# Centralized
python -m mpsl.main_centralized "${common[@]}" \
  --save_file_name cocoqa-centralized

# FedAvg
python -m mpsl.main_fl "${common[@]}" "${clients[@]}" \
  --save_file_name cocoqa-fedavg

# FedCLIP
python -m mpsl.main_fl "${common[@]}" "${clients[@]}" \
  --use_adapter_approach True \
  --include_image_adapter True --include_text_adapter True \
  --save_file_name cocoqa-fedclip
```

> [!NOTE]
> Metrics and weights are saved to `checkpoints/<save_file_name>.{json,pth}`.

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

[MIT License](LICENSE)
