<h1 align="center">MPSL</h1>
<p align="center"><strong>Fine-Tuning Multimodal Transformers on Edge:<br>A Parallel Split Learning Approach</strong></p>
<p align="center">Timo Fudala · <a href="https://vtsouval.github.io/"><strong>Vasileios Tsouvalas</strong></a> · Nirvana Meratnia</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.06355" title="Paper"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/paper-dark.svg"><img src="docs/assets/icons/paper.svg" width="34" height="34" alt="Paper"></picture></a>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/separator-dark.svg"><img src="docs/assets/icons/separator.svg" width="24" height="34" alt=""></picture>
  <a href="https://nousphera.github.io/MPSL/" title="Project"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/project-dark.svg"><img src="docs/assets/icons/project.svg" width="34" height="34" alt="Project"></picture></a>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/separator-dark.svg"><img src="docs/assets/icons/separator.svg" width="24" height="34" alt=""></picture>
  <a href="https://nousphera.github.io/assets/posters/mpsl-a0-landscape.pdf" title="Poster"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/poster-dark.svg"><img src="docs/assets/icons/poster.svg" width="34" height="34" alt="Poster"></picture></a>
</p>

## Abstract

**Multimodal Parallel Split Learning (MPSL)** moves transformer fine-tuning from resource-constrained clients to a *shared server*. Clients retain *lightweight modality tokenizers* and send representations for *parallel server processing*. Local objectives drive a **single, sample-weighted server update**, with gradients returned to clients.

MPSL supports *early and late fusion* across **vision, language and audio**, covering question answering, sentiment and emotion recognition, action recognition, and image–text retrieval. It **substantially reduces client computation**, with *task-dependent accuracy trade-offs*.

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

```bash
git clone https://github.com/Nousphera/MPSL.git
cd MPSL
bash scripts/setup.sh
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Data and weights

Download [COCO-QA annotations](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/) and [COCO 2014 images](https://cocodataset.org/#download), then prepare the data:

```bash
python scripts/prepare_data.py --dataset coco-qa --source raw --root data
python scripts/prepare_data.py --dataset coco-qa --root data --validate
mkdir -p checkpoints
```

<details>
<summary><strong>Datasets</strong></summary>

| Dataset / source | Folder in `raw/` |
| :--- | :--- |
| [`coco-qa`](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/) · [images](https://cocodataset.org/#download) | `coco-qa/`<br>`ms-coco/` |
| [`coco-retrieval`](https://cocodataset.org/#download) · [annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/) | `ms-coco/`<br>`karpathy/` |
| [`flickr30k`](https://shannon.cs.illinois.edu/DenotationGraph/) · [annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/) | `flickr30k-images/`<br>`karpathy/` |
| [`t4sa`](http://www.t4sa.it/) | `b-t4sa/` |
| [`ucf101`](https://www.crcv.ucf.edu/data/UCF101.php) | `UCF-101/` |
| [`kinetics-sounds`](https://github.com/cvdfoundation/kinetics-dataset) | `k400/` |
| [`meld`](https://affective-meld.github.io/) | `MELD.Raw/` |

</details>

Place the [Meta-Transformer base encoder](https://github.com/invictus717/MetaTransformer#-model-zoo) at `checkpoints/Meta-Transformer_base_patch16_encoder.pth`.

> [!NOTE]
> Image/text and AST audio tokenizers load on first use.

## Train and evaluate

```bash
python -m mpsl.main_mpsl \
  --dataset coco-qa --dataset_split_type noniid --dirichlet_alpha 0.1 \
  --nr_of_clients 100 --batch_size 500 --nr_of_epochs 15 \
  --nr_of_last_encoder_blocks_to_finetune 6 \
  --num_workers 0 --test_num_workers 0 \
  --save_file_name cocoqa-mpsl
```

> [!NOTE]
> Metrics and weights are saved to `checkpoints/cocoqa-mpsl.{json,pth}`.

<details>
<summary><strong>Baselines</strong></summary>

| Method | Module |
| :--- | :--- |
| Centralized | `mpsl.main_centralized` |
| FedAvg | `mpsl.main_fl` |
| FedCLIP | `mpsl.main_fl` + adapters |

Omit `--test_num_workers` for all baselines. For centralized training, also omit `--nr_of_clients`, `--dataset_split_type` and `--dirichlet_alpha`.

FedCLIP adds:

```bash
--use_adapter_approach True \
--include_image_adapter True \
--include_text_adapter True
```

</details>

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
