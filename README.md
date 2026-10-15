<h1 align="center">MPSL</h1>
<p align="center"><strong>Fine-Tuning Multimodal Transformers on Edge:<br>A Parallel Split Learning Approach</strong></p>
<p align="center">Timo Fudala · <a href="https://vtsouval.github.io/"><strong>Vasileios Tsouvalas</strong></a> · Nirvana Meratnia</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.06355" title="Paper"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/paper-dark.svg"><img src="docs/assets/icons/paper.svg" width="34" height="34" alt="Paper"></picture></a>
  &nbsp; · &nbsp;
  <a href="https://nousphera.github.io/MPSL/" title="Project"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/project-dark.svg"><img src="docs/assets/icons/project.svg" width="34" height="34" alt="Project"></picture></a>
  &nbsp; · &nbsp;
  <a href="https://nousphera.github.io/assets/posters/mpsl-a0-landscape.pdf" title="Poster"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/icons/poster-dark.svg"><img src="docs/assets/icons/poster.svg" width="34" height="34" alt="Poster"></picture></a>
</p>

## Abstract

**Multimodal Parallel Split Learning (MPSL)** moves transformer fine-tuning from resource-constrained clients to a shared server. Clients retain lightweight modality tokenizers and send representations for **parallel server processing**. Local objectives drive a **single, sample-weighted server update**, with gradients returned to clients. MPSL supports early and late fusion across vision, language and audio, covering question answering, sentiment and emotion recognition, action recognition, and image–text retrieval. It substantially reduces client computation, with task-dependent accuracy trade-offs.

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
<summary><strong>Dataset sources and layouts</strong></summary>

Paths below are relative to `raw/`. Replace `coco-qa` in the commands with the dataset key.

| Dataset key | Source | Extracted layout |
| :--- | :--- | :--- |
| `coco-qa` | [Annotations](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/) · [Images](https://cocodataset.org/#download) | `coco-qa/{train,test}/{questions,img_ids,answers}.txt`; `ms-coco/{train2014,val2014}/*.jpg` |
| `coco-retrieval` | [COCO images](https://cocodataset.org/#download) · [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/) | `ms-coco/{train2014,val2014}/*.jpg`; `karpathy/dataset_coco.json` |
| `flickr30k` | [Images](https://shannon.cs.illinois.edu/DenotationGraph/) · [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/) | `flickr30k-images/*.jpg`; `karpathy/dataset_flickr30k.json` |
| `t4sa` | [B-T4SA](http://www.t4sa.it/) | `b-t4sa/b-t4sa_{train,test}.txt`; `b-t4sa/raw_tweets_text.csv` (`id,text`); `b-t4sa/data/` |
| `ucf101` | [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) | `UCF-101/<class>/*.avi`; shipped list selects the 51-class audio subset |
| `kinetics-sounds` | [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) | `k400/videos/<split>/<class>/*.mp4`; shipped list selects 30 classes |
| `meld` | [MELD raw media](https://affective-meld.github.io/) | `MELD.Raw/{train/train_splits,dev/dev_splits_complete,test/output_repeated_splits_test}/*.mp4` |

Preparation retains the original files in place. Keep `raw/` available. Add `--check-media` for full media decoding or `--hash-media` for content hashes. Retrieval uses the annotated train/test splits and five captions per image. MELD combines training/dev clips and preserves its numeric class map.

</details>

Place the [Meta-Transformer base encoder](https://github.com/invictus717/MetaTransformer#-model-zoo) at `checkpoints/Meta-Transformer_base_patch16_encoder.pth`. Image/text and AST audio tokenizers load on first use; caches are stored under `cache/` and the Hugging Face cache.

## Train and evaluate

```bash
python -m mpsl.main_mpsl \
  --dataset coco-qa --dataset_split_type noniid --dirichlet_alpha 0.1 \
  --nr_of_clients 100 --batch_size 500 --nr_of_epochs 15 \
  --nr_of_last_encoder_blocks_to_finetune 6 \
  --num_workers 0 --test_num_workers 0 \
  --save_file_name cocoqa-mpsl
```

Each epoch evaluates on the test split. Metrics and weights are saved to `checkpoints/cocoqa-mpsl.{json,pth}`. Use a distinct save name per run and a server batch size divisible by the client count.

<details>
<summary><strong>Baselines and configuration</strong></summary>

Adapt the command above:

| Method | Module | Changes |
| :--- | :--- | :--- |
| Centralized | `mpsl.main_centralized` | Omit `--nr_of_clients`, `--dataset_split_type`, `--dirichlet_alpha`, `--test_num_workers` |
| FedAvg | `mpsl.main_fl` | Omit `--test_num_workers` |
| FedCLIP | `mpsl.main_fl` | Omit `--test_num_workers`; add `--use_adapter_approach True --include_image_adapter True --include_text_adapter True` |

Keep data, seed and partitions fixed across comparisons. The default learning rate is `1e-4` and seed is `2024`. Retrieval trains all encoder blocks with `--nr_of_last_encoder_blocks_to_finetune -1`.

Override paths with `--torch_data_dir`, `--model_weights_dir`, `--pre_processors_cache_dir` and `--tokenizer_weights_cache_dir`. All modules expose `--help`.

Action audio supports `--audio_time_dimension 1s` or `10.24s`. MELD's default compatibility key `3.88s` selects a 2.56-second clip padded to 766 filterbank frames.

</details>

**[Published results →](https://nousphera.github.io/MPSL/#findings)**

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
