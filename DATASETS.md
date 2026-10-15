# Dataset acquisition and preparation

The seven paper datasets use **prepared manifests consumed by the normal MPSL,
centralized and federated loaders**. Preparation references acquired source files
in place: it never renames images, moves videos, deletes archives, or downloads a
corpus implicitly. Keep the source directory and annotation files at their saved
absolute paths. Prepare again under a new output root after relocating them.

## Common commands

Install the experiment environment with `bash scripts/setup.sh`, then activate
`.venv`. Acquire and extract the provider's data below into `raw/` or another
location with enough space. Respect each provider's access and usage terms.

```bash
python scripts/prepare_data.py --help
python scripts/prepare_data.py --dataset coco-qa --source /data/raw --root data
python scripts/prepare_data.py --dataset coco-qa --root data --validate
```

The output is `data/coco-qa/manifest.json`. Pass **`--torch_data_dir data`** to the
trainer, not `data/coco-qa` and not `/data/raw`. The same convention applies to
all seven names below. Train/test must both be nonempty; preparation rejects
missing records, duplicate media within a split (except repeated COCO-QA questions
about an image), overlapping media/identities between splits, inconsistent labels,
path traversal and symlinks that escape the source root. Nothing is silently skipped.

Every referenced image is fully decoded. Every video is opened and its first frame
and audio/video stream rates checked. Add `--check-media` to fully decode every
video and audio stream; this can take substantial time. Annotation SHA256 hashes,
media sizes, sample identities, class maps and counts are recorded. Add
`--hash-media` during preparation to hash every media byte; validation then checks
those hashes too. The default size checks do not detect same-size content changes.
The training loaders check all files in their split and retain the manifest SHA256
in the experiment result. Run `--validate --check-media` before expensive training.

## COCO-QA (`coco-qa`)

Acquire the [COCO-QA annotation ZIP](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/)
and the **2014 train and validation images** from [COCO Downloads](https://cocodataset.org/#download).
Keep the original COCO filenames:

```text
raw/
  coco-qa/train/{questions,img_ids,answers}.txt
  coco-qa/test/{questions,img_ids,answers}.txt
  ms-coco/train2014/COCO_train2014_000000000009.jpg
  ms-coco/val2014/COCO_val2014_000000000042.jpg
```

```bash
python scripts/prepare_data.py --dataset coco-qa --source raw --root data
```

Each question is paired with its original numeric image ID. The vocabulary covers
both annotation splits, matching the recovered closed-vocabulary task; frequency
orders answers, with alphabetical tie breaking. Classifier size comes from that
saved vocabulary (430 for the full expected release), rather than the number of
answers observed in one split. This explicit ordering can differ from old NumPy
tie ordering; do not reuse an old classifier without checking its label map.

## COCO retrieval (`coco-retrieval`)

Use the same original 2014 COCO images and obtain the JSON-only annotation archive
from the [Karpathy split release](https://cs.stanford.edu/people/karpathy/deepimagesent/).
Place `dataset_coco.json` under `raw/karpathy/`:

```bash
python scripts/prepare_data.py --dataset coco-retrieval --source raw --root data
```

The adapter follows each record's `filepath` and `filename`; a training record can
refer to `val2014`. It preserves the recovered `train`/`test` selection and excludes
`val` and `restval` (it does not silently switch to the common 113k training recipe).
The first five captions are retained, as in the recovered COCO loader; records with
fewer than five fail validation. Evaluation size comes from the manifest, not a
hard-coded 5,000 when running an explicitly prepared subset.

## Flickr30K (`flickr30k`)

Request images through the [Flickr30K provider](https://shannon.cs.illinois.edu/DenotationGraph/)
and obtain `dataset_flickr30k.json` from the same [Karpathy release](https://cs.stanford.edu/people/karpathy/deepimagesent/).
The provider's access process and image terms apply; images are not redistributed.

```text
raw/flickr30k-images/<original filename>.jpg
raw/karpathy/dataset_flickr30k.json
```

```bash
python scripts/prepare_data.py --dataset flickr30k --source raw --root data
```

Captions and splits come from each annotation record, and images are matched by
filename. There is no dependency on a Hugging Face row order or a process-global
caption cache. The `train`/`test` convention excludes Karpathy `val`; the first
five captions are used. `raw/flickr30k/images/` and `raw/images/` are also accepted.

## Balanced T4SA (`t4sa`)

Request **B-T4SA** from [the T4SA provider](http://www.t4sa.it/). Supply the acquired
images, train/test lists, and raw tweet text; authentication and current source
availability are controlled by the provider. The release does not scrape tweets
or infer missing text. Expected layout:

```text
raw/b-t4sa/b-t4sa_train.txt
raw/b-t4sa/b-t4sa_test.txt
raw/b-t4sa/raw_tweets_text.csv
raw/b-t4sa/data/<directory>/<tweet-id>-1.jpg
```

The CSV must have `id,text` columns. List rows are `relative/image.jpg label`.
IDs stay strings, so large numeric tweet IDs retain all digits. Labels are
`0=negative`, `1=neutral`, `2=positive`.

```bash
python scripts/prepare_data.py --dataset t4sa --source raw --root data
```

Passing `--source raw/b-t4sa` directly also works. No pickle cache is loaded or
written, and preparing another root cannot reuse the previous root's tweet text.

## UCF101 audio subset (`ucf101`)

Acquire `UCF101.rar` through the [UCF101 provider](https://www.crcv.ucf.edu/data/UCF101.php).
Extract it with an archive utility supporting RAR (for example `unrar` or 7-Zip)
into a separate source directory. On Linux install the distribution's `unrar` or
`7zip` package; macOS users can use a RAR-capable archive application. No extraction
utility is needed by the trainer once the files exist.

```text
raw/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi
```

```bash
python scripts/prepare_data.py --dataset ucf101 --source raw --root data --check-media
```

The shipped `src/mpsl/available_datasets/multimodal/action_recognition/assets/ucf101/samples.txt`
selects the recovered 51 classes with audio and their train/test assignment. This
is not the full 101-class task. A source with class directories directly beneath
its root, or under `ucf101/`, is also accepted. Missing selected videos fail before
training; constructors never redownload or delete anything.

## Kinetics-Sounds (`kinetics-sounds`)

Follow the [Kinetics provider's acquisition and extraction instructions](https://github.com/cvdfoundation/kinetics-dataset)
for Kinetics-400. The recovered subset follows [AV-CIL](https://github.com/weiguoPian/AV-CIL_ICCV2023).
For the provider's class-organized layout, prepare from the directory containing
`k400/` (or from its parent containing `kinetics-dataset/`):

```text
raw/k400/videos/train/playing accordion/OU79KEfzOTc_000146_000156.mp4
```

```bash
python scripts/prepare_data.py --dataset kinetics-sounds --source raw --root data --check-media
```

The shipped `assets/kinetics_sounds/samples.txt` and `class_names.txt` under
`action_recognition/` define the selected clips, train/test split and all 30 labels.
An already acquired subset arranged as `<class with spaces or underscores>/<file>`
is accepted too. Every selected file is checked. Preparation does not duplicate
the hundreds-of-GB parent corpus; **do not delete the referenced source after
preparation**. Provider scripts that rearrange raw data are separate, explicit
acquisition steps, outside this repository's importer.

## MELD (`meld`)

Acquire the **raw media**, not precomputed features, using the download links on
[the MELD project page](https://affective-meld.github.io/). Both the original host
and a provider-linked Hugging Face mirror are listed there. Extract
`MELD.Raw.tar.gz`, then its train/dev/test media archives into their corresponding
folders:

```text
raw/MELD.Raw/train/train_splits/dia0_utt0.mp4
raw/MELD.Raw/dev/dev_splits_complete/dia0_utt0.mp4
raw/MELD.Raw/test/output_repeated_splits_test/dia0_utt0.mp4
```

```bash
python scripts/prepare_data.py --dataset meld --source raw --root data --check-media
```

The shipped `emotion_recognition/assets/meld/samples.txt` supplies utterance text
and split assignments, including the recovered training/dev combination. The
fixed class map is neutral, surprise, fear, sadness, joy, disgust and anger. A
source rooted at `MELD.Raw/`, or the older `emotion/train_diaN_uttN.mp4` layout, is
also supported. No shell `find`, archive extraction, renaming or deletion runs
inside the loader.

**Audio compatibility:** the recovered pipeline selects a 2.56-second clip and,
after resampling to 16 kHz, pads to 122,880 samples (7.68 seconds), producing
`1 × 766 × 128` filterbanks. This release preserves that input shape and clip
convention. The model option is historically named `3.88s`; it is not the actual
waveform duration. MELD now selects that compatible option by default. Evaluation
uses the first clip and deterministic padding; training samples a random clip.
A new, genuinely 3.88-second recipe would require changing tokenizer dimensions
and validating it separately.

## Explicit subsets and evaluation conventions

For UCF101, Kinetics-Sounds and MELD, `--samples-file /path/to/samples.txt` accepts an
explicit subset in the same format as the shipped file. The manifest marks it as
custom and hashes the list. Both splits must remain present; the global label map
is retained even if a split lacks classes. This supports smaller acquired datasets
and execution fixtures without claiming the full paper split was recovered.
For image/text datasets, supply the corresponding source-format annotation subset
in a separate source root. Counts and annotation hashes identify it.

UCF/Kinetics training uses random frames and the recovered image augmentation;
evaluation uses the center decoded frame and resize/center-crop. Audio is mono,
resampled to 16 kHz, aligned around that frame and padded to a fixed size. Use
`--audio_time_dimension 1s` (98 frames) or `10.24s` (1,024 frames). Invalid/empty
streams fail explicitly. These deterministic evaluation fixes can change results
relative to the original randomly sampled evaluation.

CIFAR100 remains an additional unimodal baseline with its torchvision acquisition
path; it is not one of the seven multimodal datasets above. All published metrics,
full corpus availability, and accelerator performance require separate validation.
