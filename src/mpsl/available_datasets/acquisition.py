"""Official dataset acquisition recipes; paper split definitions stay unchanged."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from urllib.parse import urlparse

from .downloads import download, extract_archive


@dataclass(frozen=True)
class Resource:
    url: str
    filename: str
    destination: str


RESOURCES = {
    "cocoqa": Resource("https://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip", "cocoqa-2015-05-17.zip", "coco-qa"),
    "coco-train": Resource("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip", "ms-coco"),
    "coco-val": Resource("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip", "ms-coco"),
    "karpathy": Resource("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", "caption_datasets.zip", "karpathy"),
    "ucf101": Resource("http://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar", "UCF101.rar", "."),
    "meld": Resource("https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz", "MELD.Raw.tar.gz", "."),
}
MANUAL = {
    "flickr30k": "Download the Flickr30K images via https://shannon.cs.illinois.edu/DenotationGraph/ and place them in raw/flickr30k-images/<image>.jpg (or put flickr30k-images.tar.gz in raw/).",
    "t4sa": "Request T4SA access at http://www.t4sa.it/. Put b-t4sa_imgs.tar and raw_tweets_text.csv in raw/, or extract them under raw/b-t4sa/ with b-t4sa_train.txt, b-t4sa_test.txt, data/, and raw_tweets_text.csv.",
}
ENCODER_NAME = "Meta-Transformer_base_patch16_encoder.pth"
ENCODER_ID = "19ahcN2QKknkir_bayhTW5rucuAiX0OXq"
KINETICS_INDEX = "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
KINETICS_REPLACEMENTS = "https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz"


def _resource(key, source):
    resource = RESOURCES[key]
    cache = source / ".downloads"
    local = source / resource.filename
    archive = local if local.is_file() else download(resource.url, cache / resource.filename)
    print(f"Extracting {archive.name}", flush=True)
    extract_archive(archive, source / resource.destination)


def acquire_sources(dataset, source, *, samples_file=None):
    from .preparation import _records
    source = Path(source).resolve()
    source.mkdir(parents=True, exist_ok=True)
    try:
        _records(dataset, source, samples_file)
        return
    except FileNotFoundError:
        pass
    if dataset in ("coco-qa", "coco-retrieval"):
        # Annotations identify the exact original image split for every record.
        annotation = "cocoqa" if dataset == "coco-qa" else "karpathy"
        expected = ([f"coco-qa/{split}/{name}.txt" for split in ("train", "test")
                     for name in ("questions", "img_ids", "answers")]
                    if dataset == "coco-qa" else ["karpathy/dataset_coco.json"])
        if not all((source / path).is_file() for path in expected):
            _resource(annotation, source)
        try:
            _records(dataset, source, samples_file)
            return  # Shared COCO images may already be complete.
        except FileNotFoundError:
            pass
        for split in ("train", "val"):
            # Extraction is repeatable, including a partially populated image folder.
            _resource(f"coco-{split}", source)
    elif dataset == "flickr30k":
        for filename in ("flickr30k-images.tar.gz", "flickr30k-images.tar", "flickr30k-images.zip"):
            if (source / filename).is_file():
                extract_archive(source / filename, source)
                break
        if not any((source / folder).is_dir() for folder in ("flickr30k-images", "flickr30k/images", "images")):
            raise FileNotFoundError(MANUAL[dataset])
        if not (source / "karpathy/dataset_flickr30k.json").is_file():
            _resource("karpathy", source)
    elif dataset == "t4sa":
        if (source / "b-t4sa_imgs.tar").is_file():
            extract_archive(source / "b-t4sa_imgs.tar", source)
        try:
            _records(dataset, source, samples_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{MANUAL[dataset]}\nMissing: {error}") from error
    elif dataset == "ucf101":
        _resource("ucf101", source)
    elif dataset == "meld":
        _resource("meld", source)
        base = source / "MELD.Raw"
        for split in ("train", "dev", "test"):
            # The original release bundles a second archive for each split.
            for suffix in (".tar.gz", ".tar"):
                archive = base / f"{split}{suffix}"
                if archive.is_file():
                    extract_archive(archive, base / split)
                    break
    elif dataset == "kinetics-sounds":
        _kinetics(source, samples_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # Fail explicitly if the provider release lacks any required paper sample.
    _records(dataset, source, samples_file)


def _kinetics(source, samples_file):
    from .preparation import asset_dir, find_file
    samples = Path(samples_file) if samples_file else asset_dir("kinetics-sounds") / "samples.txt"
    required = {}
    for line in samples.read_text().splitlines():
        if not line.strip():
            continue
        relative = line.split()[0]
        name = Path(relative).name
        if name in required and required[name] != relative:
            raise ValueError(f"Ambiguous Kinetics filename: {name}")
        required[name] = relative
    cache = source / ".downloads/kinetics"
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / "extracted.json"
    selection = hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest()
    saved = json.loads(receipt.read_text()) if receipt.exists() else {}
    completed = saved.get("shards", {}) if saved.get("selection") == selection else {}

    def missing():
        result = {}
        for name, relative in required.items():
            spaced = str(Path(Path(relative).parts[0].replace("_", " ")) / name)
            try:
                find_file(source, relative, f"kinetics-sounds/{relative}", spaced, f"kinetics-sounds/{spaced}", f"k400/videos/train/{spaced}", f"kinetics-dataset/k400/videos/train/{spaced}")
            except FileNotFoundError:
                result[name] = relative
        return result

    pending = missing()
    if not pending:
        return
    print("Kinetics-Sounds: scanning the official K400 training shards; only paper-listed clips are retained. The upstream transfer is several hundred GB.", flush=True)
    index = download(KINETICS_INDEX, cache / "k400_train_path.txt")
    urls = [KINETICS_REPLACEMENTS, *index.read_text().splitlines()]
    for number, url in enumerate(urls, 1):
        if not url.strip():
            continue
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.netloc != "s3.amazonaws.com" or not parsed.path.startswith("/kinetics/400/"):
            raise ValueError(f"Unexpected K400 archive source: {url}")
        previous = completed.get(url)
        if previous is not None and all((source / "kinetics-sounds" / path).is_file() and (source / "kinetics-sounds" / path).stat().st_size == size for path, size in previous.items()):
            continue
        print(f"Kinetics shard {number}/{len(urls)}; {len(pending)} clips remaining", flush=True)
        archive = download(url, cache / Path(parsed.path).name)
        files = extract_archive(archive, source / "kinetics-sounds", selected=pending)
        completed[url] = {str(p.relative_to(source / "kinetics-sounds")): p.stat().st_size for p in files}
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"selection": selection, "shards": completed}, indent=2) + "\n")
        temporary.replace(receipt)
        archive.unlink()  # Avoid retaining the full K400 corpus after selection.
        pending = missing()
        if not pending:
            return
    raise FileNotFoundError(f"The official K400 archives did not supply {len(pending)} required clips. First missing: {next(iter(pending))}. No incomplete dataset manifest was created.")


def ensure_encoder(directory, *, allow_download=True):
    """Download the authors' base encoder once and verify all tensor names/shapes."""
    import torch
    from timm.models.vision_transformer import Block
    directory = Path(directory)
    target = directory / ENCODER_NAME
    if not target.is_file():
        if not allow_download:
            raise FileNotFoundError(f"Place https://drive.google.com/file/d/{ENCODER_ID}/view at {target}, or rerun without --no-download.")
        import gdown
        from gdown.exceptions import FileURLRetrievalError
        from requests.exceptions import RequestException
        directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        print(f"Downloading {ENCODER_NAME}", flush=True)
        try:
            result = gdown.download(id=ENCODER_ID, output=str(partial), quiet=False, resume=True)
            if not result:
                raise OSError("No file returned")
        except (FileURLRetrievalError, RequestException, OSError) as error:
            raise OSError(f"Encoder download unavailable. Download https://drive.google.com/file/d/{ENCODER_ID}/view to {target} and rerun.") from error
        candidate = partial
    else:
        candidate = target
    try:
        weights = torch.load(candidate, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        if candidate != target:
            candidate.unlink(missing_ok=True)
        raise ValueError(f"Invalid encoder checkpoint: {candidate}. Replace it with the official base encoder.") from error
    with torch.device("meta"):
        expected = torch.nn.Sequential(*[Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True) for _ in range(12)]).state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys() or any(not isinstance(weights[k], torch.Tensor) or weights[k].shape != expected[k].shape for k in expected):
        if candidate != target:
            candidate.unlink(missing_ok=True)
        raise ValueError(f"Invalid Meta-Transformer base encoder: {candidate}")
    if candidate != target:
        candidate.replace(target)
    return target
