"""Read-only import of acquired paper datasets into validated training manifests.

Media stays in its source directory. No constructor downloads, archive extraction,
renaming, cache pickle loading, or deletion is performed.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

DATASETS = (
    "coco-qa",
    "t4sa",
    "coco-retrieval",
    "flickr30k",
    "ucf101",
    "kinetics-sounds",
    "meld",
)
ASSETS = Path(__file__).parent / "multimodal"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(source, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source-relative path: {relative}")
    path = (source / relative).resolve()
    if not path.is_relative_to(source.resolve()):
        raise ValueError(f"Source path escapes its root: {relative}")
    return path


def find_file(source, *candidates):
    for candidate in candidates:
        path = safe_path(source, candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing source file; expected one of: {candidates} below {source}"
    )


def asset_dir(dataset):
    category = "emotion_recognition" if dataset == "meld" else "action_recognition"
    return ASSETS / category / "assets" / dataset.replace("-", "_")


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _records(dataset, source, samples_file=None):
    records = {"train": [], "test": []}
    annotations = []
    classes = {}

    def annotation(*candidates):
        path = find_file(source, *candidates)
        annotations.append(path)
        return path

    def add(split, path, **fields):
        records[split].append({"path": str(path.relative_to(source)), **fields})

    if dataset == "coco-qa":
        raw = {}
        for split in records:
            raw[split] = {}
            for name in ("questions", "img_ids", "answers"):
                path = annotation(f"coco-qa/{split}/{name}.txt", f"{split}/{name}.txt")
                raw[split][name] = read_lines(path)
            if len({len(v) for v in raw[split].values()}) != 1:
                raise ValueError(f"COCO-QA {split} annotation lengths disagree")
        counts = Counter(a for split in raw.values() for a in split["answers"])
        classes = {
            answer: i
            for i, (answer, _) in enumerate(
                sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            )
        }
        if len(classes) > 430:
            raise ValueError(
                "COCO-QA contains more than the supported 430 answer classes"
            )
        for split, fields in raw.items():
            folder = "train2014" if split == "train" else "val2014"
            for question, image_id, answer in zip(
                fields["questions"], fields["img_ids"], fields["answers"]
            ):
                image_id = int(image_id)
                path = find_file(
                    source,
                    f"ms-coco/{folder}/COCO_{folder}_{image_id:012d}.jpg",
                    f"{folder}/COCO_{folder}_{image_id:012d}.jpg",
                    f"ms-coco/{folder}/{image_id}.jpg",
                )
                add(
                    split,
                    path,
                    text=question,
                    label=classes[answer],
                    identity=str(image_id),
                )
    elif dataset in ("coco-retrieval", "flickr30k"):
        name = (
            "dataset_coco.json"
            if dataset == "coco-retrieval"
            else "dataset_flickr30k.json"
        )
        path = annotation(f"karpathy/{name}", name)
        for row in json.loads(path.read_text())["images"]:
            split = row["split"]
            if (
                split not in records
            ):  # Preserve recovered train/test convention, exclude val/restval.
                continue
            name = row["filename"]
            if dataset == "coco-retrieval":
                folder = row.get("filepath")
                if not folder:
                    folder = (
                        "train2014"
                        if "train2014" in name
                        else "val2014" if "val2014" in name else None
                    )
                if folder not in ("train2014", "val2014"):
                    raise ValueError(
                        f"COCO image needs an explicit train2014/val2014 filepath: {name}"
                    )
                path = find_file(source, f"ms-coco/{folder}/{name}", f"{folder}/{name}")
            else:
                path = find_file(
                    source,
                    f"flickr30k-images/{name}",
                    f"flickr30k/images/{name}",
                    f"images/{name}",
                    name,
                )
            captions = [s["raw"] for s in row["sentences"]]
            if len(captions) < 5 or any(
                not isinstance(c, str) or not c.strip() for c in captions
            ):
                raise ValueError(f"Expected at least five nonempty captions for {name}")
            add(
                split, path, captions=captions[:5], identity=str(row.get("imgid", name))
            )
    elif dataset == "t4sa":
        tweets = annotation("b-t4sa/raw_tweets_text.csv", "raw_tweets_text.csv")
        with tweets.open(newline="", encoding="utf-8") as stream:
            text = {}
            for row in csv.DictReader(stream):
                if row["id"] in text:
                    raise ValueError(f'Duplicate tweet id: {row["id"]}')
                text[row["id"]] = row["text"]
        classes = {"negative": 0, "neutral": 1, "positive": 2}
        for split in records:
            path = annotation(f"b-t4sa/b-t4sa_{split}.txt", f"b-t4sa_{split}.txt")
            for line in read_lines(path):
                relative, label = line.rsplit(maxsplit=1)
                tweet_id = Path(relative).stem.split("-")[0]
                if tweet_id not in text:
                    raise ValueError(f"Missing tweet text: {tweet_id}")
                add(
                    split,
                    find_file(source, f"b-t4sa/{relative}", relative),
                    text=text[tweet_id],
                    label=int(label),
                    identity=tweet_id,
                )
    else:
        assets = asset_dir(dataset)
        samples = (
            Path(samples_file).resolve() if samples_file else assets / "samples.txt"
        )
        annotations.append(samples)
        if dataset == "ucf101":
            classes = {
                name: i
                for i, name in enumerate(
                    sorted(
                        {line.split()[1] for line in read_lines(assets / "samples.txt")}
                    )
                )
            }
            annotations.append(assets / "samples.txt")
        else:
            classes = {
                name: int(label)
                for name, label in (
                    line.split() for line in read_lines(assets / "class_names.txt")
                )
            }
            annotations.append(assets / "class_names.txt")
        for line in samples.read_text(
            encoding="windows-1252" if dataset == "meld" else "utf-8"
        ).splitlines():
            if not line.strip():
                continue
            if dataset == "meld":
                match = re.fullmatch(r"(\S+) (\S+) \[(.*)\] (train|test)", line)
                if not match:
                    raise ValueError(f"Invalid MELD sample record: {line}")
                relative, name, text, split = match.groups()
                filename = Path(relative).name
                prefix, filename = filename.split("_", 1)
                folders = {
                    "train": "train_splits",
                    "dev": "dev_splits_complete",
                    "test": "output_repeated_splits_test",
                }
                if prefix not in folders:
                    raise ValueError(f"Unknown MELD source split: {prefix}")
                candidates = [
                    relative,
                    f"{prefix}/{folders[prefix]}/{filename}",
                    f"MELD.Raw/{prefix}/{folders[prefix]}/{filename}",
                    f"{folders[prefix]}/{filename}",
                ]
                fields = {"text": text.replace("’", "'")}
                label = classes[name]
            else:
                relative, name, split = line.split()
                fields = {}
                if dataset == "ucf101":
                    label = classes[name]
                    candidates = [relative, f"UCF-101/{relative}", f"ucf101/{relative}"]
                else:
                    label = int(name)
                    class_name = Path(relative).parts[0]
                    if classes[class_name] != label:
                        raise ValueError(
                            f"Kinetics label does not match the global class map: {relative}"
                        )
                    spaced = str(
                        Path(class_name.replace("_", " ")) / Path(relative).name
                    )
                    candidates = [
                        relative,
                        spaced,
                        f"kinetics-sounds/{spaced}",
                        f"k400/videos/train/{spaced}",
                        f"kinetics-dataset/k400/videos/train/{spaced}",
                    ]
            if split not in records:
                raise ValueError(f"Invalid split: {split}")
            add(
                split,
                find_file(source, *candidates),
                label=label,
                identity=relative,
                **fields,
            )
    return records, classes, list(dict.fromkeys(annotations))


def _check_records(manifest):
    source = Path(manifest["source_root"])
    classes = manifest["class_to_idx"]
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("Class map must be contiguous from zero")
    split_paths = {}
    split_identities = {}
    split_files = {}
    file_identities = {}
    for split in ("train", "test"):
        records = manifest["records"][split]
        if not records:
            raise ValueError(f"Empty {split} split")
        paths = []
        identities = []
        physical_files = []
        for row in records:
            path = safe_path(source, row["path"])
            if row["path"] not in file_identities:
                stat = path.stat()
                file_identities[row["path"]] = (stat.st_dev, stat.st_ino)
            physical_files.append(file_identities[row["path"]])
            paths.append(row["path"])
            identities.append(row["identity"])
            if manifest["dataset"] in ("coco-retrieval", "flickr30k"):
                if len(row.get("captions", [])) != 5 or any(
                    not isinstance(c, str) or not c.strip() for c in row["captions"]
                ):
                    raise ValueError("Retrieval records require five nonempty captions")
            elif "label" not in row:
                raise ValueError("Classification record is missing its label")
            if "label" in row and (
                type(row["label"]) is not int or not 0 <= row["label"] < len(classes)
            ):
                raise ValueError(f"Label outside global class map: {row}")
            if "text" in row and (
                not isinstance(row["text"], str) or not row["text"].strip()
            ):
                raise ValueError(f'Empty text: {row["path"]}')
        if manifest["dataset"] != "coco-qa" and len(set(physical_files)) != len(paths):
            raise ValueError(f"Duplicate media in {split}")
        split_paths[split] = set(paths)
        split_identities[split] = set(identities)
        split_files[split] = set(physical_files)
        if manifest["counts"][split] != len(records):
            raise ValueError("Manifest counts do not match its records")
    if (
        split_paths["train"] & split_paths["test"]
        or split_identities["train"] & split_identities["test"]
        or split_files["train"] & split_files["test"]
    ):
        raise ValueError("Train/test media overlap")
    if set(manifest["files"]) != split_paths["train"] | split_paths["test"]:
        raise ValueError("Manifest file inventory does not cover every record")


def inspect_media(path, full_decode=False):
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        from PIL import Image

        with Image.open(path) as image:
            image.load()  # verify() alone does not catch truncated JPEG payloads.
    else:
        import av

        with av.open(str(path)) as container:
            if not container.streams.audio or not container.streams.video:
                raise ValueError(f"Both audio and video streams are required: {path}")
            video = container.streams.video[0]
            rate = video.average_rate or video.guessed_rate
            if (
                rate is None
                or rate <= 0
                or container.streams.audio[0].codec_context.sample_rate <= 0
            ):
                raise ValueError(f"Invalid stream rates: {path}")
            if next(container.decode(video), None) is None:
                raise ValueError(f"Empty video stream: {path}")
        if full_decode:
            from mpsl.available_datasets.utils.media import read_video_audio

            read_video_audio(path)


def prepare_dataset(
    dataset, root, source, *, samples_file=None, hash_media=False, check_media=False
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    source, root = Path(source).resolve(), Path(root).resolve()
    destination = root / dataset
    if destination.exists():
        raise FileExistsError(f"{destination} exists; use --validate or a new --root")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if samples_file and dataset not in ("ucf101", "kinetics-sounds", "meld"):
        raise ValueError("--samples-file is only for UCF101, Kinetics-Sounds and MELD")
    records, classes, annotations = _records(dataset, source, samples_file)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "source_root": str(source),
        "class_to_idx": classes,
        "records": records,
        "counts": {split: len(rows) for split, rows in records.items()},
        "custom_samples": samples_file is not None,
        "annotations": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in annotations
        ],
        "files": {row["path"]: {} for rows in records.values() for row in rows},
    }
    _check_records(manifest)
    for relative in manifest["files"]:
        path = safe_path(source, relative)
        inspect_media(path, full_decode=check_media)
        manifest["files"][relative] = {"size": path.stat().st_size}
        if hash_media:
            manifest["files"][relative]["sha256"] = sha256(path)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive destination creation prevents overwriting an existing preparation.
    destination.mkdir()
    try:
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
    except BaseException:
        (destination / "manifest.json").unlink(missing_ok=True)
        destination.rmdir()
        raise
    return manifest


def load_manifest(root, dataset):
    path = Path(root) / dataset / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Run python scripts/prepare_data.py --dataset {dataset} --root "{root}" --source /path/to/acquired/data'
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != dataset:
        raise ValueError("Unexpected dataset manifest schema or name")
    _check_records(manifest)
    return manifest


def validate_dataset(root, dataset, *, check_media=False):
    manifest = load_manifest(root, dataset)
    for annotation in manifest["annotations"]:
        if sha256(annotation["path"]) != annotation["sha256"]:
            raise ValueError(f'Annotation changed: {annotation["path"]}')
    for relative, info in manifest["files"].items():
        path = safe_path(Path(manifest["source_root"]), relative)
        if not path.is_file() or path.stat().st_size != info["size"]:
            raise ValueError(f"Source media missing or size changed: {path}")
        if "sha256" in info and sha256(path) != info["sha256"]:
            raise ValueError(f"Source media hash changed: {path}")
        inspect_media(path, full_decode=check_media)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Prepared dataset root consumed by --torch_data_dir",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Acquired, extracted original dataset root (read only)",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        help="Explicit UCF/Kinetics/MELD subset in the shipped sample-list format; recorded as custom",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing manifest and all referenced files",
    )
    parser.add_argument(
        "--hash-media",
        action="store_true",
        help="SHA256 every source media file; reads the entire corpus",
    )
    parser.add_argument(
        "--check-media",
        action="store_true",
        help="Fully decode audio and video (images always fully decoded)",
    )
    args = parser.parse_args(argv)
    if args.validate:
        if args.source or args.samples_file or args.hash_media:
            parser.error(
                "--validate uses the saved sources and hashing policy; do not pass import options"
            )
        result = validate_dataset(args.root, args.dataset, check_media=args.check_media)
    else:
        if args.source is None:
            parser.error("--source is required when preparing a dataset")
        result = prepare_dataset(
            args.dataset,
            args.root,
            args.source,
            samples_file=args.samples_file,
            hash_media=args.hash_media,
            check_media=args.check_media,
        )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "counts": result["counts"],
                "classes": len(result["class_to_idx"]),
                "manifest": str(args.root.resolve() / args.dataset / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
