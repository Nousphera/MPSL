"""Source-format fixtures exercise the same seven loaders used by experiments."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from mpsl.available_datasets import load_data
from mpsl.models import InputModality


def picture(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def source_fixture(tmp_path, dataset):
    source = tmp_path / "raw data"
    source.mkdir()
    if dataset == "coco-qa":
        for split, folder, idx, answer in [
            ("train", "train2014", 11, "cat"),
            ("test", "val2014", 22, "zebra"),
        ]:
            picture(
                source / "ms-coco" / folder / f"COCO_{folder}_{idx:012d}.jpg",
                (idx, 0, 0),
            )
            ann = source / "coco-qa" / split
            ann.mkdir(parents=True)
            for name, text in [
                ("questions", "What animal?"),
                ("img_ids", str(idx)),
                ("answers", answer),
            ]:
                (ann / f"{name}.txt").write_text(text + "\n")
    elif dataset in ("coco-retrieval", "flickr30k"):
        rows = []
        # Non-positional IDs, train uses a val2014 image: must follow filename/filepath.
        for split, idx, folder in [("train", 999, "val2014"), ("test", 1, "train2014")]:
            name = f"{idx}.jpg"
            base = (
                source / "ms-coco" / folder
                if dataset == "coco-retrieval"
                else source / "flickr30k-images"
            )
            picture(base / name, (idx % 255, 20, 50))
            rows.append(
                dict(
                    split=split,
                    imgid=idx,
                    filepath=folder,
                    filename=name,
                    sentences=[{"raw": f"{split} caption {i}"} for i in range(5)],
                )
            )
        ann = source / "karpathy"
        ann.mkdir()
        (
            ann
            / (
                "dataset_coco.json"
                if dataset == "coco-retrieval"
                else "dataset_flickr30k.json"
            )
        ).write_text(json.dumps({"images": rows}))
    elif dataset == "t4sa":
        base = source / "b-t4sa"
        base.mkdir()
        with (base / "raw_tweets_text.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            for split, idx, label in [
                ("train", "784760943188189184", 2),
                ("test", "784760943188189185", 0),
            ]:
                picture(base / "data" / f"{idx}-1.jpg", (20, label, 30))
                (base / f"b-t4sa_{split}.txt").write_text(f"data/{idx}-1.jpg {label}\n")
                writer.writerow([idx, f"{split} exact tweet"])
    else:
        from test_current_dependencies import media_fixture

        samples = []
        for split, name in [("train", "a"), ("test", "b")]:
            if dataset == "ucf101":
                relative, label = f"ApplyEyeMakeup/{name}.mkv", "ApplyEyeMakeup"
                line = f"{relative} {label} {split}"
                target = source / "UCF-101" / relative
            elif dataset == "kinetics-sounds":
                relative, label = f"playing_accordion/{name}.mkv", "11"
                line = f"{relative} {label} {split}"
                target = (
                    source
                    / "k400"
                    / "videos"
                    / "train"
                    / "playing accordion"
                    / f"{name}.mkv"
                )
            else:
                relative, label = f"neutral/{split}_dia0_utt0.mkv", "neutral"
                line = f"{relative} {label} [some [nested] text] {split}"
                target = (
                    source
                    / split
                    / (
                        "train_splits"
                        if split == "train"
                        else "output_repeated_splits_test"
                    )
                    / "dia0_utt0.mkv"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            media_fixture(target)
            samples.append(line)
        ann = tmp_path / "samples.txt"
        ann.write_text("\n".join(samples) + "\n")
        return source, ann
    return source, None


@pytest.mark.parametrize(
    "name",
    [
        "coco-qa",
        "t4sa",
        "coco-retrieval",
        "flickr30k",
        "ucf101",
        "kinetics-sounds",
        "meld",
    ],
)
def test_prepare_then_primary_loader(tmp_path, monkeypatch, name):
    from mpsl.available_datasets.preparation import prepare_dataset, validate_dataset

    source, samples = source_fixture(tmp_path, name)
    original = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "prepared"
    manifest = prepare_dataset(
        name, root, source, samples_file=samples, hash_media=True, check_media=True
    )
    assert manifest["counts"] == {"train": 1, "test": 1}
    assert validate_dataset(root, name)["counts"] == manifest["counts"]
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    ds = load_data(
        name,
        num_partitions=1,
        min_num_samples=1,
        transform=ToTensor(),
        global_args=SimpleNamespace(audio_time_dimension="1s"),
    )
    train, test = ds[0], ds.load_test_set()
    x, y = test[0]
    x2, y2 = test[0]
    assert y == y2 if isinstance(y, list) else int(y) == int(y2)
    for key in x:
        if isinstance(x[key], torch.Tensor):
            assert torch.isfinite(x[key]).all()
            torch.testing.assert_close(x[key], x2[key], rtol=0, atol=0)
    if name == "coco-qa":
        assert ds.num_classes == test.num_classes == 2
        assert int(train[1]) != int(y)
        assert int(y) < test.num_classes
    elif name in ("coco-retrieval", "flickr30k"):
        assert y == [f"test caption {i}" for i in range(5)]
        assert train[1] == [f"train caption {i}" for i in range(5)]
        assert (
            load_data(
                name,
                num_partitions=1,
                min_num_samples=1,
                transform=ToTensor(),
                global_args=SimpleNamespace(),
            )[0][1]
            == train[1]
        )
    elif name == "t4sa":
        assert x[InputModality.TEXT] == "test exact tweet"
    else:
        assert (
            ds.num_classes
            == test.num_classes
            == {"ucf101": 51, "kinetics-sounds": 30, "meld": 7}[name]
        )
        assert x[InputModality.AUDIO].shape[-1] == 128
        if name == "meld":
            assert x[InputModality.AUDIO].shape == (1, 766, 128)
    assert original == {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        prepare_dataset(name, root, source, samples_file=samples)
    file = next(p for p in source.rglob("*") if p.suffix in (".jpg", ".mkv"))
    file.write_bytes(file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|size|hash"):
        validate_dataset(root, name)


def test_eval_audio_transform_is_deterministic():
    pytest.importorskip("torchaudio")
    from mpsl.available_datasets.utils.dataset_utils import load_transform

    transform = load_transform("audio", train=False, duration=16000)
    audio = torch.randn(1, 32000)
    torch.testing.assert_close(transform(audio), transform(audio), rtol=0, atol=0)


def test_missing_preparation_gives_command(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_data(
            "coco-qa",
            num_partitions=1,
            min_num_samples=1,
            transform=ToTensor(),
            global_args=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "name", ["coco-qa", "t4sa", "ucf101", "kinetics-sounds", "meld"]
)
def test_prepared_classification_batch_updates_real_split_trainer(
    tmp_path, monkeypatch, name
):
    from mpsl.available_datasets.preparation import prepare_dataset
    from mpsl.trainers.implementations.classification.mpsl_trainer import MPSLTrainer

    source, samples = source_fixture(tmp_path, name)
    prepare_dataset(name, tmp_path / "prepared", source, samples_file=samples)
    monkeypatch.setenv("TORCH_DATA_DIR", str(tmp_path / "prepared"))
    dataset = load_data(
        name,
        num_partitions=1,
        min_num_samples=1,
        transform=ToTensor(),
        global_args=SimpleNamespace(audio_time_dimension="1s"),
    )

    class Client(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 8)

        def forward(self, x):
            count = len(next(iter(x.values())))
            columns = []
            for key in (InputModality.IMAGE, InputModality.AUDIO):
                value = x.get(key)
                columns.append(
                    value.reshape(count, -1).mean(1)
                    if value is not None
                    else torch.zeros(count)
                )
            columns.extend(
                [
                    torch.tensor(
                        [len(v) / 100 for v in x.get(InputModality.TEXT, [""] * count)]
                    ),
                    torch.ones(count),
                ]
            )
            return self.layer(torch.stack(columns, dim=1))

    client, server = Client(), torch.nn.Linear(8, dataset.num_classes)
    before = server.weight.detach().clone()
    result = MPSLTrainer("early", [])._early_fusion_epoch(
        torch.device("cpu"),
        server,
        torch.optim.SGD(server.parameters(), lr=0.01),
        {0: iter(torch.utils.data.DataLoader(dataset, batch_size=1))},
        {0: client},
        True,
        {0: torch.optim.SGD(client.parameters(), lr=0.01)},
        1,
        SimpleNamespace(nr_of_clients=1),
    )
    assert not torch.equal(before, server.weight)
    assert all(torch.isfinite(p).all() for p in server.parameters())


@pytest.mark.parametrize("name", ["coco-retrieval", "flickr30k"])
def test_prepared_retrieval_batch_trains_and_evaluates(tmp_path, monkeypatch, name):
    from mpsl.available_datasets.preparation import prepare_dataset
    from mpsl.trainers.implementations.image_text_retrieval.mpsl_trainer import (
        MPSLTrainer,
    )
    from mpsl.trainers.implementations.experiment_results import ExperimentResults

    source, _ = source_fixture(tmp_path, name)
    annotation = next((source / "karpathy").glob("*.json"))
    rows = json.loads(annotation.read_text())
    extra = dict(
        rows["images"][0],
        imgid=44,
        filename="44.jpg",
        sentences=[{"raw": f"different caption {i}"} for i in range(5)],
    )
    rows["images"].append(extra)
    annotation.write_text(json.dumps(rows))
    picture(
        source
        / ("ms-coco/val2014" if name == "coco-retrieval" else "flickr30k-images")
        / "44.jpg",
        (0, 180, 30),
    )
    prepare_dataset(name, tmp_path / "prepared", source)
    monkeypatch.setenv("TORCH_DATA_DIR", str(tmp_path / "prepared"))
    ds = load_data(
        name,
        num_partitions=1,
        min_num_samples=1,
        transform=ToTensor(),
        global_args=SimpleNamespace(),
    )

    class Encoder(torch.nn.Module):
        def __init__(self, client=False):
            super().__init__()
            self.client = client
            self.image = torch.nn.Linear(3, 3)
            self.text = torch.nn.Linear(3, 3)

        def forward(self, x):
            im, txt = x[InputModality.IMAGE], x[InputModality.TEXT]
            if self.client:
                im = im.mean((2, 3))
                txt = torch.tensor(
                    [
                        [
                            [len(s) / 100, sum(map(ord, s)) / 1000, int(s[-1]) / 5]
                            for s in captions
                        ]
                        for captions in txt
                    ]
                )
            return {
                InputModality.IMAGE: self.image(im),
                InputModality.TEXT: self.text(txt),
            }

    client, server = Encoder(True), Encoder()
    initial = server.image.weight.detach().clone()
    trainer, results = MPSLTrainer(5, 1, 2), ExperimentResults()
    optimizer = torch.optim.SGD(client.parameters(), lr=0.01)
    trainer._train_epoch(
        results,
        server,
        torch.optim.SGD(server.parameters(), lr=0.01),
        {
            0: iter(
                torch.utils.data.DataLoader(
                    ds, batch_size=2, collate_fn=ds.get_collate_fn()
                )
            )
        },
        {0: client},
        True,
        {0: optimizer},
        {0: torch.optim.lr_scheduler.StepLR(optimizer, 1)},
        1,
        1,
        SimpleNamespace(nr_of_clients=1),
    )
    assert not torch.equal(initial, server.image.weight)
    trainer._test_epoch(
        torch.device("cpu"),
        results,
        server,
        client,
        torch.utils.data.DataLoader(
            ds.load_test_set(), batch_size=2, collate_fn=ds.get_collate_fn()
        ),
        1,
    )
    assert results.test_metric == [[[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]]


def test_rejects_missing_nonfinal_media_and_split_overlap(tmp_path):
    from mpsl.available_datasets.preparation import prepare_dataset

    source, _ = source_fixture(tmp_path, "flickr30k")
    (source / "flickr30k-images/999.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        prepare_dataset("flickr30k", tmp_path / "prepared", source)
    picture(source / "flickr30k-images/999.jpg", "red")
    path = source / "karpathy/dataset_flickr30k.json"
    annotation = json.loads(path.read_text())
    annotation["images"][1]["filename"] = "999.jpg"
    path.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="overlap"):
        prepare_dataset("flickr30k", tmp_path / "prepared", source)


def test_rejects_escaping_source_symlink(tmp_path):
    from mpsl.available_datasets.preparation import prepare_dataset

    source, _ = source_fixture(tmp_path, "flickr30k")
    original = source / "flickr30k-images/999.jpg"
    external = tmp_path / "external.jpg"
    original.rename(external)
    original.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        prepare_dataset("flickr30k", tmp_path / "prepared", source)


def test_rejects_truncated_image_during_preparation(tmp_path):
    from mpsl.available_datasets.preparation import prepare_dataset

    source, _ = source_fixture(tmp_path, "flickr30k")
    path = source / "flickr30k-images/999.jpg"
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(OSError):
        prepare_dataset("flickr30k", tmp_path / "prepared", source)


def test_invalid_video_rate_and_empty_frames_fail_before_sampling():
    pytest.importorskip("torchaudio")
    from mpsl.available_datasets.utils.dataset_utils import VideoAudioDataset

    dataset = object.__new__(VideoAudioDataset)
    for frames, fps in [(torch.empty(0, 3, 16, 16), 8), (torch.empty(8, 3, 16, 16), 0)]:
        with pytest.raises(ValueError, match="frames|rates"):
            dataset._get_clip_pts(
                frames, torch.empty(1, 16000), {"video_fps": fps, "audio_fps": 16000}
            )


def test_model_head_and_saved_provenance_use_prepared_vocabulary(tmp_path, monkeypatch):
    pytest.importorskip("open_clip")
    from mpsl.available_datasets.preparation import prepare_dataset, sha256
    from mpsl.models import get_split_model_pair_and_trainer
    import mpsl.models as models
    from mpsl.trainers.implementations.experiment_results import ExperimentResults
    from mpsl.utils.argument_utils import (
        build_base_argument_parser,
        validate_base_argument_constraints,
    )

    source, _ = source_fixture(tmp_path, "coco-qa")
    root = tmp_path / "prepared"
    prepare_dataset("coco-qa", root, source)
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    args = build_base_argument_parser().parse_args(
        [
            "--dataset",
            "coco-qa",
            "--save_file_name",
            "fixture",
            "--torch_data_dir",
            str(root),
        ]
    )
    validate_base_argument_constraints(args)
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(fusion_type="early"), None, True

    monkeypatch.setattr(models, "get_meta_vqa_split_model", factory)
    get_split_model_pair_and_trainer(args, torch.device("cpu"))
    assert received["num_classes"] == 2
    result = ExperimentResults(args).to_json()
    assert result["dataset"]["sha256"] == sha256(root / "coco-qa/manifest.json")
    assert result["dataset"]["counts"] == {"train": 1, "test": 1}
    assert result["configuration"]["save_file_name"] == "fixture"


def test_meld_cli_selects_compatible_default_audio_shape():
    pytest.importorskip("open_clip")
    from mpsl.utils.argument_utils import (
        build_base_argument_parser,
        validate_base_argument_constraints,
    )

    args = build_base_argument_parser().parse_args(
        ["--dataset", "meld", "--save_file_name", "fixture"]
    )
    validate_base_argument_constraints(args)
    assert args.audio_time_dimension == "3.88s"
