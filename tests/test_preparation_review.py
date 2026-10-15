"""Independent checks of prepared-data integrity and sampling semantics."""

import json
import random
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torchvision.transforms import ToTensor

from test_prepared_datasets import picture, source_fixture
from mpsl.available_datasets import load_data
from mpsl.available_datasets.preparation import prepare_dataset


def test_preparation_rejects_cross_split_hardlinks(tmp_path):
    source, _ = source_fixture(tmp_path, "flickr30k")
    train = source / "flickr30k-images/999.jpg"
    test = source / "flickr30k-images/1.jpg"
    test.unlink()
    test.hardlink_to(train)
    with pytest.raises(ValueError, match="overlap"):
        prepare_dataset("flickr30k", tmp_path / "prepared", source)
    assert not (tmp_path / "prepared/flickr30k").exists()


@pytest.mark.parametrize("split,filename", [("train", "999.jpg"), ("test", "1.jpg")])
def test_primary_loader_enforces_opted_in_content_hash(tmp_path, monkeypatch, split, filename):
    source, _ = source_fixture(tmp_path, "flickr30k")
    path = source / "flickr30k-images" / filename
    # BMP's fixed payload size lets us change decoded content without size drift.
    Image.new("RGB", (32, 24), "red").save(path, format="BMP")
    root = tmp_path / "prepared"
    prepare_dataset("flickr30k", root, source, hash_media=True)
    size = path.stat().st_size
    Image.new("RGB", (32, 24), "blue").save(path, format="BMP")
    assert path.stat().st_size == size
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    with pytest.raises(ValueError, match="hash|changed"):
        dataset = load_data("flickr30k", num_partitions=1, min_num_samples=1,
                            transform=ToTensor(), global_args=SimpleNamespace())
        if split == "test":
            dataset.load_test_set()


def test_retrieval_partition_uses_explicit_seed_without_global_rng(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, "flickr30k")
    annotation = source / "karpathy/dataset_flickr30k.json"
    content = json.loads(annotation.read_text())
    prototype = content["images"][0]
    for idx in range(20, 43):
        name = f"{idx}.jpg"
        picture(source / "flickr30k-images" / name, (idx, 10, 30))
        content["images"].append(dict(prototype, filename=name, imgid=idx))
    annotation.write_text(json.dumps(content))
    root = tmp_path / "prepared"
    prepare_dataset("flickr30k", root, source)
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    partitions = []
    for global_seed, partition_seed in [(1, 7), (2, 7), (1, 9)]:
        random.seed(global_seed)
        state = random.getstate()
        dataset = load_data("flickr30k", num_partitions=3, min_num_samples=1,
                            seed=partition_seed, transform=ToTensor(), global_args=SimpleNamespace())
        partitions.append([dataset.load_partition(cid).indices for cid in range(3)])
        assert random.getstate() == state
    assert partitions[0] == partitions[1]
    assert partitions[0] != partitions[2]
    assert sorted(i for part in partitions[0] for i in part) == list(range(24))


def test_audio_window_at_clip_start_stays_frame_centered(tmp_path, monkeypatch):
    torchaudio = pytest.importorskip("torchaudio")
    from mpsl.available_datasets.prepared import audio_features

    monkeypatch.setattr(torchaudio.compliance.kaldi, "fbank", lambda waveform, **kwargs: waveform)
    waveform = torch.arange(32000, dtype=torch.float32).reshape(1, -1)
    actual = audio_features(waveform, 16000, 1., center=.1)
    # The original interval [center-.5, center+.5] intersects the source at
    # [0, .6], with right padding to the one-second feature window.
    expected = torch.nn.functional.pad(waveform[:, :9600], (0, 6400))
    expected -= expected.mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
