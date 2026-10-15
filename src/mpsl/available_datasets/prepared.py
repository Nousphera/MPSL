"""Experiment loader adapters for immutable, prepared source manifests."""

import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from mpsl.available_datasets.preparation import load_manifest, safe_path, sha256
from mpsl.models.modalities import InputModality


class PreparedDataset(Dataset):
    custom_collate_fn = None

    def __init__(self, train=False, transform=None, **kwargs):
        self.train = train
        root = Path(os.environ["TORCH_DATA_DIR"])
        self.manifest = load_manifest(root, self.dataset_name)
        self.manifest_sha256 = sha256(root / self.dataset_name / "manifest.json")
        self.source = Path(self.manifest["source_root"])
        self.records = self.manifest["records"]["train" if train else "test"]
        self._classes = self.manifest["class_to_idx"]
        self._class_to_idx = self._classes
        self.targets = [row.get("label", i) for i, row in enumerate(self.records)]
        self.transform = transform
        # Every required path is checked, not just the last entry of a subset.
        checked = set()
        for row in self.records:
            path = safe_path(self.source, row["path"])
            if path in checked:
                continue
            info = self.manifest["files"][row["path"]]
            if not path.is_file() or path.stat().st_size != info["size"]:
                raise ValueError(
                    f"Prepared source missing or changed: {path}; run scripts/prepare_data.py --validate"
                )
            if "sha256" in info and sha256(path) != info["sha256"]:
                raise ValueError(f"Prepared source hash changed: {path}")
            checked.add(path)

    @property
    def num_classes(self):
        return len(self._classes) if self._classes else len(self.records)

    def __len__(self):
        return len(self.records)

    def path(self, index):
        return safe_path(self.source, self.records[index]["path"])


class ImageTextDataset(PreparedDataset):
    retrieval = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_processor = None
        if self.transform is None:
            from transformers import AutoImageProcessor

            cache = os.environ.get("PRE_PROCESSORS_CACHE_DIR")
            self.data_processor = AutoImageProcessor.from_pretrained(
                "facebook/vit-mae-base", cache_dir=cache
            )

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.path(index)) as source:
            image = source.convert("RGB")
        image = (
            self.transform(image)
            if self.transform is not None
            else self.data_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
        )
        if self.retrieval:
            return {InputModality.IMAGE: image}, row["captions"]
        return {InputModality.IMAGE: image, InputModality.TEXT: row["text"]}, row[
            "label"
        ]


def retrieval_collate(batch):
    return torch.stack([x[InputModality.IMAGE] for x, _ in batch]), np.asarray(
        [y for _, y in batch], dtype=str
    )


class RetrievalDataset(ImageTextDataset):
    retrieval = True
    custom_collate_fn = staticmethod(retrieval_collate)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from mpsl.available_datasets.multimodal.image_text_retrieval.data_partitioner import (
            image_text_retrieval__data_partitioner,
        )

        self.custom_partitioner_function = image_text_retrieval__data_partitioner


def audio_features(audio, rate, duration, *, center=None, start=None):
    import torchaudio

    if rate <= 0 or audio.ndim != 2 or audio.shape[-1] == 0:
        raise ValueError("Audio must have samples and a positive sample rate")
    audio = audio.mean(0, keepdim=True)
    if rate != 16000:
        audio = torchaudio.functional.resample(audio, rate, 16000)
    target = int(duration * 16000)
    offset = (
        max(0, int((center - duration / 2) * 16000))
        if center is not None
        else int((start or 0) * 16000)
    )
    end = offset + target
    if center is not None:
        # Preserve the centered interval at the source boundary instead of
        # shifting it forward; pad the missing tail.
        end = min(end, max(0, int((center + duration / 2) * 16000)))
    audio = audio[:, offset:end]
    audio = torch.nn.functional.pad(audio, (0, max(0, target - audio.shape[-1])))
    audio = audio - audio.mean()
    return torchaudio.compliance.kaldi.fbank(
        audio,
        htk_compat=True,
        sample_frequency=16000,
        window_type="hanning",
        num_mel_bins=128,
    )


class AudioImageDataset(PreparedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = getattr(kwargs.get("global_args"), "audio_time_dimension", "10.24s")
        durations = {"1s": 1, "10.24s": 10.255}
        if key not in durations:
            raise ValueError(
                f"Action recognition audio_time_dimension must be one of {list(durations)}"
            )
        self.duration = durations[key]
        self.transform = self.transform or T.Compose(
            [
                T.Resize(256),
                *(
                    [
                        T.RandomCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(brightness=0.2, saturation=0.2),
                    ]
                    if self.train
                    else [T.CenterCrop(224)]
                ),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        from mpsl.available_datasets.utils.media import read_video_audio

        frames, audio, info = read_video_audio(self.path(index))
        fps = info["video_fps"]
        if not np.isfinite(fps) or fps <= 0 or len(frames) == 0:
            raise ValueError(f"Invalid frame count/rate: {self.path(index)}")
        frame_index = random.randrange(len(frames)) if self.train else len(frames) // 2
        image = Image.fromarray(frames[frame_index].permute(1, 2, 0).numpy())
        features = audio_features(
            audio, info["audio_fps"], self.duration, center=frame_index / fps
        )
        return {
            InputModality.IMAGE: self.transform(image),
            InputModality.AUDIO: features,
        }, self.targets[index]


class AudioTextDataset(PreparedDataset):
    def __init__(self, tubelet_size=4, num_patches=16, video_frame_rate=25, **kwargs):
        super().__init__(**kwargs)
        if video_frame_rate <= 0 or tubelet_size <= 0 or num_patches <= 0:
            raise ValueError("Clip dimensions and frame rate must be positive")
        self.clip_duration = num_patches * tubelet_size / video_frame_rate
        # Preserve MELD frontend dimensions: 2.56-second clip padded
        # to 122880 samples *after* resampling to 16 kHz, hence 766 mel frames.
        # The model option historically called this "3.88s"; see README.md#train-and-evaluate.
        self.feature_duration = 7.68

    def __getitem__(self, index):
        from mpsl.available_datasets.utils.media import read_audio

        audio, rate = read_audio(self.path(index))
        available = audio.shape[-1] / rate
        start = (
            random.uniform(0, max(0, available - self.clip_duration))
            if self.train
            else 0
        )
        audio = audio[:, int(start * rate) : int((start + self.clip_duration) * rate)]
        features = audio_features(audio, rate, self.feature_duration).unsqueeze(0)
        return {
            InputModality.AUDIO: features,
            InputModality.TEXT: self.records[index]["text"],
        }, self.targets[index]
