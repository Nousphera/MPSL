import fractions
import math
import os
import random

import torch
import torchaudio
import torchvision
import transformers
from torchvision.datasets.utils import download_and_extract_archive


def download_and_extract_data(data_pairs_to_download):
    for full_extracted_folder_path, unextracted_folder_path, data_url in data_pairs_to_download:
        if not os.path.exists(full_extracted_folder_path):
            print('Downloading data')

            download_and_extract_archive(url=data_url, download_root=unextracted_folder_path, remove_finished=True)


class UniformClipSampler:

    def __init__(self, clip_duration, stride=None, backpad_last=False, eps=1e-6, ):
        self._clip_duration = fractions.Fraction(clip_duration)
        self._stride = stride if stride is not None else self._clip_duration
        self._eps = eps
        self._backpad_last = backpad_last
        self._current_clip_index = 0
        assert self._stride > 0, "stride must be positive"

    def _clip_start_end(self, video_duration, backpad_last):
        delta = self._stride - self._clip_duration
        last_end_time = -delta
        clip_start = fractions.Fraction(last_end_time + delta)
        clip_end = fractions.Fraction(clip_start + self._clip_duration)
        if backpad_last:
            buffer_amount = max(0, clip_end - video_duration)
            clip_start -= buffer_amount
            clip_start = fractions.Fraction(max(0, clip_start))  # handle rounding
            clip_end = fractions.Fraction(clip_start + self._clip_duration)
        return clip_start, clip_end

    def __call__(self, video_duration):
        clip_start, clip_end = self._clip_start_end(video_duration, backpad_last=self._backpad_last)
        _, next_clip_end = self._clip_start_end(video_duration, backpad_last=self._backpad_last)

        return clip_start, clip_end

    def reset(self):
        self._current_clip_index = 0


class RandomClipSampler:

    def __init__(self, clip_duration):
        self._clip_duration = fractions.Fraction(clip_duration)
        self._current_clip_index = 0
        self._current_aug_index = 0

    def __call__(self, video_duration):
        max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        clip_start_sec = fractions.Fraction(random.uniform(0, max_possible_clip_start))
        return clip_start_sec, clip_start_sec + self._clip_duration

    def reset(self):
        self._current_clip_index = 0


class RandomShortSideScale(torch.nn.Module):

    def __init__(self, min_size, max_size, interpolation="bilinear"):  # , dtype=torch.uint8):
        super().__init__()
        self.min_size, self.max_size, self.interpolation = min_size, max_size, interpolation

    def forward(self, x):
        size = torch.randint(self.min_size, self.max_size + 1, (1,)).item()
        return self.short_side_scale(x, size=size)

    def short_side_scale(self, x, size):
        assert len(x.shape) == 4
        _, _, h, w = x.shape
        new_h, new_w = (int(math.floor((h / w) * size)), size) if w < h else (size, int(math.floor((w / h) * size)))
        return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=False)


class UniformTemporalSubsample(torch.nn.Module):

    def __init__(self, num_samples: int = 16):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        assert self.num_samples > 0 and t > 0, "Invalid temporal dimension or num_samples"
        indices = torch.linspace(0, t - 1, self.num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, dim=0, index=indices)


def load_transform(modality='video', train=True, **kwargs):
    if modality == 'audio':
        return torchvision.transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=kwargs.get('sample_rate', 16000), new_freq=16000),
            torchvision.transforms.RandomCrop(size=(1, kwargs.get('duration', math.floor(10.24 * 16000))),
                                              pad_if_needed=True),
            torchvision.transforms.Lambda(lambda x: x - x.mean()),
            torchvision.transforms.Lambda(lambda x: torchaudio.compliance.kaldi.fbank(x,
                                                                                      htk_compat=True,
                                                                                      sample_frequency=16000,
                                                                                      window_type='hanning',
                                                                                      num_mel_bins=kwargs.get(
                                                                                          'num_mel_bins', 128))),
        ])
    elif modality == 'video':
        assert 'image_processor' in kwargs, "The 'image_processor' key must be present in kwargs for video."
        if train:
            return torchvision.transforms.Compose([
                UniformTemporalSubsample(num_samples=kwargs.get('num_samples', 16)),
                torchvision.transforms.Lambda(lambda x: x / 255.),
                # NOTE: We perform normalization directly with `image_processor`, so we can omit this.
                # torchvision.transforms.Normalize(mean=kwargs.get('mean', [0.5, 0.5, 0.5]), std=kwargs.get('std', [0.5, 0.5, 0.5])),
                RandomShortSideScale(min_size=kwargs.get('min_size', 256), max_size=kwargs.get('max_size', 320)),
                # , dtype=kwargs.get('dtype', torch.uint8)),
                torchvision.transforms.RandomCrop(size=(kwargs.get('img_size', 224), kwargs.get('img_size', 224))),
                torchvision.transforms.RandomHorizontalFlip(p=kwargs.get('flip_prob', 0.5)),
                torchvision.transforms.Lambda(lambda x: torch.stack(
                    [torch.tensor(kwargs['image_processor'](images=i, return_tensors="np")['pixel_values']) for i in
                     x])),
                torchvision.transforms.Lambda(lambda x: x.squeeze().permute(1, 0, 2, 3)),
                # [Channels x Frames x Width x Height]
            ])
        else:
            return torchvision.transforms.Compose([
                UniformTemporalSubsample(num_samples=kwargs.get('num_samples', 16)),
                # NOTE: We perform rescale, normalization and resize directly with `image_processor`, so we can omit those.
                # torchvision.transforms.Lambda(lambda x: x/255.),
                # torchvision.transforms.Normalize(mean=kwargs.get('mean', [0.5, 0.5, 0.5]), std=kwargs.get('std', [0.5, 0.5, 0.5])),
                # torchvision.transforms.Resize(size=(kwargs.get('img_size', 224), kwargs.get('img_size', 224))),
                torchvision.transforms.Lambda(lambda x: torch.stack(
                    [torch.tensor(kwargs['image_processor'](images=i, return_tensors="np")['pixel_values']) for i in
                     x])),
                torchvision.transforms.Lambda(lambda x: x.squeeze().permute(1, 0, 2, 3)),
                # [Channels x Frames x Width x Height]
            ])


class VideoAudioDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            video_files,
            targets,
            cache_dir,
            train=True,
            tubelet_size=2,
            num_patches=16,
            audio_sample_rate=16000,
            video_frame_rate=25,
            img_size=224,
            include_video_modality=True,
            **kwargs
    ):

        self.video_files, self.targets = video_files, targets

        self._classes = sorted(set(self.targets))
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        self.train = train
        self.tubelet_size = tubelet_size
        self.num_patches = num_patches

        self.include_video_modality = include_video_modality
        self._image_processor = transformers.VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base",
                                                                                    cache_dir=os.path.join(cache_dir,
                                                                                                           'videomae'),
                                                                                    do_rescale=False if train else True,
                                                                                    use_fast=True)
        self.audio_sample_rate = audio_sample_rate
        self.video_frame_rate = video_frame_rate
        # NOTE: Overwrite these values for each dataset.
        self._clip_sampler = RandomClipSampler(self.video_duration) if train else UniformClipSampler(
            self.video_duration)
        self._video_transform = load_transform(modality='video', train=train, img_size=img_size,
                                               num_samples=self.num_patches, image_processor=self._image_processor)
        self._audio_transform = load_transform(modality='audio', train=train, duration=self.audio_duration,
                                               sample_rate=self.audio_sample_rate)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def video_duration(self):
        return self.num_patches * self.tubelet_size / self.video_frame_rate

    @property
    def audio_duration(self):
        return math.floor(self.video_duration * self.audio_sample_rate)

    def __len__(self):
        return len(self.video_files)

    def _get_clip_pts(self, video, audio, info):
        start, end = self._clip_sampler(video.shape[0] / info[
            'video_fps'])  # if video.shape[0]/info['video_fps']>self.video_duration(info['video_fps']) else (0.0, video.shape[0]/info['video_fps'])
        video_start = max(0, math.floor(start * info['video_fps']))
        video_end = min(math.ceil(end * info['video_fps']), video.shape[0])
        audio_start = max(0, math.floor(start * info['audio_fps']))
        audio_end = min(math.ceil(end * info['audio_fps']), audio.shape[-1])
        return [video_start, video_end], [audio_start, audio_end]

    def __getitem__(self, idx):
        fp, label = self.video_files[idx], self.targets[idx]
        video, audio, info = torchvision.io.read_video(fp, output_format='TCHW', pts_unit='sec')
        if 'audio_fps' not in info.keys():
            print(fp, info, video, audio)
        assert info[
                   'audio_fps'] == self.audio_sample_rate, f"{fp} sr {info['audio_fps']} does not match expected sr of {self.audio_sample_rate}."
        # Get clip points from video
        video_pts, audio_pts = self._get_clip_pts(video, audio, info)

        # Extract audio modality
        audio = audio.mean(0, keepdim=True)[:, audio_pts[0]:audio_pts[1]]
        if self._audio_transform:
            audio = self._audio_transform(audio).unsqueeze(0)

        data = {'audio': audio, 'label': torch.tensor(self._class_to_idx[label], dtype=torch.long)}

        # Extract video modality
        if self.include_video_modality:
            video = video[video_pts[0]:video_pts[1], ::]

            if self._video_transform:
                video = self._video_transform(video)

            data['video'] = video

        return data