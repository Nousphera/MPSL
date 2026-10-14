from pathlib import Path
import os
import pathlib
import random
import shutil
import urllib
import rarfile

import cv2
import torch
import torchaudio
import torchvision
from PIL import Image

from mpsl.models import InputModality

VIDEO_FILETYPE = 'avi'
NR_OF_CLASSES = 51  # There are 51 classes with audio instead of the total 101 classes.


class UCF101:
    """
    UCF101 is an activity recognition dataset that originally has video, audio pairs. We use image, audio pairs.

    Read more about the dataset at https://www.crcv.ucf.edu/data/UCF101.php
    """

    custom_collate_fn = None
    download_url = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'

    def __init__(self, train=True, transform=None, **kwargs):
        self.root_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'ucf101')
        self.samples_file_path = str(Path(__file__).parent / 'assets/ucf101/samples.txt')

        # Local import to avoid a circular dependency with the model's implementation class.
        from mpsl.models.meta_transformer.implementations.multimodal.action_recognition.models import AVAILABLE_TIME_DIMENSIONS
        time_dimension_key = kwargs['global_args'].audio_time_dimension
        time_dimension = AVAILABLE_TIME_DIMENSIONS[time_dimension_key]

        self.audio_duration_in_sec = time_dimension['audio_duration_in_sec']

        self.img_transform = transform if transform is not None else torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.RandomCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2), torchvision.transforms.ToTensor(),])
        self.train = train

        self._download(self.root_dir)

        # Load filepaths and labels
        self.video_files = [f"{self.root_dir}/{l.split(' ')[0]}" for l in open(self.samples_file_path).readlines() if l.split(" ")[-1].strip() == ('train' if train else 'test')]
        self.targets = [l.split("/")[0] for l in open(self.samples_file_path).readlines() if l.split(" ")[-1].strip() == ('train' if train else 'test')]
        self._classes = sorted(set(self.targets))
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

    @property
    def num_classes(self):
        return len(self._classes)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.targets[idx]
        label = torch.tensor(self._class_to_idx[label], dtype=torch.long)

        # Open the video file and capture properties
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_in_sec = num_frames / frame_rate  # Total video duration in seconds

        # Choose a random frame
        random_time = random.uniform(0, duration_in_sec)
        random_frame_idx = int(random_time * frame_rate)

        # Set video position to the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError(f"Failed to read frame {random_frame_idx} from video {video_path}")

        # Convert the frame to RGB format for PIL and apply image transformations
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        if self.img_transform:
            frame = self.img_transform(frame)

        audio, sr = torchaudio.load(video_path, format=VIDEO_FILETYPE)

        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            sr = 16000

        # Calculate start and end samples for audio centered around the frame
        half_duration = self.audio_duration_in_sec / 2
        audio_center_time = random_time
        audio_start_time = max(0, audio_center_time - half_duration)
        audio_end_time = min(duration_in_sec, audio_center_time + half_duration)

        # Convert time to sample indices
        start_sample = int(audio_start_time * sr)
        end_sample = int(audio_end_time * sr)
        target_samples = int(self.audio_duration_in_sec * sr)

        # Extract audio clip and zero-pad if shorter than target duration
        audio_clip = audio[:, start_sample:end_sample]
        if audio_clip.shape[1] < target_samples:
            padding = (0, target_samples - audio_clip.shape[1])
            audio_clip = torch.nn.functional.pad(audio_clip, padding)

        audio_clip = audio_clip - audio_clip.mean()
        audio_clip = torchaudio.compliance.kaldi.fbank(audio_clip, htk_compat=True, sample_frequency=sr, window_type='hanning', num_mel_bins=128)
        return {
            InputModality.IMAGE: frame,
            InputModality.AUDIO: audio_clip
        }, label

    @staticmethod
    def _download(output_dir):
        def download_file(url, dest):
            if not os.path.exists(dest):
                context = urllib.request.ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=context) as response, open(dest, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

        # Step 0: Create dataset folder
        if not pathlib.Path(output_dir).exists():
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        temp_file_path = 'temp.rar'
        temp_folder_path = './temp'

        # Step 1: Download dataset if necessary
        if not pathlib.Path(os.path.join(output_dir, temp_file_path)).exists():
            print("Downloading dataset...")
            download_file(__class__.download_url, os.path.join(output_dir, temp_file_path))

        # Step 2: Extract dataset (if exists)
        if not os.path.exists(os.path.join(output_dir, temp_folder_path)) and pathlib.Path(
                os.path.join(output_dir, temp_file_path)).exists():
            with rarfile.RarFile(os.path.join(output_dir, temp_file_path)) as rf:
                rf.extractall(os.path.join(output_dir, temp_folder_path))

        # Step 3: Reorganize dataset
        if pathlib.Path(os.path.join(output_dir, temp_folder_path, 'ucf101')):
            src_path = pathlib.Path(os.path.join(output_dir, temp_folder_path, 'ucf101'))
            for video_file in src_path.glob('*/v_*.avi'):
                class_id = video_file.parent.name
                class_dir = pathlib.Path(output_dir) / class_id
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(video_file), str(class_dir / video_file.name))
            print("Dataset preparation complete.")

        # Step 5: Cleanup
        if os.path.exists(os.path.join(output_dir, temp_folder_path)):
            shutil.rmtree(os.path.join(output_dir, temp_folder_path))  # Remove extracted directory

        if os.path.exists(os.path.join(output_dir, temp_file_path)):
            os.remove(os.path.join(output_dir, temp_file_path))  # Remove downloaded .rar file
