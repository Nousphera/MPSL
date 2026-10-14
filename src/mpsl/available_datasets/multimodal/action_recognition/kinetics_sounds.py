from pathlib import Path
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
from tqdm import tqdm

from mpsl.models import InputModality

NR_OF_CLASSES = 30


class KineticsSounds:
    """
    Kinetics-Sounds is a subset of Kinetics 400 and an activity recognition dataset that originally has video, audio pairs. Our implementation has image, audio pairs, however.

    Read more about the dataset in general at https://paperswithcode.com/dataset/kinetics-sound
    To acquire the data for the Kinetics-Sounds subset itself, we use https://github.com/weiguoPian/AV-CIL_ICCV2023
    """

    custom_collate_fn = None

    def __init__(self, train=True, transform=None, **kwargs):
        self.root_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'kinetics-sounds')
        self.kinetics_400_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'kinetics-dataset', 'k400', 'videos')
        self.samples_file_path = str(Path(__file__).parent / 'assets/kinetics_sounds/samples.txt')

        # Local import to avoid a circular dependency with the model's implementation class.
        from mpsl.models.meta_transformer.implementations.multimodal.action_recognition.models import AVAILABLE_TIME_DIMENSIONS
        time_dimension_key = kwargs['global_args'].audio_time_dimension
        time_dimension = AVAILABLE_TIME_DIMENSIONS[time_dimension_key]

        self.audio_duration_in_sec = time_dimension['audio_duration_in_sec']

        self.img_transform = transform if transform is not None else torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.RandomCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2), torchvision.transforms.ToTensor()])
        self.train = train

        with open(str(Path(__file__).parent / 'assets/kinetics_sounds/class_names.txt')) as f:
            self.label_to_index = {line.split()[0]: int(line.split()[1]) for line in f}
            self.classes = list(self.label_to_index.keys())

        self.extract_subset()
        self.video_files, self.labels = self.load_video_files(self.train, join_with_root_dir=True)

    def extract_subset(self):
        """Extracts the Kinetics-Sounds data from the Kinetics-400 directory and copies it into a separate folder."""
        all_video_files = []

        for is_train in [True, False]:
            video_files, _ = self.load_video_files(is_train)
            all_video_files.extend(video_files)

        # The last file has already been copied. Hence, the subset has already been extracted.
        if os.path.exists(os.path.join(self.root_dir, all_video_files[len(all_video_files) - 1])):
            return

        if not os.path.exists(os.path.join(self.kinetics_400_dir, 'train')):
            raise Exception('The Kinetics-Sounds dataset has not been properly downloaded and extracted. Please refer to the readme file of this repository for instructions regarding this dataset.')

        print('Extracting the Kinetics-Sounds subset. This may take a moment.')

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        for video_path in tqdm(all_video_files):
            src_file = os.path.join(self.kinetics_400_dir, 'train', video_path)
            dest_file = os.path.join(self.root_dir, video_path)

            if not os.path.exists(dest_file):
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copyfile(src_file, dest_file)

    def load_video_files(self, train: bool, join_with_root_dir=False):
        video_files, labels = [], []

        with open(self.samples_file_path) as f:
            for line in f:
                video_path, label, data_split = line.split()
                path_chunks = video_path.split('/')
                class_folder_name = path_chunks[0].replace('_', ' ')
                video_name = path_chunks[1]
                video_path = f"{class_folder_name}/{video_name}"

                if (train and data_split == 'train') or (not train and data_split == 'test'):
                    video_files.append(os.path.join(self.root_dir, video_path) if join_with_root_dir else video_path)
                    labels.append(int(label))

        labels = np.asarray(labels)

        return video_files, labels

    @property
    def _classes(self):
        return np.unique(self.classes)

    @property
    def num_classes(self):
        return NR_OF_CLASSES

    @property
    def targets(self):
        return self.labels

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        try:
            video_path = self.video_files[idx]
            label = self.labels[idx]

            if not os.path.exists(video_path):
                raise Exception(f'Video does not exist at path {video_path}')

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

            from mpsl.available_datasets.utils.media import read_audio
            audio, sr = read_audio(video_path)

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
        except Exception as error:
            print(f'Error occurred in kinetics_sounds __get_item__ for idx {idx}: {error}')

            raise error
