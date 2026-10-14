from pathlib import Path
import csv
import os
import shutil
import subprocess
import tarfile

import requests

from mpsl.available_datasets.utils import dataset_utils
from mpsl.models import InputModality

NR_OF_CLASSES = 7


class MELD(dataset_utils.VideoAudioDataset):
    """
    MELD is an emotion recognition dataset on video, audio, text triplets (and subsequently also audio, text pairs).
    We focus on the audio, text pairs within this dataset.

    Read more about the dataset at https://affective-meld.github.io/
    """

    custom_collate_fn = None
    download_url = 'http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'

    def __init__(self,
                 train=True,
                 transform=None,
                 tubelet_size=4,
                 num_patches=16,
                 audio_sample_rate=48000,
                 video_frame_rate=25,
                 img_size=224,
                 download=True,
                 **kwargs):

        self.root_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'meld')
        self.cache_dir = os.environ['PRE_PROCESSORS_CACHE_DIR']
        self.samples_file_path = str(Path(__file__).parent / 'assets/meld/samples.txt')

        # Download dataset (if needed)
        if download:
            self._download(self.root_dir)

        # Load filepaths and labels
        video_files = [f"{self.root_dir}/{l.split(' ')[0]}" for l in open(self.samples_file_path).readlines() if l.split(" ")[-1].strip() == ('train' if train else 'test')]
        targets = [l.split("/")[0] for l in open(self.samples_file_path).readlines() if l.split(" ")[-1].strip() == ('train' if train else 'test')]

        # Load VideoDataset dataset
        super().__init__(
            video_files=video_files,
            targets=targets,
            cache_dir=self.cache_dir,
            train=train,
            tubelet_size=tubelet_size,
            num_patches=num_patches,
            audio_sample_rate=audio_sample_rate,
            video_frame_rate=video_frame_rate,
            img_size=img_size,
            include_video_modality=False,
            **kwargs
        )

        # Load text inputs and text processor
        self.text_inputs = [l.split('[')[1].split(']')[0].replace("’", "'") for l in
                            open(self.samples_file_path, encoding='windows-1252').readlines() \
                            if l.split(" ")[-1].strip() == ('train' if train else 'test')]

        self._classes = sorted(set(self.targets))
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        text = self.text_inputs[idx]

        return {
            InputModality.AUDIO: sample['audio'],
            InputModality.TEXT: text
        }, sample['label']

    @staticmethod
    def _download(root_dir):
        os.makedirs(root_dir, exist_ok=True)

        _need_download = int(
            subprocess.check_output(f"find {root_dir} -type f -name '*.mp4' | wc -l", shell=True).strip()) != 13707

        if _need_download:
            # Step 1: Download tar if it doesn't exist
            if not os.path.exists(os.path.join(root_dir, __class__.download_url.split('/')[-1])):
                response = requests.get(__class__.download_url, stream=True)
                with open(os.path.join(root_dir, __class__.download_url.split('/')[-1]), 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            # Step 2: Extract main tarball if not already extracted
            if not os.path.exists(os.path.join(root_dir, __class__.download_url.split('/')[-1].replace('.tar.gz', ''))):
                with tarfile.open(os.path.join(root_dir, __class__.download_url.split('/')[-1]), 'r:gz') as tar:
                    tar.extractall(path=root_dir)

            # Step 3: Extract sub-tarballs if needed
            for tar_file in ['train.tar.gz', 'dev.tar.gz', 'test.tar.gz']:
                tar_path = os.path.join(root_dir, 'MELD.Raw', tar_file)
                target_dir = os.path.join(root_dir, tar_file.split('.')[0])
                if os.path.exists(tar_path) and not os.path.exists(target_dir):
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(path=target_dir)

            # Step 4: Process metadata and create data structure
            class_names, samples = {}, []

            for csv_path in ['./train/train_sent_emo.csv', './MELD.Raw/dev_sent_emo.csv', './MELD.Raw/test_sent_emo.csv']:
                split = os.path.basename(csv_path).split('_')[0]
                prefix = 'dev_splits_complete' if split == 'dev' else (
                    'output_repeated_splits_test' if split == 'test' else 'train_splits')
                src_dir = os.path.join(root_dir, split, prefix)

                with open(os.path.join(root_dir, csv_path), 'r') as file:
                    for row in csv.DictReader(file):
                        class_name = row['Emotion'].strip().replace(',', '').replace(' ', '_')
                        src_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
                        dest_filename = f"{split}_dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
                        src_path = os.path.join(src_dir, src_filename)
                        if os.path.exists(src_path):
                            os.makedirs(class_dir := os.path.join(root_dir, class_name), exist_ok=True)
                            shutil.move(src_path, os.path.join(class_dir, dest_filename))
                            class_names.setdefault(class_name, len(class_names))
                            utterance = row['Utterance'].strip() if 'Utterance' in row else "[]"
                            samples.append(
                                f"{class_name}/{dest_filename} {class_name} [{utterance}] {'train ' if split in ['train', 'dev'] else 'test'}")

            if not os.path.exists(os.path.join(root_dir, 'class_names.txt')):
                # Step 5: Save metadata
                with open(os.path.join(root_dir, 'class_names.txt'), 'w') as f:
                    for class_name, class_id in class_names.items():
                        f.write(f"{class_name}\t{class_id}\n")

            # Step 6: Cleanup
            for path in ['train', 'dev', 'test', 'MELD.Raw', 'train/train_sent_emo.csv', 'MELD.Raw.tar.gz']:
                full_path = os.path.join(root_dir, path)

                if os.path.exists(full_path):
                    shutil.rmtree(full_path, ignore_errors=True) if os.path.isdir(full_path) else os.remove(full_path)
