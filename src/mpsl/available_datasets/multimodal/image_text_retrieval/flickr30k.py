import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor
import torch

from mpsl.available_datasets.multimodal.image_text_retrieval import data_partitioner
from mpsl.models import InputModality
from mpsl.available_datasets.utils.dataset_utils import download_and_extract_data

HUGGINGFACE_IMAGES_DATASET = 'nlphuji/flickr30k'
TEST_SET_LEN = 1_000
NR_OF_CAPTIONS_PER_IMAGE = 5

# All labels of the dataset, cached in-memory. Our data partitioning (to allow for iid and non-iid) requires for the targets to be present during construction of partitions.
# Instead of maintaining a collection for each instance of this class, we'll maintain a singleton instance to speed things up and keep memory usage low.
cached_dataset_targets = []


def get_split_name(train):
    return 'train' if train else 'test'


def collate_fn(batch):
    images = []
    text = []

    for element in batch:
        X, y = element
        img, txt = X[InputModality.IMAGE], y

        images.append(img)
        text.append(txt)

    return torch.stack(images), np.asarray(text, dtype='str')


class Flickr30K:
    """
    Uses https://huggingface.co/datasets/nlphuji/flickr30k for the full dataset, and divides it into train, validation and test set as per the Karpathy splits (https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

    The data splits are 29000 train images, 1014 validation images, 1000 test images, where each image has 5 captions.
    """
    custom_partitioner_function = data_partitioner
    custom_collate_fn = collate_fn

    def __init__(self, train=False, transform=None, **kwargs):
        super().__init__()

        self.custom_partitioner_function = data_partitioner.image_text_retrieval__data_partitioner

        self.data_processor_name = 'vit-mae-base'
        self.transform = transform
        self.data_processor = None if self.transform is not None else AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=os.path.join(os.environ['PRE_PROCESSORS_CACHE_DIR'], self.data_processor_name))

        self.train = train

        # The corresponding indices in the HuggingFace dataset
        self.indices = []
        self.y = []

        self.karpathy_folder_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'karpathy')
        self.karpathy_splits_file_path = os.path.join(self.karpathy_folder_dir, 'dataset_flickr30k.json')

        self.download_and_extract_split_data()
        self.load_karpathy_splits_raw_data()

        self.dataset = load_dataset(HUGGINGFACE_IMAGES_DATASET)['test']
        self.extract_targets()

    def extract_targets(self):
        for idx in range(len(self.indices)):
            captions = cached_dataset_targets[self.indices[idx]]
            targets = [captions[0]] if NR_OF_CAPTIONS_PER_IMAGE == 1 else captions

            self.y.append(targets)

    def load_karpathy_splits_raw_data(self):
        relevant_split = get_split_name(self.train)

        with open(self.karpathy_splits_file_path) as file:
            for element in json.load(file)['images']:
                split = element['split']

                sentences = [sentence['raw'] for sentence in element['sentences']]
                cached_dataset_targets.append(sentences)

                if split == relevant_split:
                    img_id = element['imgid']

                    self.indices.append(img_id)

    def download_and_extract_split_data(self):
        download_and_extract_data(
            [
                # Karpathy splits. Note: make sure to include https, otherwise the file won't be found.
                (self.karpathy_folder_dir, self.karpathy_folder_dir,
                 'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip')
            ]
        )

    @property
    def _classes(self):
        return np.array([])

    @property
    def num_classes(self):
        return len(np.unique(self.targets))

    def __getitem__(self, idx):
        try:
            data = self.dataset[self.indices[idx]]
            image = data['image']

            if self.data_processor is not None:
                image = self.data_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

            if self.transform is not None:
                image = self.transform(image)

            return {
                InputModality.IMAGE: image
            }, self.y[idx]
        except Exception as error:
            print(f'An exception occurred while loading values for Flickr30K with idx {idx}')

            raise error


    @property
    def targets(self):
        # Note that this function is only used by the DistributedDataset construction code. Using the text captions as targets for its construction yields undesired results.
        # Instead, we'll have the datasets be constructed by using the indices of the images instead.
        return self.indices

    def __len__(self):
        return len(self.targets)
