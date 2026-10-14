import json
import os

import numpy as np
from torchvision.datasets.folder import default_loader as image_file_loader
from transformers import AutoImageProcessor

from mpsl.available_datasets.multimodal.image_text_retrieval.flickr30k import get_split_name
from mpsl.available_datasets.multimodal.vqa.coco_qa import rename_image_files, original_file_name_to_image_id
from mpsl.models import InputModality
from mpsl.available_datasets.utils.dataset_utils import download_and_extract_data

from mpsl.available_datasets.multimodal.image_text_retrieval.flickr30k import collate_fn
from mpsl.available_datasets.multimodal.image_text_retrieval import data_partitioner

FILETYPE = 'jpg'

TRAIN_SET_LEN = 82_783
TEST_SET_LEN = 5_000
NR_OF_CAPTIONS_PER_IMAGE = 5


class CocoRetrieval:
	"""
	MS-COCO for image-text retrieval is a subset of MS-COCO for which 5 captions are supplied with each image, similar to Flickr30k.
	We refer to this dataset as 'coco-retrieval' within this codebase.
	Note that we're utilizing the 2014 version of the MS-COCO dataset in combination with the 2014 (although updated on May 17, 2015) COCO-QA captions.
	Hence the downloaded files of this class are similar to those of COCO-QA and have therefore been copied.

	The dataset is divided into train, validation and test sets as per the Karpathy splits (https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

	The data splits are 82783 train images, 5000 validation images, 5000 test images, where each image has 5 captions.
	"""

	custom_collate_fn = collate_fn

	def __init__(self, train=False, transform=None, **kwargs):
		super().__init__()

		self.custom_partitioner_function = data_partitioner.image_text_retrieval__data_partitioner
		self.data_processor_name = 'vit-mae-base'
		self.transform = transform
		self.data_processor = None if self.transform is not None else AutoImageProcessor.from_pretrained(
			"facebook/vit-mae-base",
			cache_dir=os.path.join(os.environ['PRE_PROCESSORS_CACHE_DIR'], self.data_processor_name))

		self.train = train

		self.image_ids = []
		# The corresponding captions
		self.y = []

		root_dir = os.environ['TORCH_DATA_DIR']
		self.ms_coco_root_path = os.path.join(root_dir, 'ms-coco')
		self.train_images_path = os.path.join(self.ms_coco_root_path, 'train2014')
		self.val_images_path = os.path.join(self.ms_coco_root_path, 'val2014')
		self.karpathy_folder_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'karpathy')
		self.karpathy_splits_file_path = os.path.join(self.karpathy_folder_dir, 'dataset_coco.json')

		self.images_path = self.train_images_path if train else self.val_images_path

		self.download_and_extract_data(root_dir)
		rename_image_files(self.ms_coco_root_path, self.train_images_path, self.val_images_path)

		self.load_karpathy_splits_raw_data()

	def load_karpathy_splits_raw_data(self):
		relevant_split = get_split_name(self.train)

		with open(self.karpathy_splits_file_path) as file:
			for element in json.load(file)['images']:
				split = element['split']

				if split == relevant_split:
					captions = [caption['raw'] for caption in element['sentences']]
					# Some images have more than 5 captions which will cause issues with uneven shapes. Hence we'll stick to a maximum of 5 captions per image at all times.
					targets = [captions[0]] if NR_OF_CAPTIONS_PER_IMAGE == 1 else captions[:5]

					self.image_ids.append(original_file_name_to_image_id(element['filename']))
					self.y.append(targets)

	def download_and_extract_data(self, root_dir):
		download_and_extract_data(
			[
				# MS-COCO
				(self.train_images_path, os.path.join(root_dir, 'ms-coco'),
				 'http://images.cocodataset.org/zips/train2014.zip'),
				(self.val_images_path, os.path.join(root_dir, 'ms-coco'),
				 # Note that the validation instead of test set should be used.
				 'http://images.cocodataset.org/zips/val2014.zip'),

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
			with image_file_loader(os.path.join(self.images_path, f'{self.image_ids[idx]}.{FILETYPE}')) as image:
				if self.data_processor is not None:
					image = self.data_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

				if self.transform is not None:
					image = self.transform(image)

			return {
				InputModality.IMAGE: image
			}, self.y[idx]
		except Exception as error:
			print(f'An exception occurred while loading values for MS-COCO with idx {idx}')

			raise error


	@property
	def targets(self):
		# Note that this function is only used by the DistributedDataset construction code. Using the text captions as targets for its construction yields undesired results.
		# Instead, we'll have the datasets be constructed by using the indices of the images instead.
		return self.image_ids

	def __len__(self):
		return len(self.targets)
