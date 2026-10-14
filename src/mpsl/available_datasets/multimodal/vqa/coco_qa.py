import os

import numpy as np
from torchvision.datasets.folder import default_loader as image_file_loader
from tqdm import tqdm
from transformers import AutoImageProcessor

from mpsl.available_datasets.utils.dataset_utils import download_and_extract_data
from mpsl.models import InputModality
from mpsl.utils.file_utils import read_file_to_np_array, save_file, read_file_as_lines

FILETYPE = 'jpg'
# The maximum number of unique answers in both the train and val sets combined.
MAX_NR_OF_UNIQUE_ANSWERS_FOR_COCO_QA = 430
NR_OF_CLASSES = MAX_NR_OF_UNIQUE_ANSWERS_FOR_COCO_QA


def original_file_name_to_image_id(original_file_name):
	"""
	Returns an original filename of e.g. COCO_train2014_000000475546.jpg as 475546
	"""
	return int(original_file_name.replace('COCO_train2014_', '').replace('COCO_val2014_', '').replace(f'.{FILETYPE}', ''))


def rename_image_files(ms_coco_root_path, train_images_path, val_images_path):
	"""
	By default, image filenames are denoted as 'COCO_train2014_000000000009' for the image with id 9.
	This function will truncate filenames by changing them to merely their id. Hence 'COCO_train2014_000000000009,jpg' -> '9.jpg'

	Renaming the files like this will prevent us from having to pad the zeroes in the names accordingly.
	"""
	setup_file_path = os.path.join(ms_coco_root_path, 'setup.log')
	finished_text = 'Finished renaming files'

	if os.path.exists(setup_file_path) and finished_text in read_file_as_lines(setup_file_path):
		return

	print('Renaming all MS-COCO images')

	for folder in [train_images_path, val_images_path]:
		all_files = os.listdir(folder)

		for file in tqdm(all_files):
			if 'COCO' in file:
				os.rename(os.path.join(folder, file), os.path.join(folder, f"{original_file_name_to_image_id(file)}.{FILETYPE}"))

	save_file(setup_file_path, f'{finished_text}\n', 'a')


class COCO_QA:
	"""
	COCO-QA is a dataset that has questions and answers for a subset of the original MS-COCO dataset.
	Note that we're utilizing the 2014 version of the MS-COCO dataset in combination with the 2014 (although updated on May 17, 2015) COCO-QA annotations.

	Read more about the dataset at https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/
	"""

	custom_collate_fn = None

	def __init__(self, train=False, transform=None, **kwargs):
		super().__init__()
		self.data_processor_name = 'vit-mae-base'
		self.data_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=os.path.join(os.environ['PRE_PROCESSORS_CACHE_DIR'], self.data_processor_name))
		self.transform = transform

		self.train = train

		self.image_ids = None
		self.questions = None
		self.answers = None

		self.class_to_word_list = None

		root_dir = os.environ['TORCH_DATA_DIR']
		self.ms_coco_root_path = os.path.join(root_dir, 'ms-coco')
		self.coco_qa_root_path = os.path.join(root_dir, 'coco-qa')
		self.train_images_path = os.path.join(self.ms_coco_root_path, 'train2014')
		self.val_images_path = os.path.join(self.ms_coco_root_path, 'val2014')
		self.images_path = self.train_images_path if train else self.val_images_path

		self.download_and_extract_data(root_dir)
		self.rename_test_folder()
		rename_image_files(self.ms_coco_root_path, self.train_images_path, self.val_images_path)

		self.load_raw_data(self.train)
		self.convert_answers_to_classes()

	def convert_answers_to_classes(self):
		all_answers = []

		# We need to also take into account words that are in the test set, such that we can assign them a class id.
		for answers_set in [self.get_answers_for_dataset_type(train=True), self.get_answers_for_dataset_type(train=False)]:
			for answer in answers_set:
				all_answers.append(answer)

		# Sort on the X-most frequent occurrences
		unique_elements, frequency = np.unique(all_answers, return_counts=True)
		sorted_indexes = np.argsort(frequency)[::-1]

		# Create a dictionary to map words to class labels (indices), as well as the list of the sorted most frequent words themselves.
		# The list can be used to find a word by its class, the class being the index in the list.
		self.class_to_word_list = unique_elements[sorted_indexes]
		word_to_class_dict = {word: index for index, word in enumerate(self.class_to_word_list)}

		# Mapping all original answers to their respective class labels
		self.answers = [word_to_class_dict.get(word, -1) for word in self.answers]

	def load_raw_data(self, train):
		root_path = os.path.join(self.coco_qa_root_path, 'train' if train else 'val')

		self.questions = read_file_to_np_array(os.path.join(root_path, 'questions.txt'), 'str')
		self.image_ids = read_file_to_np_array(os.path.join(root_path, 'img_ids.txt'), 'int64')
		self.answers = self.get_answers_for_dataset_type(train)

	def get_answers_for_dataset_type(self, train):
		root_path = os.path.join(self.coco_qa_root_path, 'train' if train else 'val')

		return read_file_to_np_array(os.path.join(root_path, 'answers.txt'), 'str')

	def rename_test_folder(self):
		"""
		By default, the COCO-QA dataset uses a 'train' and 'test' folder, even though the 'train' and 'val' folders of MS-COCO are used.
		Hence, we'll rename the 'test' folder to 'val' as well.
		"""
		setup_file_path = os.path.join(self.ms_coco_root_path, 'setup.log')
		finished_text = 'Finished renaming test folder'

		if os.path.exists(setup_file_path) and finished_text in read_file_as_lines(setup_file_path):
			return

		print('Renaming test folder')

		os.rename(os.path.join(self.coco_qa_root_path, 'test'), os.path.join(self.coco_qa_root_path, 'val'))

		save_file(setup_file_path, f'{finished_text}\n', 'a')

	def download_and_extract_data(self, root_dir):
		download_and_extract_data(
			[
				# MS-COCO
				(self.train_images_path, os.path.join(root_dir, 'ms-coco'),
				 'http://images.cocodataset.org/zips/train2014.zip'),
				(self.val_images_path, os.path.join(root_dir, 'ms-coco'),
				 # Note that the validation instead of test set should be used.
				 'http://images.cocodataset.org/zips/val2014.zip'),

				# COCO-QA
				(self.coco_qa_root_path, self.coco_qa_root_path,
				 'http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip')
			]
		)
	@property
	def _classes(self):
		return np.unique(self.answers)

	@property
	def num_classes(self):
		return len(self._classes)

	def __getitem__(self, idx):
		try:
			with image_file_loader(os.path.join(self.images_path, f'{self.image_ids[idx]}.{FILETYPE}')) as image:
				if self.data_processor is not None:
					image = self.data_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

				if self.transform is not None:
					image = self.transform(image)

			return {
				InputModality.IMAGE: image,
				InputModality.TEXT: self.questions[idx]
			}, self.answers[idx]
		except Exception as error:
			print(f'An exception occurred while loading values for COCO-QA with idx {idx}')

			raise error

	@property
	def targets(self):
		return self.answers

	def __len__(self):
		return len(self.targets)
