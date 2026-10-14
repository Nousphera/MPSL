import os

import numpy as np
from torchvision.datasets.folder import default_loader as image_file_loader
from transformers import AutoImageProcessor
import pandas as pd
import pickle

from mpsl.models import InputModality

# As per https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
# To prevent OSError: image file is truncated (2 bytes not processed)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


NR_OF_CLASSES = 3
IMAGE_FILETYPE = 'jpg'
RAW_TWEETS_FILE_NAME = 'raw_tweets_text.csv'
TWEET_ID_TO_DF_IDX_DICT_FILE_NAME = 'tweet_id_to_df_idx_dict.pkl'

# Treated as a global singleton instance, given it's large size.
# Represents the text associated with a tweet. Stored in a large Pandas DataFrame consisting of all T4SA tweets (so not only ones of the B-T4SA balanced subset).
# The df has two columns: <id, text>, where the id refers to the id of the tweet and the text refers to the actual textual tweet itself.
ALL_TWEETS_DF = None
TWEET_ID_TO_DF_IDX_DICT = None


def parse_tweet_id_from_image_file_path(image_file_path):
	"""
	Parses e.g. 'data/78476/784760943188189184-1.jpg' to '784760943188189184'
	"""
	path_chunks = image_file_path.split('/')
	full_file_name = path_chunks[len(path_chunks) - 1]
	stripped_file_name = full_file_name.split(f'.{IMAGE_FILETYPE}')[0]
	tweet_id = stripped_file_name.split('-')[0]

	return tweet_id


class T4SA:
	"""
	T4SA (Twitter for sentiment-analysis) is a dataset that has image, text pairs of tweets (text) accompanied by an image.
	Note that we're using the balanced version of the dataset, B-T4SA, which is a subset of the larger T4SA dataset.

	The dataset contains three numerical classes: 0, 1 & 2. Representing negative, neutral, and positive sentiment, respectively.

	Read more about the dataset at http://www.t4sa.it/
	"""

	custom_collate_fn = None

	def __init__(self, train=False, transform=None, **kwargs):
		super().__init__()
		self.data_processor_name = 'vit-mae-base'
		self.data_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=os.path.join(os.environ['PRE_PROCESSORS_CACHE_DIR'], self.data_processor_name))
		self.transform = transform

		self.train = train

		self.image_file_paths = []

		self.tweet_ids = []
		self.y = []

		root_dir = os.environ['TORCH_DATA_DIR']
		self.t4sa_root_path = os.path.join(root_dir, 'b-t4sa')

		if not os.path.exists(self.t4sa_root_path):
			raise Exception('T4SA dataset has not been properly downloaded and extracted. Downloading the T4SA dataset requires authentication and hence has to be performed manually. Ensure the data is downloaded into a folder named \'b-t4sa\'. We are using the B-T4SA version of the dataset. More information can be found at http://www.t4sa.it/')

		self.load_raw_data()

	def load_all_tweets_df(self):
		global ALL_TWEETS_DF

		if ALL_TWEETS_DF is None:
			ALL_TWEETS_DF = pd.read_csv(os.path.join(self.t4sa_root_path, RAW_TWEETS_FILE_NAME))

	def build_tweet_id_to_df_idx_dict_if_needed_and_load_raw_data(self):
		"""
		Finding a tweet for a given tweet_id (by finding its idx in the df by the tweet_id) is very slow.
		Instead of having to constantly perform this slow operation, we'll construct a faster lookup map once upon startup, if not already present.
		"""
		should_build_dict = not os.path.exists(os.path.join(self.t4sa_root_path, TWEET_ID_TO_DF_IDX_DICT_FILE_NAME))
		current_dataset_type = 'train' if self.train else 'test'

		global ALL_TWEETS_DF, TWEET_ID_TO_DF_IDX_DICT

		for dataset_type in ['train', 'test']:
			with open(os.path.join(self.t4sa_root_path, f"b-t4sa_{dataset_type}.txt")) as file:
				for line in file:
					# Each line is formatted in the form: 'data/78476/784760943188189184-1.jpg 0' where the first operand denotes the filepath of the image and the latter denotes the class label
					data = line.strip().split(' ')
					image_file_path = data[0]
					tweet_id = parse_tweet_id_from_image_file_path(image_file_path)

					if should_build_dict:
						if TWEET_ID_TO_DF_IDX_DICT is None:
							print('Building TWEET_ID_TO_DF_IDX_MAP. Can take up to 10 minutes.')
							TWEET_ID_TO_DF_IDX_DICT = dict()

						found_indices = ALL_TWEETS_DF.index[ALL_TWEETS_DF['id'] == int(tweet_id)]

						if len(found_indices) == 0:
							raise Exception(f'No tweet found for the given tweet_id: {tweet_id}')

						tweet_idx_in_df = found_indices[0]

						if tweet_id not in TWEET_ID_TO_DF_IDX_DICT:
							TWEET_ID_TO_DF_IDX_DICT[tweet_id] = tweet_idx_in_df

					# Regardless of building the dict for all dataset types if necessary, we also need to load the paths of the images, labels, and tweet_ids specifically for the chosen dataset type.
					if dataset_type == current_dataset_type:
						self.image_file_paths.append(image_file_path)
						self.y.append(data[1])
						self.tweet_ids.append(tweet_id)
		
		self.image_file_paths = np.asarray(self.image_file_paths, dtype='str')
		self.tweet_ids = np.asarray(self.tweet_ids, dtype='str')
		self.y = np.asarray(self.y, dtype='long')

		if should_build_dict:
			print(f'Built TWEET_ID_TO_DF_IDX_MAP with a length of {len(TWEET_ID_TO_DF_IDX_DICT)}')
			print('Saving dict')

			with open(os.path.join(self.t4sa_root_path, TWEET_ID_TO_DF_IDX_DICT_FILE_NAME), 'wb') as save_file:
				pickle.dump(TWEET_ID_TO_DF_IDX_DICT, save_file, protocol=pickle.HIGHEST_PROTOCOL)

	def load_tweet_id_to_df_idx_dict_if_needed(self):
		global TWEET_ID_TO_DF_IDX_DICT

		if TWEET_ID_TO_DF_IDX_DICT is None:
			print(f'Loading TWEET_ID_TO_DF_IDX_MAP')

			with open(os.path.join(self.t4sa_root_path, TWEET_ID_TO_DF_IDX_DICT_FILE_NAME), 'rb') as save_file:
				TWEET_ID_TO_DF_IDX_DICT = pickle.load(save_file)

	def load_raw_data(self):
		self.load_all_tweets_df()
		self.build_tweet_id_to_df_idx_dict_if_needed_and_load_raw_data()
		self.load_tweet_id_to_df_idx_dict_if_needed()

	@property
	def _classes(self):
		return np.unique([0, 1, 2])

	@property
	def num_classes(self):
		return NR_OF_CLASSES

	def __getitem__(self, idx):
		global ALL_TWEETS_DF, TWEET_ID_TO_DF_IDX_DICT

		try:
			with image_file_loader(os.path.join(self.t4sa_root_path, self.image_file_paths[idx])) as image:
				if self.data_processor is not None:
					image = self.data_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

				if self.transform is not None:
					image = self.transform(image)

			return {
				InputModality.IMAGE: image,
				InputModality.TEXT: ALL_TWEETS_DF['text'][TWEET_ID_TO_DF_IDX_DICT[self.tweet_ids[idx]]]
			}, self.y[idx]
		except Exception as error:
			print(f'An exception occurred while loading values for T4SA with idx {idx} and tweet_id {self.tweet_ids[idx]}: {error}')

			raise error

	@property
	def targets(self):
		return self.y

	def __len__(self):
		return len(self.y)
