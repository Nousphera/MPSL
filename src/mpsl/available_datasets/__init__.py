import typing

import numpy as np
import numpy.typing as npt
import torch

# Import only the selected dataset; registry inspection performs no downloads.
from importlib import import_module

dataloaders = {'coco-qa': ('mpsl.available_datasets.multimodal.vqa.coco_qa', 'COCO_QA'), 't4sa': ('mpsl.available_datasets.multimodal.sentiment_analysis.t4sa', 'T4SA'), 'coco-retrieval': ('mpsl.available_datasets.multimodal.image_text_retrieval.coco_retrieval', 'CocoRetrieval'), 'flickr30k': ('mpsl.available_datasets.multimodal.image_text_retrieval.flickr30k', 'Flickr30K'), 'ucf101': ('mpsl.available_datasets.multimodal.action_recognition.ucf_101', 'UCF101'), 'kinetics-sounds': ('mpsl.available_datasets.multimodal.action_recognition.kinetics_sounds', 'KineticsSounds'), 'meld': ('mpsl.available_datasets.multimodal.emotion_recognition.meld', 'MELD')}


class Subset(torch.utils.data.Subset):

	@property
	def num_classes(self):
		return self.dataset.num_classes


class DataLoader(torch.utils.data.DataLoader):

	@property
	def num_classes(self):
		return self.dataset.num_classes


class DirichletDataPartitioner:
	"""
	As per https://github.com/adap/flower
	"""

	def __init__(
			self,
			dataset: torch.utils.data.Dataset,
			num_partitions: int,
			alpha: typing.Union[int, float, typing.List[float], np.ndarray],
			min_partition_size: int = 10,
			self_balancing: bool = False,
			shuffle: bool = True,
			seed: typing.Optional[int] = 42,
			custom_partitioner_function=None
	) -> None:

		self.dataset = dataset
		self._num_partitions = num_partitions
		self._check_num_partitions_greater_than_zero()
		self._alpha = self._initialize_alpha(alpha)
		self._min_partition_size: int = min_partition_size
		self._self_balancing = self_balancing
		self._shuffle = shuffle
		self._seed = seed
		self._rng = np.random.default_rng(seed=self._seed)
		
		if min_partition_size < 1 or num_partitions * min_partition_size > len(dataset):
			raise ValueError("Not enough samples for the requested minimum partition sizes")
		# Utility attributes
		self._avg_num_of_samples_per_partition: typing.Optional[float] = None
		self._unique_classes: typing.Optional[typing.Union[typing.List[int], typing.List[str]]] = None
		self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
		self._partition_id_to_indices_determined = False

		# Some datasets require separate code for partitioning, for which we need to override default behavior.
		self.custom_partitioner_function = custom_partitioner_function

	@property
	def num_partitions(self) -> int:
		self._check_num_partitions_correctness_if_needed()
		self._determine_partition_id_to_indices_if_needed()

		return self._num_partitions

	def _check_num_partitions_greater_than_zero(self) -> None:
		if not self._num_partitions > 0:
			raise ValueError("The number of partitions needs to be greater than zero.")

	def _check_num_partitions_correctness_if_needed(self) -> None:
		if not self._partition_id_to_indices_determined:
			if self._num_partitions > len(self.dataset):
				raise ValueError("The number of partitions needs to be smaller than the number of samples in the dataset.")
				
	def _initialize_alpha(self, alpha: typing.Union[int, float, typing.List[float], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
		if isinstance(alpha, int):
			alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, float):
			alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, typing.List):
			if len(alpha) != self._num_partitions:
				raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
			alpha = np.asarray(alpha)
		elif isinstance(alpha, np.ndarray):
			if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
				raise ValueError("If passing alpha as an NDArray, its length needs to be of length equal to num_partitions.")
			elif alpha.ndim == 2:
				alpha = alpha.flatten()
				if alpha.shape[0] != self._num_partitions:
					raise ValueError("If passing alpha as an NDArray, its size needs to be of length equal to num_partitions.")
		else:
			raise ValueError("The given alpha format is not supported.")
		if not (alpha > 0).all():
			raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
		return alpha

	def _determine_partition_id_to_indices_if_needed(self) -> None:
		if self._partition_id_to_indices_determined:
			return

		if self.custom_partitioner_function is not None:
			self._partition_id_to_indices = self.custom_partitioner_function(self._num_partitions, len(self.dataset), shuffle_indices=True, seed=self._seed)
		else:
			# Generate information needed for Dirichlet partitioning
			targets = np.asarray(self.dataset.targets) #np.array([self.dataset[i][1] for i in range(len(self.dataset))])
			self._unique_classes = np.unique(targets).tolist()
			assert self._unique_classes is not None
			self._avg_num_of_samples_per_partition = len(self.dataset) / self._num_partitions

			# Repeat the sampling procedure based on the Dirichlet distribution until the min_partition_size is reached.
			sampling_try = 0
			while True:
				# Prepare data structure to store indices assigned to partition ids
				partition_id_to_indices: typing.Dict[int, typing.List[int]] = {nid: [] for nid in range(self._num_partitions)}
				# Iterate over all unique labels
				for k in self._unique_classes:
					# Access all the indices associated with class k
					indices_representing_class_k = np.where(targets == k)[0]
					# Determine division (the fractions) of the data representing class k among the partitions
					class_k_division_proportions = self._rng.dirichlet(self._alpha)
					nid_to_proportion_of_k_samples = {nid: class_k_division_proportions[nid] for nid in range(self._num_partitions)}
					# Balancing
					if self._self_balancing:
						assert self._avg_num_of_samples_per_partition is not None
						for nid in nid_to_proportion_of_k_samples.copy():
							if len(partition_id_to_indices[nid]) > self._avg_num_of_samples_per_partition:
								nid_to_proportion_of_k_samples[nid] = 0
						sum_proportions = sum(nid_to_proportion_of_k_samples.values())
						for nid in nid_to_proportion_of_k_samples:
							nid_to_proportion_of_k_samples[nid] /= sum_proportions
					# Determine the split indices
					cumsum_division_fractions = np.cumsum(list(nid_to_proportion_of_k_samples.values()))
					cumsum_division_numbers = cumsum_division_fractions * len(indices_representing_class_k)
					indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
					split_indices = np.split(indices_representing_class_k, indices_on_which_split)
					# Append new indices (coming from class k) to the existing indices
					for nid in range(self._num_partitions):
						partition_id_to_indices[nid].extend(split_indices[nid].tolist())
				# Check if the indices assignment meets the min_partition_size
				min_sample_size_on_client = min(len(indices) for indices in partition_id_to_indices.values())
				if min_sample_size_on_client >= self._min_partition_size:
					break
				sampling_try += 1
				if sampling_try == 10:
					raise ValueError("The max number of attempts (10) was reached. Please update the values of alpha and try again.")

			# Shuffle the indices if shuffle is True
			if self._shuffle:
				for indices in partition_id_to_indices.values():
					self._rng.shuffle(indices)
			self._partition_id_to_indices = partition_id_to_indices

		self._partition_id_to_indices_determined = True

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		self._determine_partition_id_to_indices_if_needed()
		indices = self._partition_id_to_indices[partition_id]

		return Subset(self.dataset, indices)

class DistributedDataset(torch.utils.data.Dataset):
	"""
	As per FederatedDataset of https://github.com/adap/flower
	"""

	def __init__(
			self,
			dataloader,
			transform=None,
			alpha=0.5,
			num_partitions=10,
			min_partition_size=10,
			self_balancing=True, # federated params
			shuffle=False,
			seed=42, # random-ness params
			**kwargs,
		):

		self._dataloader = dataloader
		self._dataloader_args = {'transform': transform, 'seed': seed, 'global_args': kwargs['global_args']}

		# Add all dataset-specific params
		self._dataloader_args.update(kwargs)

		# Create train dataloader
		self.train_ds = self._dataloader(
			train=True,
			**self._dataloader_args
		)

		custom_partitioner_function = self.train_ds.custom_partitioner_function if hasattr(self.train_ds, 'custom_partitioner_function') else None

		# Create data partitioner
		self.partitioner = DirichletDataPartitioner(
			dataset=self.train_ds,
			num_partitions=num_partitions,
			alpha=alpha,
			min_partition_size=min_partition_size,
			self_balancing=self_balancing,
			shuffle=shuffle,
			seed=seed,
			custom_partitioner_function=custom_partitioner_function
		)

		self._seed = seed

	@property
	def num_classes(self):
		return self.train_ds.num_classes

	@property
	def classes(self):
		return list(self.train_ds._classes.keys())

	def __getitem__(self, idx):
		return self.train_ds[idx]

	def __len__(self):
		return len(self.train_ds)

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		partition = self.partitioner.load_partition(partition_id)

		return partition

	def load_test_set(self) -> torch.utils.data.Dataset:
		test_ds = self._dataloader(
			train=False,
			**self._dataloader_args
		)
		return test_ds

	def get_collate_fn(self):
		return self._dataloader.custom_collate_fn


def available_datasets():
	return list(dataloaders.keys())


def load_data(name='coco-qa', num_partitions=10, min_num_samples=10, split='iid', seed=42, transform=None, global_args=None):
	assert name in dataloaders.keys(), 'Dataset `{}` is not available. Available datasets are `{}`'.format(name, available_datasets())

	if split == 'iid':
		alpha = 100000.0
	elif split == 'noniid':
		alpha = getattr(global_args, 'dirichlet_alpha', 0.1)
	elif isinstance(split, (int, float)):
		alpha = split
	else:
		raise ValueError('`split` can be either `iid`, `noniid`, `int` or `float`. Passed {} of type {}'.format(split, type(split)))

	module_name, class_name = dataloaders[name]
	ds = getattr(import_module(module_name), class_name)

	return DistributedDataset(
		dataloader=ds,
		transform=transform,
		alpha=alpha, # Note: alpha determines the data distribution among partitions
		num_partitions=num_partitions,
		min_partition_size=min_num_samples,
		self_balancing=True,
		shuffle=False,
		seed=seed,
		global_args=global_args
	)
