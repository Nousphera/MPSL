import math
from random import shuffle

import typing


def image_text_retrieval__data_partitioner(num_partitions, dataset_length, shuffle_indices=False, use_balancing=True):
    """
    This task requires a custom data partitioner because there are no classes. We have 29_000 images that all occur once, each with 5 text captions.
    Hence, there is no possibility for an iid distribution; we will always be partitioning the samples over all clients at random.
    """
    assert num_partitions > 0

    all_indices = [x for x in range(dataset_length)]

    if shuffle_indices:
        shuffle(all_indices)

    partition_id_to_indices: typing.Dict[int, typing.List[int]] = dict()
    nr_of_indices_per_partition = math.floor(dataset_length / num_partitions) if use_balancing else math.ceil(dataset_length / num_partitions)

    nr_of_remaining_indices_to_balance = dataset_length % nr_of_indices_per_partition
    balancing_idx_to_use = dataset_length - 1
    nr_of_balanced_indices = 0

    current_partition_indices = []
    current_partition_id = 0
    current_partition_length = 0

    for element_nr, idx in enumerate(all_indices):
        if use_balancing and element_nr == dataset_length - nr_of_remaining_indices_to_balance:
            break

        current_partition_indices.append(idx)
        current_partition_length += 1

        if current_partition_length == nr_of_indices_per_partition:
            if use_balancing and nr_of_balanced_indices < nr_of_remaining_indices_to_balance:
                current_partition_indices.append(all_indices[balancing_idx_to_use])
                balancing_idx_to_use -= 1
                nr_of_balanced_indices += 1

            partition_id_to_indices[current_partition_id] = current_partition_indices

            current_partition_id += 1
            current_partition_length = 0
            current_partition_indices = []

    # Assigning the indices of the last partition, as the loop will have ended prior to assigning it
    if len(current_partition_indices) > 0:
        partition_id_to_indices[current_partition_id] = current_partition_indices

    return partition_id_to_indices
