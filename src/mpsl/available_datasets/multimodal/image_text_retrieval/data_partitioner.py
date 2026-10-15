import math
from random import Random, shuffle


def image_text_retrieval__data_partitioner(
    num_partitions, dataset_length, shuffle_indices=False, use_balancing=True, seed=None
):
    """Partition image/caption groups without dropping or duplicating an image.

    Balanced sizes differ by at most one; unbalanced mode uses consecutive
    ceil-sized chunks. Always return the requested IDs, with empty partitions
    if there are fewer images than clients. Experiment loaders separately
    require enough images for their minimum client size.
    """
    if num_partitions < 1 or dataset_length < 0:
        raise ValueError("Use a positive client count and nonnegative dataset length")
    indices = list(range(dataset_length))
    if shuffle_indices:
        if seed is None:
            shuffle(indices)
        else:
            Random(seed).shuffle(indices)
    size, remainder = divmod(dataset_length, num_partitions)
    parts = {}
    offset = 0
    for client_id in range(num_partitions):
        length = (
            size + (client_id < remainder)
            if use_balancing
            else math.ceil(dataset_length / num_partitions)
        )
        parts[client_id] = indices[offset : offset + length]
        offset += length
    return parts
