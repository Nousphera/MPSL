"""ucf101 experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import AudioImageDataset

NR_OF_CLASSES = 51


class UCF101(AudioImageDataset):
    dataset_name = 'ucf101'
