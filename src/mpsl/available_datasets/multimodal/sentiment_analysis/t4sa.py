"""t4sa experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import ImageTextDataset

NR_OF_CLASSES = 3


class T4SA(ImageTextDataset):
    dataset_name = 't4sa'
