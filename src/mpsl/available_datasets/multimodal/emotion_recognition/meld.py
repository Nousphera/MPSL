"""meld experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import AudioTextDataset

NR_OF_CLASSES = 7


class MELD(AudioTextDataset):
    dataset_name = 'meld'
