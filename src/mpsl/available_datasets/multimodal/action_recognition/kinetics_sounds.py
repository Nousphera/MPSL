"""kinetics-sounds experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import AudioImageDataset

NR_OF_CLASSES = 30


class KineticsSounds(AudioImageDataset):
    dataset_name = 'kinetics-sounds'
