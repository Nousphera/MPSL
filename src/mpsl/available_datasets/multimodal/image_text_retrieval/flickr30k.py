"""flickr30k experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import RetrievalDataset

NR_OF_CAPTIONS_PER_IMAGE = 5
TEST_SET_LEN = 1000


class Flickr30K(RetrievalDataset):
    dataset_name = 'flickr30k'
