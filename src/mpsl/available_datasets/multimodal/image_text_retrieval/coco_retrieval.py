"""coco-retrieval experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import RetrievalDataset

NR_OF_CAPTIONS_PER_IMAGE = 5
TRAIN_SET_LEN = 82783
TEST_SET_LEN = 5000


class CocoRetrieval(RetrievalDataset):
    dataset_name = 'coco-retrieval'
