"""coco-qa experiment loader; prepare acquired sources with scripts/prepare_data.py."""
from mpsl.available_datasets.prepared import ImageTextDataset

NR_OF_CLASSES = 430
MAX_NR_OF_UNIQUE_ANSWERS_FOR_COCO_QA = 430


class COCO_QA(ImageTextDataset):
    dataset_name = 'coco-qa'
