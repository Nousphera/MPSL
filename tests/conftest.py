import gc
import pytest
import torch


@pytest.fixture(autouse=True)
def bounded_cpu_execution():
    torch.set_num_threads(1)
    yield
    from mpsl.models.meta_transformer.base.tokenizers.text_tokenizer import _ENCODERS

    _ENCODERS.clear()
    gc.collect()
