from enum import Enum

import torch
import torch.nn as nn

from einops import repeat

from mpsl.models.meta_transformer.base.tokenizers.audio_tokenizer import AudioTokenizer
from mpsl.models.meta_transformer.base.tokenizers.image_tokenizer import ImageTokenizer
from mpsl.models.meta_transformer.base.tokenizers.text_tokenizer import TextTokenizer


from mpsl.models.modalities import InputModality


class Data2Seq(nn.Module):

    def __init__(self, modality: InputModality, use_cls_embedding=False, **kwargs):
        super().__init__()

        self.use_cls_token = use_cls_embedding

        if modality == InputModality.IMAGE:
            self.tokenizer = ImageTokenizer(**kwargs)
            self.use_position_embedding = True
            self.nr_of_patches = self.tokenizer.get_nr_of_patches()
        elif modality == InputModality.TEXT:
            self.tokenizer = TextTokenizer(**kwargs)
            self.use_position_embedding = False
        elif modality == InputModality.AUDIO:
            self.tokenizer = AudioTokenizer(**kwargs)
            self.use_position_embedding = True
            self.nr_of_patches = self.tokenizer.get_nr_of_patches()

        if self.use_position_embedding:
            self.position_embedding = nn.Parameter(torch.randn(1, self.nr_of_patches + 1, kwargs['embed_dim']))

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(1, 1, kwargs['embed_dim']))

    def forward(self, x):
        x = self.tokenizer(x)

        if self.use_position_embedding:
            batch_len, num_patches, embed_dim = x.shape
        else:
            batch_len, embed_dim = x.shape

        if self.use_cls_token:
            # Add learnable class (CLS) embedding
            cls_tokens = repeat(self.class_embedding, '() n d -> b n d', b=batch_len)  # [batch_len, 1, embed_dim]
            x = torch.cat((cls_tokens, x), dim=1)

        # Add learnable positional embedding
        if self.use_position_embedding:
            num_elements = num_patches + 1 if self.use_cls_token else num_patches
            x = x + self.position_embedding[:, :num_elements]

        return x
