import torch
from torch import nn
# Note: requires dependency `openai-clip`, not `clip`.
import clip

cpu_device = torch.device('cpu')

clip_model, clip_image_pre_process = None, None


class TextTokenizer(nn.Module):

    def __init__(self, embed_dim, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.desired_target_device = kwargs['device']
        self.use_clip_encoder_for_text_embeddings = kwargs['use_clip_encoder_for_text_embeddings']
        self.use_large_encoder = kwargs['use_large_encoder']
        self.kwargs = kwargs

    def forward(self, x):
        return self.get_text_embeddings(x)

    def get_text_embeddings(self, text):
        global clip_model, clip_image_pre_process

        if clip_model is None:
            # We'll import the module here to avoid circular dependencies.
            from mpsl.models.meta_transformer.base.meta_transformer import vit_base_encoder_options

            # We should use ViT-L/14 here for both the ViT-L/14 and ViT-H/14 base encoders.
            # Hence, we should only use ViT-B/16 if the required dimension is equal to or lower than 768 (as ViT-B/16 is the lowest encoder we can use here).
            base_encoder = 'ViT-B/16' if self.embed_dim <= vit_base_encoder_options['ViT-B/16']['embed_dim'] else 'ViT-L/14'
            clip_model, clip_image_pre_process = clip.load(base_encoder, cpu_device)
            clip_model = clip_model.to(self.desired_target_device)

        text_tensor = clip.tokenize(text, truncate=True)
        text_tensor = text_tensor.to(self.desired_target_device)

        if self.use_clip_encoder_for_text_embeddings:
            text_tensor = clip_model.encode_text(text_tensor)
            text_tensor = text_tensor.to(self.desired_target_device)

        # The original text_tensor is of shape [batch_size, 77] or [batch_size, 512] if merely tokenized or also encoded, respectively. We'll need to zero pad it to the required embed_dim.
        return self.zero_padding(text_tensor, self.embed_dim)

    def zero_padding(self, text_tensor, tar_dim):
        padding_size = tar_dim - text_tensor.shape[1]
        zero_tensor = torch.zeros((text_tensor.shape[0], padding_size), device=self.desired_target_device)
        padded_tensor = torch.cat([text_tensor, zero_tensor], dim=1)

        return padded_tensor
