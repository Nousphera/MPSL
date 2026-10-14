"""OpenAI CLIP text inputs using the maintained OpenCLIP package."""

import torch
from torch import nn

# Frozen external features do not belong to the federated model state.
_ENCODERS = {}


class TextTokenizer(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.desired_target_device = torch.device(kwargs["device"])
        self.use_clip_encoder_for_text_embeddings = kwargs[
            "use_clip_encoder_for_text_embeddings"
        ]

    def forward(self, text):
        return self.get_text_embeddings(text)

    def get_text_embeddings(self, text):
        import open_clip

        # Same BPE vocabulary, context 77 and truncation as OpenAI CLIP.
        tokens = open_clip.tokenize(text).to(self.desired_target_device)
        if self.use_clip_encoder_for_text_embeddings:
            name = "ViT-B-16" if self.embed_dim <= 768 else "ViT-L-14"
            key = (name, str(self.desired_target_device))
            if key not in _ENCODERS:
                encoder, _, _ = open_clip.create_model_and_transforms(
                    name,
                    pretrained="openai",
                    device=self.desired_target_device,
                    force_quick_gelu=True,
                )
                _ENCODERS[key] = encoder.eval().requires_grad_(False)
            with torch.no_grad():
                tokens = _ENCODERS[key].encode_text(tokens, normalize=False)
        return self.zero_padding(tokens, self.embed_dim)

    def zero_padding(self, values, target_dim):
        if target_dim < values.shape[1]:
            raise ValueError(
                "The target dimension is smaller than the text representation"
            )
        values = values.float()
        return torch.nn.functional.pad(values, (0, target_dim - values.shape[1]))
