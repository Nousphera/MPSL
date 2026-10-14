import torch
import torch.nn as nn


def compute_f_and_t_dim(patch_size, embed_dim, fstride, tstride, input_fdim, input_tdim):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)

    test_proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]

    return f_dim, t_dim


class AudioPatchEmbedding(nn.Module):
    """
    Audio to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=1, embed_dim=768, fstride=10, tstride=10, input_fdim=128, input_tdim=98, norm_layer=None, bias=True, flatten=True):
        """
        :param input_fdim: The num_mel_bins used when building the fbank operation.
        :param input_tdim: The time dimension. E.g with the UCF101 dataset, we use a timeframe of 1s which is translated to a tdim of 98
        """
        super().__init__()

        img_size = (img_size, img_size)

        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.img_size = img_size
        self.patch_size = patch_size

        f_dim, t_dim = compute_f_and_t_dim(self.patch_size, embed_dim, self.fstride, self.tstride, self.input_fdim, self.input_tdim)
        self.num_patches = f_dim * t_dim

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size, patch_size), stride=(fstride, tstride), bias=bias)

    def forward(self, x):
        if x.dim() == 3:
            # Audio embeddings originally do not include a number of channels but the PatchEmbed expects this to be included.
            B, H, W = x.shape
            x = x.reshape(B, 1, H, W)

        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)

        return x


class AudioTokenizer(nn.Module):

    def __init__(self, img_size, patch_size, embed_dim, bias=True, **kwargs):
        super().__init__()

        self.patch_embedding = AudioPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
            input_fdim=128,
            input_tdim=kwargs['input_tdim']
        )

    def forward(self, data):
        return self.patch_embedding(data)

    def get_nr_of_patches(self):
        return self.patch_embedding.num_patches
