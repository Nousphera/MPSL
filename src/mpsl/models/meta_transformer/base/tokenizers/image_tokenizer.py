import torch.nn as nn


class ImagePatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None, bias=True, flatten=True):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)

        return x


class ImageTokenizer(nn.Module):

    def __init__(self, img_size, patch_size, embed_dim, bias=True):
        super().__init__()

        self.patch_embedding = ImagePatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, bias=bias)

    def forward(self, data):
        return self.patch_embedding(data)

    def get_nr_of_patches(self):
        return self.patch_embedding.num_patches
