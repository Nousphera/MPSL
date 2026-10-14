import os

from timm.models.vision_transformer import Block
from torch import nn

from mpsl.models.meta_transformer.base.weights_utils import get_encoder_weights

vit_base_encoder_options = {
    'ViT-Ti/16': {
        'embed_dim': 192,
        'nr_of_blocks': 12,
        'patch_size': 16,
        'num_heads': 3
    },
    'ViT-S/16': {
        'embed_dim': 384,
        'nr_of_blocks': 12,
        'patch_size': 16,
        'num_heads': 6
    },
    'ViT-B/16': {
        'embed_dim': 768,
        'nr_of_blocks': 12,
        'patch_size': 16,
        'num_heads': 12
    },
    'ViT-L/14': {
        'embed_dim': 1024,
        'nr_of_blocks': 24,
        'patch_size': 14,
        'num_heads': 16
    },
    'ViT-H/14': {
        'embed_dim': 1280,
        'nr_of_blocks': 32,
        'patch_size': 14,
        'num_heads': 16
    }
}

trainable_params_options = {
    'classifier_only': ['classifier.weight', 'classifier.bias'],
    'classifier_and_image_tokenizer': ['classifier.weight', 'classifier.bias', 'image_tokenizer.position_embedding', 'image_tokenizer.class_embedding', 'image_tokenizer.tokenizer.patch_embedding.proj.weight'],
    'classifier_and_audio_tokenizer': ['classifier.weight', 'classifier.bias',
                                       'audio_tokenizer.position_embedding', 'audio_tokenizer.class_embedding', 'audio_tokenizer.tokenizer.patch_embedding.proj.weight'],

    'classifier_and_image_tokenizer_and_audio_tokenizer': [
        'classifier.weight', 'classifier.bias',
        'image_tokenizer.position_embedding', 'image_tokenizer.class_embedding', 'image_tokenizer.tokenizer.patch_embedding.proj.weight',
        'audio_tokenizer.position_embedding', 'audio_tokenizer.class_embedding', 'audio_tokenizer.tokenizer.patch_embedding.proj.weight'],
}


def get_trainable_params_keys(default_key, chosen_key):
    return trainable_params_options[chosen_key] if chosen_key != 'default' else trainable_params_options[default_key]


class MetaTransformerBase(nn.Module):
    """
    This class is intended to be used as the base centralized model, which can be either stripped, split or extended by a separate CentralizedModel, ClientModel and ServerModel class, specific to a particular dataset.
    Instead of modifying this base instance, the respective CentralizedModel, ClientModel and ServerModel classes should be modified instead.
    """

    def __init__(
            self,
            num_classes,
            trainable_params=None,
            use_large_encoder=False,
            freeze_encoder=True,
            nr_of_encoder_blocks_to_finetune=-1,
            use_pre_layer_norm=False,
            use_post_layer_norm=False):
        """
        Uses both common CLS token and learned position embedding approaches. Position embeddings are used to encode the relative position of a patch, whereas
        the CLS token is used as an aggregated representation of all patches.

        Implementation is inspired by Meta-Transformer's implementation in their Hyper-spectrum and Video modules (Hyper-spectrum>metatransformer.py & Video>models>modelin_finetune.py).

        Note that the implementation expects the input images to already be preprocessed accordingly.
        """
        super().__init__()

        # Default to solely training the classifier. Note that this is the default of the base class and not the specific model implementation.
        if trainable_params is None:
            trainable_params = trainable_params_options['classifier_only']

        # While Meta-Transformer originally supports ViT-B/16 and ViT-L/14 as its base and large encoder backbone, respectively, we include the possibility for using different ViT encoder backbones.
        # This should explicitly be enabled by using global env variables.
        should_override_base_encoder_setting = 'VIT_BASE' in os.environ and os.environ['VIT_BASE'] != 'None'
        if should_override_base_encoder_setting:
            self.base_encoder_name = os.environ['VIT_BASE']

            # We need to override the use_large_encoder parameter to ensure that it will be set accordingly, as it might be used in other places.
            use_large_encoder = os.environ['VIT_BASE'] == 'ViT-L/14'
        else:
            self.base_encoder_name = 'ViT-L/14' if use_large_encoder else 'ViT-B/16'

        base_encoder = vit_base_encoder_options[self.base_encoder_name]

        self.num_classes = num_classes
        self.use_large_encoder = use_large_encoder
        self.embed_dim = base_encoder['embed_dim']
        self.patch_size = base_encoder['patch_size']
        self.nr_of_encoder_blocks = base_encoder['nr_of_blocks']
        self.freeze_encoder = freeze_encoder
        self.nr_of_last_encoder_blocks_to_finetune = nr_of_encoder_blocks_to_finetune if nr_of_encoder_blocks_to_finetune != -1 else (0 if self.freeze_encoder else self.nr_of_encoder_blocks)
        self.trainable_params = trainable_params
        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_post_layer_norm = use_post_layer_norm

        encoder = nn.Sequential(*[
            Block(
                dim=self.embed_dim,
                num_heads=base_encoder['num_heads'],
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(base_encoder['nr_of_blocks'])])

        self.pre_layer_norm = nn.LayerNorm(self.embed_dim) if use_pre_layer_norm else nn.Identity()

        self.encoder = encoder

        # Used for classification, to extract the cls_token's patch/output
        self.to_latent = nn.Identity()

        self.post_layer_norm = nn.LayerNorm(self.embed_dim) if use_post_layer_norm else nn.Identity()

        if num_classes > 0:
            self.classifier = nn.Linear(self.embed_dim, num_classes)

        self._pre_loaded_model_weights = None
        self.pre_load_encoder_weights(use_large_encoder=use_large_encoder)

    def pre_load_encoder_weights(self, use_large_encoder):
        self._pre_loaded_model_weights = get_encoder_weights(self.state_dict(), use_large_encoder=use_large_encoder, include_classifier=self.num_classes > 0)

    def apply_pre_loaded_weights(self):
        load_result = self.load_state_dict(self._pre_loaded_model_weights, strict=False)
        print(f'Applying model weights result: {load_result}')

        # We no longer require the weights, so we can free the used memory.
        self._pre_loaded_model_weights = None

        earliest_block_idx_to_finetune = self.nr_of_encoder_blocks - self.nr_of_last_encoder_blocks_to_finetune

        finetune_attn_layers, finetune_mlp_layers, finetune_norm_layers = True, True, True

        for name, param in self.named_parameters():
            if 'encoder' in name and self.nr_of_last_encoder_blocks_to_finetune != -1:
                # Parses '9' from e.g. encoder.9.attn.qkv.bias
                block_idx = int(name.split('encoder.')[1].split('.')[0])

                requires_grad = False

                if block_idx >= earliest_block_idx_to_finetune:
                    if 'attn' in name and finetune_attn_layers:
                        requires_grad = True
                    elif 'mlp' in name and finetune_mlp_layers:
                        requires_grad = True
                    elif 'norm' in name and finetune_norm_layers:
                        requires_grad = True

                param.requires_grad = requires_grad

            elif ('encoder' in name and self.freeze_encoder) or ('encoder' not in name and name not in self.trainable_params):
                param.requires_grad = False

            if ('pre_layer_norm' in name and self.use_pre_layer_norm) or ('post_layer_norm' in name and self.use_post_layer_norm):
                param.requires_grad = True

            if param.requires_grad:
                print(f'Enabled grad for {name}')

    def forward(self, x):
        return NotImplementedError('Forward of MetaTransformerBase is not implemented: a dataset-specific implementation should be implemented instead.')

    def __str__(self):
        return self.__class__.__name__
