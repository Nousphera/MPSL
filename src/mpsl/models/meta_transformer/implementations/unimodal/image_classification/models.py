import copy

from torch import nn

from mpsl.models.meta_transformer.base.data2seq import Data2Seq, InputModality
from mpsl.models.meta_transformer.base.meta_transformer import MetaTransformerBase, get_trainable_params_keys
from mpsl.models.meta_transformer.base.weights_utils import get_image_tokenizer_weights
from mpsl.utils.mpsl_utils import client_model_requires_any_grad

centralized_base_model = None


def _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device):
    global centralized_base_model

    if centralized_base_model is None:
        centralized_base_model = CentralizedModel(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            device=device
        )


def get_centralized_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device):
    _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device)

    return centralized_base_model


def get_split_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device):
    _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device)

    _client_model = ClientModel(device)

    return _client_model, ServerModel(), client_model_requires_any_grad(_client_model)


class CentralizedModel(MetaTransformerBase):

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device):
        super().__init__(num_classes=100, trainable_params=get_trainable_params_keys('classifier_and_image_tokenizer', trainable_params_key), use_pre_layer_norm=use_pre_layer_norm, use_post_layer_norm=use_post_layer_norm, use_large_encoder=use_large_encoder, freeze_encoder=freeze_encoder, nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune)

        self.device = device

        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_post_layer_norm = use_post_layer_norm
        self.image_tokenizer = Data2Seq(modality=InputModality.IMAGE, use_cls_embedding=True, img_size=224, patch_size=self.patch_size, embed_dim=self.embed_dim, bias=False)

        self.pre_load_tokenizer_and_norm_layer_weights()
        self.apply_pre_loaded_weights()

    def pre_load_tokenizer_and_norm_layer_weights(self):
        tokenizer_weights = get_image_tokenizer_weights(self.base_encoder_name)

        self._pre_loaded_model_weights['image_tokenizer.class_embedding'] = tokenizer_weights['class_embedding'].unsqueeze(0).unsqueeze(0)
        self._pre_loaded_model_weights['image_tokenizer.position_embedding'] = tokenizer_weights['positional_embedding'].unsqueeze(0)
        self._pre_loaded_model_weights['image_tokenizer.tokenizer.patch_embedding.proj.weight'] = tokenizer_weights['conv1.weight']

        if self.use_pre_layer_norm:
            self._pre_loaded_model_weights['pre_layer_norm.weight'] = tokenizer_weights['ln_pre.weight']
            self._pre_loaded_model_weights['pre_layer_norm.bias'] = tokenizer_weights['ln_pre.bias']

        if self.use_post_layer_norm:
            self._pre_loaded_model_weights['post_layer_norm.weight'] = tokenizer_weights['ln_post.weight']
            self._pre_loaded_model_weights['post_layer_norm.bias'] = tokenizer_weights['ln_post.bias']

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        x = x[InputModality.IMAGE]
        x = x.to(self.device)

        x = self.image_tokenizer(x)
        x = self.pre_layer_norm(x)

        x = self.encoder(x)

        # Extracting the CLS token's patch as output for classification
        x = self.to_latent(x[:, 0])

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x


class ClientModel(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

        self.image_tokenizer = copy.deepcopy(centralized_base_model.image_tokenizer)
        self.pre_layer_norm = copy.deepcopy(centralized_base_model.pre_layer_norm)

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        x = x[InputModality.IMAGE]
        x = x.to(self.device)

        x = self.image_tokenizer(x)
        x = self.pre_layer_norm(x)

        return x


class ServerModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Note that for the ServerModel, we do not need to copy from the centralized model instance, as there only will be a single ServerModel instance.
        self.encoder = centralized_base_model.encoder
        self.to_latent = centralized_base_model.to_latent
        self.post_layer_norm = centralized_base_model.post_layer_norm
        self.classifier = centralized_base_model.classifier

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        x = self.encoder(x)

        # Extracting the CLS token's patch as output for classification
        x = self.to_latent(x[:, 0])

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x
