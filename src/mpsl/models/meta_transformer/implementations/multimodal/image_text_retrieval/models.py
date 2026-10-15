import copy

import torch
from torch import nn

from mpsl.models.meta_transformer.base.data2seq import Data2Seq, InputModality
from mpsl.models.meta_transformer.base.meta_transformer import MetaTransformerBase, get_trainable_params_keys
from mpsl.models.meta_transformer.base.weights_utils import get_image_tokenizer_weights
from mpsl.utils.mpsl_utils import client_model_requires_any_grad

centralized_base_model = None


def _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device):
    global centralized_base_model

    if centralized_base_model is None:
        centralized_base_model = CentralizedModel(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
            device=device
        )


def get_centralized_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, use_adapter_approach=False, include_image_adapter=False, include_text_adapter=False, include_unified_adapter=False):
    if not use_adapter_approach:
        _initialize_base_model(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
            device=device
        )

        return centralized_base_model

    return CentralizedModelWithAdapter(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        trainable_params_key=trainable_params_key,
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device,
        include_image_adapter=include_image_adapter,
        include_text_adapter=include_text_adapter,
        include_unified_adapter=include_unified_adapter
    )


def get_split_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device):
    _initialize_base_model(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        freeze_encoder=freeze_encoder,
        nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
        trainable_params_key=trainable_params_key,
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device
    )

    _client_model = ClientModel(device)

    return _client_model, ServerModel(), client_model_requires_any_grad(_client_model)


def get_federated_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, use_adapter_approach=False, include_image_adapter=False, include_text_adapter=False, include_unified_adapter=False):
    if not use_adapter_approach:
        return get_centralized_model(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
            device=device,
            use_adapter_approach=False
        )

    return CentralizedModelWithAdapter(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        trainable_params_key=trainable_params_key,
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device,
        include_image_adapter=include_image_adapter,
        include_text_adapter=include_text_adapter,
        include_unified_adapter=include_unified_adapter
    )


class CentralizedModel(MetaTransformerBase):

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, pretrained=True, encoder_name=None):
        super().__init__(num_classes=-1, trainable_params=get_trainable_params_keys('classifier_and_image_tokenizer', trainable_params_key), use_pre_layer_norm=use_pre_layer_norm, use_post_layer_norm=use_post_layer_norm, use_large_encoder=use_large_encoder, freeze_encoder=freeze_encoder, nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune, pretrained=pretrained, encoder_name=encoder_name)

        self.device = device

        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_post_layer_norm = use_post_layer_norm

        self.image_tokenizer = Data2Seq(modality=InputModality.IMAGE, img_size=224, patch_size=self.patch_size,
                                        embed_dim=self.embed_dim, bias=False, use_cls_embedding=True)
        self.text_tokenizer = Data2Seq(modality=InputModality.TEXT, embed_dim=self.embed_dim,
                                       use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
                                       use_large_encoder=use_large_encoder, device=device, pretrained=self.pretrained)

        if pretrained:
            self.pre_load_tokenizers_and_norm_layer_weights()
        self.apply_pre_loaded_weights()

    def pre_load_tokenizers_and_norm_layer_weights(self):
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
        # Because the original text input cannot be converted to a Tensor, we are restricted to using the CPU until the text has been tokenized.
        self.text_tokenizer = self.text_tokenizer.to(torch.device('cpu'))

        return self

    def forward(self, x):
        image, text = x[InputModality.IMAGE], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text_embeddings = []

        for image_captions in text:
            # the text input consists of an array of arrays, where each array contains one or multiple captions for a single image
            text_embeddings.append(self.text_tokenizer(image_captions).reshape(1, len(image_captions), self.embed_dim))
            # text_embeddings for each image are of shape [1, nr_of_captions_per_image, embed_dim]

        # The concatenated text_embeddings are now of shape [batch_size, nr_of_captions_per_image, embed_dim]
        text = torch.cat(text_embeddings)
        text = text.to(self.device)

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        text = self.pre_layer_norm(text)
        image = self.pre_layer_norm(image)

        image = self.encoder(image)
        text = self.encoder(text)

        image_cls_patch_embedding = self.to_latent(image[:, 0])

        image = self.post_layer_norm(image_cls_patch_embedding)
        text = self.post_layer_norm(text)

        return {
            InputModality.IMAGE: image,
            InputModality.TEXT: text
        }


class ClientModel(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.embed_dim = centralized_base_model.embed_dim

        self.image_tokenizer = copy.deepcopy(centralized_base_model.image_tokenizer)
        self.text_tokenizer = copy.deepcopy(centralized_base_model.text_tokenizer)

        self.pre_layer_norm = copy.deepcopy(centralized_base_model.pre_layer_norm)

    def switch_to_device(self, device):
        self.to(device)
        # Because the original text input cannot be converted to a Tensor, we are restricted to using the CPU until the text has been tokenized.
        self.text_tokenizer = self.text_tokenizer.to(torch.device('cpu'))

        return self

    def forward(self, x):
        image, text = x[InputModality.IMAGE], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text_embeddings = []

        for image_captions in text:
            # the text input consists of an array of arrays, where each array contains one or multiple captions for a single image
            text_embeddings.append(self.text_tokenizer(image_captions).reshape(1, len(image_captions), self.embed_dim))
            # text_embeddings for each image are of shape [1, nr_of_captions_per_image, embed_dim]

        # The concatenated text_embeddings are now of shape [batch_size, nr_of_captions_per_image, embed_dim]
        text = torch.cat(text_embeddings)
        text = text.to(self.device)

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        text = self.pre_layer_norm(text)
        image = self.pre_layer_norm(image)

        return {
            InputModality.IMAGE: image,
            InputModality.TEXT: text
        }


class ServerModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Note that for the ServerModel, we do not need to copy from the centralized model instance, as there only will be a single ServerModel instance.
        self.encoder = centralized_base_model.encoder
        self.to_latent = centralized_base_model.to_latent
        self.post_layer_norm = centralized_base_model.post_layer_norm

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        image, text = x[InputModality.IMAGE], x[InputModality.TEXT]

        image = self.encoder(image)
        text = self.encoder(text)

        image_cls_patch_embedding = self.to_latent(image[:, 0])

        image = self.post_layer_norm(image_cls_patch_embedding)
        text = self.post_layer_norm(text)

        return {
            InputModality.IMAGE: image,
            InputModality.TEXT: text
        }


class CentralizedModelWithAdapter(CentralizedModel):
    """
    Implementation of the CentralizedModel, using the adapter technique used by the authors of FedCLIP (training an adapter in combination with a frozen encoder).
    """

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings,
                 device, include_image_adapter, include_text_adapter, include_unified_adapter):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, True, -1,
                         trainable_params_key, use_clip_encoder_for_text_embeddings, device)

        self.device = device
        self.include_image_adapter = include_image_adapter
        self.include_text_adapter = include_text_adapter
        self.include_unified_adapter = include_unified_adapter

        if self.include_unified_adapter:
            self.unified_adapter = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Tanh(),
                nn.Linear(self.embed_dim, self.embed_dim),
                # Softmax is to ensure values to lie in the range of [0, 1], considering we will be multiplying the encoded features with the attention features.
                nn.Softmax(dim=1)
            )
        else:
            if self.include_image_adapter:
                self.image_adapter = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Tanh(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Softmax(dim=1)
                )

            if self.include_text_adapter:
                self.text_adapter = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Tanh(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Softmax(dim=1)
                )

        print('--- (Adapter approach) Previous printed enabled grads are overriden by: ---')

        # If we are using an attention-based adapter (FedCLIP approach), then we should override any global_args and freeze the encoder itself
        for name, param in self.named_parameters():
            if 'image_tokenizer' in name or 'adapter' in name:
                param.requires_grad = True
            elif 'encoder' in name:
                param.requires_grad = False

            if param.requires_grad:
                print(f'Enabled grad for {name}')

    def forward(self, x):
        image, text = x[InputModality.IMAGE], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text_embeddings = []

        for image_captions in text:
            # the text input consists of an array of arrays, where each array contains one or multiple captions for a single image
            text_embeddings.append(self.text_tokenizer(image_captions).reshape(1, len(image_captions), self.embed_dim))
            # text_embeddings for each image are of shape [1, nr_of_captions_per_image, embed_dim]

        # The concatenated text_embeddings are now of shape [batch_size, nr_of_captions_per_image, embed_dim]
        text = torch.cat(text_embeddings)
        text = text.to(self.device)

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        text = self.pre_layer_norm(text)
        image = self.pre_layer_norm(image)

        image = self.encoder(image)
        text = self.encoder(text)

        image_cls_patch_embedding = self.to_latent(self.include_adapter_attention(image[:, 0], InputModality.IMAGE))
        text = self.include_adapter_attention(text, InputModality.TEXT)

        image = self.post_layer_norm(image_cls_patch_embedding)
        text = self.post_layer_norm(text)

        return {
            InputModality.IMAGE: image,
            InputModality.TEXT: text
        }

    def include_adapter_attention(self, embedding, modality: InputModality):
        if self.include_unified_adapter:
            embedding = torch.mul(self.unified_adapter(embedding), embedding)
        elif modality == InputModality.IMAGE and self.include_image_adapter:
            embedding = torch.mul(self.image_adapter(embedding), embedding)
        elif modality == InputModality.TEXT and self.include_text_adapter:
            embedding = torch.mul(self.text_adapter(embedding), embedding)

        return embedding
    