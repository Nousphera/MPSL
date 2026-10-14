import copy
from enum import Enum

import torch
from torch import nn

from mpsl.models.meta_transformer.base.data2seq import Data2Seq, InputModality
from mpsl.models.meta_transformer.base.meta_transformer import MetaTransformerBase, \
    get_trainable_params_keys
from mpsl.models.meta_transformer.base.weights_utils import supported_audio_tokenizer_checkpoints, \
    get_image_tokenizer_weights, get_audio_tokenizer_weights
from mpsl.utils.mpsl_utils import client_model_requires_any_grad


class FusionType(Enum):

    EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN = 'early_with_both_cls_and_pooling_mean',
    LATE_WITH_BOTH_CLS_AND_POOLING_MEAN = 'late_with_both_cls_and_pooling_mean',


AVAILABLE_TIME_DIMENSIONS = {
    '10.24s': {
        'audio_duration_in_sec': 10.255,
        'input_tdim': 1024
    },
    '1s': {
        'audio_duration_in_sec': 1,
        'input_tdim': 98
    }
}

centralized_base_model = None


def _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes):
    global centralized_base_model

    if centralized_base_model is None:
        centralized_base_model = CentralizedModel(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            device=device,
            fusion_type=fusion_type,
            audio_time_dimension=audio_time_dimension,
            num_classes=num_classes
        )


def get_centralized_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes, use_adapter_approach=False, include_image_adapter=False, include_audio_adapter=False, include_unified_adapter=False):
    if not use_adapter_approach:
        _initialize_base_model(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            device=device,
            fusion_type=fusion_type,
            audio_time_dimension=audio_time_dimension,
            num_classes=num_classes
        )

        return centralized_base_model

    return CentralizedModelWithAdapter(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        trainable_params_key=trainable_params_key,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes,
        include_image_adapter=include_image_adapter,
        include_audio_adapter=include_audio_adapter,
        include_unified_adapter=include_unified_adapter
    )


def get_split_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes):
    _initialize_base_model(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        freeze_encoder=freeze_encoder,
        nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
        trainable_params_key=trainable_params_key,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes
    )

    _client_model = ClientModel(device)

    return _client_model, ServerModel(), client_model_requires_any_grad(_client_model)


def get_federated_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes, use_adapter_approach=False, include_image_adapter=False, include_audio_adapter=False, include_unified_adapter=False):
    if not use_adapter_approach:
        return get_centralized_model(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            device=device,
            fusion_type=fusion_type,
            audio_time_dimension=audio_time_dimension,
            num_classes=num_classes,
            use_adapter_approach=use_adapter_approach
        )

    return CentralizedModelWithAdapter(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        trainable_params_key=trainable_params_key,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes,
        include_image_adapter=include_image_adapter,
        include_audio_adapter=include_audio_adapter,
        include_unified_adapter=include_unified_adapter
    )


class CentralizedModel(MetaTransformerBase):

    def __init__(
            self,
            use_pre_layer_norm,
            use_post_layer_norm,
            use_large_encoder,
            freeze_encoder,
            nr_of_encoder_blocks_to_finetune,
            trainable_params_key,
            device,
            fusion_type,
            audio_time_dimension,
            num_classes):
        super().__init__(num_classes=num_classes, trainable_params=get_trainable_params_keys('classifier_and_image_tokenizer_and_audio_tokenizer', trainable_params_key), use_pre_layer_norm=use_pre_layer_norm, use_post_layer_norm=use_post_layer_norm, use_large_encoder=use_large_encoder, freeze_encoder=freeze_encoder, nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune)

        self.device = device

        self.audio_tokenizer_weights_model_name_key = 'audioset-0.4593'
        self.audio_tokenizer_weights_model_name = supported_audio_tokenizer_checkpoints[self.audio_tokenizer_weights_model_name_key]

        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_post_layer_norm = use_post_layer_norm

        self.fusion_type = FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN if fusion_type == 'default' else FusionType[fusion_type.upper()]

        if audio_time_dimension not in AVAILABLE_TIME_DIMENSIONS:
            raise Exception(f'Incorrect time dimension has been chosen ({audio_time_dimension}). The available options are: {AVAILABLE_TIME_DIMENSIONS.keys()}')

        self.input_tdim = AVAILABLE_TIME_DIMENSIONS[audio_time_dimension]['input_tdim']
        self.nr_of_visual_patches = -1

        self.setup_tokenizers_for_fusion()

        self.pre_load_tokenizers_and_norm_layer_weights()
        self.apply_pre_loaded_weights()

    def pre_load_tokenizers_and_norm_layer_weights(self):
        # ====== Image tokenizer weights ======
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

        # ====== Audio tokenizer weights ======
        tokenizer_weights = get_audio_tokenizer_weights(self.audio_tokenizer_weights_model_name_key, self.audio_tokenizer_weights_model_name)

        self._pre_loaded_model_weights['audio_tokenizer.class_embedding'] = tokenizer_weights['audio_spectrogram_transformer.embeddings.cls_token']
        # AST uses a timeframe of 10.24s instead of the time dimensions that we support. Shapes of the positional embeddings therefore mismatch, and we'll refrain from loading them.
        # self._pre_loaded_model_weights['audio_tokenizer.position_embedding'] = tokenizer_weights['audio_spectrogram_transformer.embeddings.position_embeddings'][:, :1213]
        self._pre_loaded_model_weights['audio_tokenizer.tokenizer.patch_embedding.proj.weight'] = tokenizer_weights['audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight']

    def switch_to_device(self, device):
        self.to(device)

        return self

    def setup_tokenizers_for_fusion(self):
        match self.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN | FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                self.image_tokenizer = Data2Seq(modality=InputModality.IMAGE, img_size=224, patch_size=self.patch_size, embed_dim=self.embed_dim, bias=False, use_cls_embedding=True)
                self.audio_tokenizer = Data2Seq(modality=InputModality.AUDIO, input_tdim=self.input_tdim, img_size=224, patch_size=self.patch_size, embed_dim=self.embed_dim, bias=False, use_cls_embedding=True)

        b, nr_of_patches, embed_dim = self.image_tokenizer.position_embedding.shape
        self.nr_of_visual_patches = nr_of_patches

    def forward(self, x):
        image, audio = x[InputModality.IMAGE], x[InputModality.AUDIO]

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        x = self.forward_with_fusion(image, audio)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x

    def forward_with_fusion(self, image, audio):
        batch_len = len(audio)

        match self.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((image, audio), dim=1)

                x = self.pre_layer_norm(fused_embeddings)
                x = self.encoder(x)

                image_cls_patch_embedding = self.to_latent(x[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the CLS patch for audio immediately follows after the last visual patch.
                audio_cls_patch_embedding = self.to_latent(x[:, self.nr_of_visual_patches, :]).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                image = self.pre_layer_norm(image)
                audio = self.pre_layer_norm(audio)

                image = self.encoder(image)
                audio = self.encoder(audio)

                image_cls_patch_embedding = self.to_latent(image[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                audio_cls_patch_embedding = self.to_latent(audio[:, 0, :]).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x


class ClientModel(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.embed_dim = centralized_base_model.embed_dim
        self.fusion_type = centralized_base_model.fusion_type

        self.image_tokenizer = copy.deepcopy(centralized_base_model.image_tokenizer)
        self.audio_tokenizer = copy.deepcopy(centralized_base_model.audio_tokenizer)

        self.pre_layer_norm = copy.deepcopy(centralized_base_model.pre_layer_norm)

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        image, audio = x[InputModality.IMAGE], x[InputModality.AUDIO]

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        match centralized_base_model.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((image, audio), dim=1)

                return self.pre_layer_norm(fused_embeddings)
            case FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                return {
                    InputModality.IMAGE: self.pre_layer_norm(image),
                    InputModality.AUDIO: self.pre_layer_norm(audio)
                }


class ServerModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Note that for the ServerModel, we do not need to copy from the centralized model instance, as there only will be a single ServerModel instance.
        self.embed_dim = centralized_base_model.embed_dim

        self.encoder = centralized_base_model.encoder
        self.to_latent = centralized_base_model.to_latent
        self.post_layer_norm = centralized_base_model.post_layer_norm
        self.classifier = centralized_base_model.classifier

    def switch_to_device(self, device):
        self.to(device)

        return self

    def forward(self, x):
        x = self.forward_with_fusion(x)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x

    def forward_with_fusion(self, x):
        batch_len = -1

        match centralized_base_model.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN:
                batch_len = x.size(0)
            case FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                image, audio = x[InputModality.IMAGE], x[InputModality.AUDIO]

                batch_len = len(audio)

        match centralized_base_model.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN:
                x = self.encoder(x)

                image_cls_patch_embedding = self.to_latent(x[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the CLS patch for audio immediately follows after the last visual patch.
                audio_cls_patch_embedding = self.to_latent(x[:, centralized_base_model.nr_of_visual_patches, :]).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                image = self.encoder(image)
                audio = self.encoder(audio)

                image_cls_patch_embedding = self.to_latent(image[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                audio_cls_patch_embedding = self.to_latent(audio[:, 0, :]).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x


class CentralizedModelWithAdapter(CentralizedModel):
    """
    Implementation of the CentralizedModel, using the adapter technique used by the authors of FedCLIP (training an adapter in combination with a frozen encoder).
    """

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes, include_image_adapter, include_audio_adapter, include_unified_adapter):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, True, -1, trainable_params_key, device, fusion_type, audio_time_dimension, num_classes=num_classes)

        self.device = device

        self.include_image_adapter = include_image_adapter
        self.include_audio_adapter = include_audio_adapter
        self.include_unified_adapter = include_unified_adapter

        if self.include_unified_adapter:
            self.unified_adapter = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Tanh(),
                nn.Linear(self.embed_dim, self.embed_dim),
                # Softmax is to ensure values lie in the range of [0, 1], which is needed considering we will be multiplying the encoded features with the attention features.
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

            if self.include_audio_adapter:
                self.audio_adapter = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Tanh(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Softmax(dim=1)
                )

        print('--- (Adapter approach) Previous printed enabled grads are overridden by: ---')

        # If we are using an attention-based adapter (FedCLIP approach), then we should override any global_args and freeze the encoder itself
        for name, param in self.named_parameters():
            if 'classifier' in name or 'image_tokenizer' in name or 'audio_tokenizer' in name or 'adapter' in name:
                param.requires_grad = True
            elif 'encoder' in name:
                param.requires_grad = False

            if param.requires_grad:
                print(f'Enabled grad for {name}')

    def forward(self, x):
        image, audio = x[InputModality.IMAGE], x[InputModality.AUDIO]

        image = image.to(self.device)
        image = self.image_tokenizer(image)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        x = self.forward_with_fusion_and_adapter(image, audio)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x

    def include_adapter_attention(self, embedding, modality: InputModality):
        if self.include_unified_adapter:
            embedding = torch.mul(self.unified_adapter(embedding), embedding)
        elif modality == InputModality.IMAGE and self.include_image_adapter:
            embedding = torch.mul(self.image_adapter(embedding), embedding)
        elif modality == InputModality.AUDIO and self.include_audio_adapter:
            embedding = torch.mul(self.audio_adapter(embedding), embedding)

        return embedding

    def forward_with_fusion_and_adapter(self, image, audio):
        batch_len = len(audio)

        match self.fusion_type:
            case FusionType.EARLY_WITH_BOTH_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((image, audio), dim=1)

                x = self.pre_layer_norm(fused_embeddings)

                x = self.encoder(x)

                image_cls_patch_embedding = self.to_latent(self.include_adapter_attention(x[:, 0, :], InputModality.IMAGE)).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the CLS patch for audio immediately follows after the last visual patch.
                audio_cls_patch_embedding = self.to_latent(self.include_adapter_attention(x[:, self.nr_of_visual_patches, :], InputModality.AUDIO)).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_BOTH_CLS_AND_POOLING_MEAN:
                image = self.pre_layer_norm(image)
                audio = self.pre_layer_norm(audio)

                image = self.encoder(image)
                audio = self.encoder(audio)

                image_cls_patch_embedding = self.to_latent(self.include_adapter_attention(image[:, 0, :], InputModality.IMAGE)).reshape(batch_len, 1, self.embed_dim)
                audio_cls_patch_embedding = self.to_latent(self.include_adapter_attention(audio[:, 0, :], InputModality.AUDIO)).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((image_cls_patch_embedding, audio_cls_patch_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x