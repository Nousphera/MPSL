import copy
from enum import Enum

import torch
from torch import nn

from mpsl.available_datasets.multimodal.emotion_recognition.meld import NR_OF_CLASSES
from mpsl.models.meta_transformer.base.data2seq import Data2Seq, InputModality
from mpsl.models.meta_transformer.base.meta_transformer import MetaTransformerBase, \
    get_trainable_params_keys
from mpsl.models.meta_transformer.base.weights_utils import supported_audio_tokenizer_checkpoints, \
    get_audio_tokenizer_weights
from mpsl.utils.mpsl_utils import client_model_requires_any_grad


class FusionType(Enum):

    EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN = 'early_with_audio_cls_and_pooling_mean',
    LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN = 'late_with_audio_cls_and_pooling_mean'


AVAILABLE_TIME_DIMENSIONS = {
    '3.88s': {
            'audio_duration_in_sec': 3.88,
            'input_tdim': 766
        }
}

centralized_base_model = None


def _initialize_base_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes):
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
            device=device,
            fusion_type=fusion_type,
            audio_time_dimension=audio_time_dimension,
            num_classes=num_classes
        )


def get_centralized_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes, use_adapter_approach=False, include_text_adapter=False, include_audio_adapter=False, include_unified_adapter=False):
    if not use_adapter_approach:
        _initialize_base_model(
            use_pre_layer_norm=use_pre_layer_norm,
            use_post_layer_norm=use_post_layer_norm,
            use_large_encoder=use_large_encoder,
            freeze_encoder=freeze_encoder,
            nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
            trainable_params_key=trainable_params_key,
            use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
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
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes,
        include_text_adapter=include_text_adapter,
        include_audio_adapter=include_audio_adapter,
        include_unified_adapter=include_unified_adapter
    )


def get_split_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes):
    _initialize_base_model(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        freeze_encoder=freeze_encoder,
        nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune,
        trainable_params_key=trainable_params_key,
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes
    )

    _client_model = ClientModel(device)

    return _client_model, ServerModel(), client_model_requires_any_grad(_client_model)


def get_federated_model(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes, use_adapter_approach=False, include_text_adapter=False, include_audio_adapter=False, include_unified_adapter=False):
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
            fusion_type=fusion_type,
            audio_time_dimension=audio_time_dimension,
            num_classes=num_classes
        )

    return CentralizedModelWithAdapter(
        use_pre_layer_norm=use_pre_layer_norm,
        use_post_layer_norm=use_post_layer_norm,
        use_large_encoder=use_large_encoder,
        trainable_params_key=trainable_params_key,
        use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings,
        device=device,
        fusion_type=fusion_type,
        audio_time_dimension=audio_time_dimension,
        num_classes=num_classes,
        include_text_adapter=include_text_adapter,
        include_audio_adapter=include_audio_adapter,
        include_unified_adapter=include_unified_adapter
    )


class CentralizedModel(MetaTransformerBase):

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes=NR_OF_CLASSES, pretrained=True, encoder_name=None):
        super().__init__(num_classes=num_classes, trainable_params=get_trainable_params_keys('classifier_and_audio_tokenizer', trainable_params_key), use_pre_layer_norm=use_pre_layer_norm, use_post_layer_norm=use_post_layer_norm, use_large_encoder=use_large_encoder, freeze_encoder=freeze_encoder, nr_of_encoder_blocks_to_finetune=nr_of_encoder_blocks_to_finetune, pretrained=pretrained, encoder_name=encoder_name)

        self.device = device
        self.use_clip_encoder_for_text_embeddings = use_clip_encoder_for_text_embeddings

        self.audio_tokenizer_weights_model_name_key = 'audioset-0.4593'
        self.audio_tokenizer_weights_model_name = supported_audio_tokenizer_checkpoints[self.audio_tokenizer_weights_model_name_key]

        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_post_layer_norm = use_post_layer_norm

        self.fusion_type = FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN if fusion_type == 'default' else FusionType[fusion_type.upper()]

        self.input_tdim = AVAILABLE_TIME_DIMENSIONS['3.88s']['input_tdim']

        self.setup_tokenizers_for_fusion()

        if pretrained:
            self.pre_load_tokenizers_and_norm_layer_weights()
        self.apply_pre_loaded_weights()

    def pre_load_tokenizers_and_norm_layer_weights(self):
        tokenizer_weights = get_audio_tokenizer_weights(self.audio_tokenizer_weights_model_name_key, self.audio_tokenizer_weights_model_name)

        self._pre_loaded_model_weights['audio_tokenizer.class_embedding'] = tokenizer_weights['audio_spectrogram_transformer.embeddings.cls_token']
        self._pre_loaded_model_weights['audio_tokenizer.tokenizer.patch_embedding.proj.weight'] = tokenizer_weights['audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight']

    def switch_to_device(self, device):
        self.to(device)
        # Because the original text input cannot be converted to a Tensor, we are restricted to using the CPU until the text has been tokenized.
        self.text_tokenizer = self.text_tokenizer.to(torch.device('cpu'))

        return self

    def setup_tokenizers_for_fusion(self):
        match self.fusion_type:
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN | FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                self.text_tokenizer = Data2Seq(modality=InputModality.TEXT, embed_dim=self.embed_dim, use_clip_encoder_for_text_embeddings=self.use_clip_encoder_for_text_embeddings, use_large_encoder=self.use_large_encoder, device=self.device, pretrained=self.pretrained)
                self.audio_tokenizer = Data2Seq(modality=InputModality.AUDIO, input_tdim=self.input_tdim, img_size=224, patch_size=self.patch_size, embed_dim=self.embed_dim, bias=False, use_cls_embedding=True)

    def forward(self, x):
        audio, text = x[InputModality.AUDIO], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text = self.text_tokenizer(text)
        text = text.to(self.device)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        x = self.forward_with_fusion(audio, text)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x

    def forward_with_fusion(self, audio, text):
        batch_len = len(text)
        text = text.reshape(batch_len, 1, self.embed_dim)
        _, nr_of_audio_patches, _ = audio.shape

        match self.fusion_type:
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((audio, text), dim=1)

                x = self.pre_layer_norm(fused_embeddings)

                x = self.encoder(x)

                audio_cls_patch_embedding = self.to_latent(x[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the text encoding immediately follows after the last audio patch.
                text_embedding = self.to_latent(x[:, nr_of_audio_patches, :]).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                audio = self.pre_layer_norm(audio)
                text = self.pre_layer_norm(text)

                audio = self.encoder(audio)
                text = self.encoder(text)

                audio_cls_patch_embedding = self.to_latent(audio[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                text_embedding = self.to_latent(text[:, 0, :]).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x


class ClientModel(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.embed_dim = centralized_base_model.embed_dim
        self.fusion_type = centralized_base_model.fusion_type

        self.text_tokenizer = copy.deepcopy(centralized_base_model.text_tokenizer)
        self.audio_tokenizer = copy.deepcopy(centralized_base_model.audio_tokenizer)

        self.pre_layer_norm = copy.deepcopy(centralized_base_model.pre_layer_norm)

    def switch_to_device(self, device):
        self.to(device)
        self.text_tokenizer = self.text_tokenizer.to(torch.device('cpu'))

        return self

    def forward(self, x):
        audio, text = x[InputModality.AUDIO], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text = self.text_tokenizer(text)
        text = text.to(self.device)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        batch_len = len(text)
        text = text.reshape(batch_len, 1, self.embed_dim)

        match centralized_base_model.fusion_type:
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((audio, text), dim=1)

                return self.pre_layer_norm(fused_embeddings)
            case FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                return {
                    InputModality.AUDIO: self.pre_layer_norm(audio),
                    InputModality.TEXT: self.pre_layer_norm(text)
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
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                batch_len = x.size(0)
                nr_of_audio_patches = x.size(1) - 1
            case FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                audio, text = x[InputModality.AUDIO], x[InputModality.TEXT]

                batch_len = len(text)
                _, nr_of_audio_patches, _ = audio.shape

        match centralized_base_model.fusion_type:
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                x = self.encoder(x)

                audio_cls_patch_embedding = self.to_latent(x[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the text encoding immediately follows after the last audio patch.
                text_embedding = self.to_latent(x[:, nr_of_audio_patches, :]).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                audio = self.encoder(audio)
                text = self.encoder(text)

                audio_cls_patch_embedding = self.to_latent(audio[:, 0, :]).reshape(batch_len, 1, self.embed_dim)
                text_embedding = self.to_latent(text[:, 0, :]).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x


class CentralizedModelWithAdapter(CentralizedModel):
    """
    Implementation of the CentralizedModel, using the adapter technique used by the authors of FedCLIP (training an adapter in combination with a frozen encoder).
    """

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes, include_text_adapter, include_audio_adapter, include_unified_adapter):
        super().__init__(use_pre_layer_norm=use_pre_layer_norm, use_post_layer_norm=use_post_layer_norm, use_large_encoder=use_large_encoder, freeze_encoder=True, nr_of_encoder_blocks_to_finetune=-1, trainable_params_key=trainable_params_key, use_clip_encoder_for_text_embeddings=use_clip_encoder_for_text_embeddings, device=device, num_classes=num_classes, fusion_type=fusion_type, audio_time_dimension=audio_time_dimension)

        self.device = device

        self.include_text_adapter = include_text_adapter
        self.include_audio_adapter = include_audio_adapter
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
            if self.include_text_adapter:
                self.text_adapter = nn.Sequential(
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

        print('--- (Adapter approach) Previous printed enabled grads are overriden by: ---')

        # If we are using an attention-based adapter (FedCLIP approach), then we should override any global_args and freeze the encoder itself
        for name, param in self.named_parameters():
            if 'classifier' in name or 'audio_tokenizer' in name or 'adapter' in name:
                param.requires_grad = True
            elif 'encoder' in name:
                param.requires_grad = False

            if param.requires_grad:
                print(f'Enabled grad for {name}')

    def forward(self, x):
        audio, text = x[InputModality.AUDIO], x[InputModality.TEXT]

        # Intentionally first tokenizing the text before sending it to the other device, because text tokenization has to happen on the cpu since text input cannot be converted to a Tensor.
        text = self.text_tokenizer(text)
        text = text.to(self.device)

        audio = audio.to(self.device)
        audio = self.audio_tokenizer(audio)

        x = self.forward_with_fusion_and_adapter(audio, text)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x

    def include_adapter_attention(self, embedding, modality: InputModality):
        if self.include_unified_adapter:
            embedding = torch.mul(self.unified_adapter(embedding), embedding)
        elif modality == InputModality.TEXT and self.include_text_adapter:
            embedding = torch.mul(self.text_adapter(embedding), embedding)
        elif modality == InputModality.AUDIO and self.include_audio_adapter:
            embedding = torch.mul(self.audio_adapter(embedding), embedding)

        return embedding

    def forward_with_fusion_and_adapter(self, audio, text):
        batch_len = len(text)
        text = text.reshape(batch_len, 1, self.embed_dim)
        _, nr_of_audio_patches, _ = audio.shape

        match self.fusion_type:
            case FusionType.EARLY_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                fused_embeddings = torch.cat((audio, text), dim=1)

                x = self.pre_layer_norm(fused_embeddings)

                x = self.encoder(x)

                audio_cls_patch_embedding = self.to_latent(self.include_adapter_attention(x[:, 0, :], InputModality.AUDIO)).reshape(batch_len, 1, self.embed_dim)
                # Since we prepend the CLS patch for each modality, the text encoding immediately follows after the last audio patch.
                text_embedding = self.to_latent(self.include_adapter_attention(x[:, nr_of_audio_patches, :], InputModality.TEXT)).reshape(batch_len, 1, self.embed_dim)

                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x
            case FusionType.LATE_WITH_AUDIO_CLS_AND_POOLING_MEAN:
                audio = self.pre_layer_norm(audio)
                text = self.pre_layer_norm(text)

                audio = self.encoder(audio)
                text = self.encoder(text)

                audio_cls_patch_embedding = self.to_latent(self.include_adapter_attention(audio[:, 0, :], InputModality.AUDIO)).reshape(batch_len, 1, self.embed_dim)
                text_embedding = self.to_latent(self.include_adapter_attention(text[:, 0, :], InputModality.TEXT)).reshape(batch_len, 1, self.embed_dim)

                # Performing global average pooling to obtain the final representations of the fused modalities
                x = torch.cat((audio_cls_patch_embedding, text_embedding), dim=1)
                kernel_size = x.size(1)
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(kernel_size, 1)).reshape(batch_len, self.embed_dim)

                return x