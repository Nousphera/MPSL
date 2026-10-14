import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

from mpsl import available_datasets as datasets
from mpsl import models
import mpsl.models.meta_transformer.implementations.multimodal.emotion_recognition.models
import mpsl.models.meta_transformer.implementations.multimodal.image_text_retrieval.models
import mpsl.models.meta_transformer.implementations.multimodal.vqa.models
from mpsl.main_centralized import setup_arguments
from mpsl.models import InputModality, get_centralized_model_and_trainer
from mpsl.utils.config_utils import set_random_seed
from mpsl.utils.cuda_utils import get_free_cuda_device_name


class VQAWrapper(models.meta_transformer.implementations.multimodal.vqa.models.CentralizedModel):
    """
    Model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, num_classes):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, num_classes=num_classes)

    def forward(self, *x):
        raw_image, text_embedding = x

        raw_image = raw_image.to(self.device)
        image_embedding = self.image_tokenizer(raw_image)

        x = self.forward_with_fusion(text_embedding, image_embedding)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x


class VQAWrapperWithAdapter(models.meta_transformer.implementations.multimodal.vqa.models.CentralizedModelWithAdapter):
    """
    Adapter model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """

    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, include_image_adapter, include_text_adapter, include_unified_adapter):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, include_image_adapter, include_text_adapter, include_unified_adapter)

    def forward(self, *x):
        raw_image, text_embedding = x

        raw_image = raw_image.to(self.device)
        image_embedding = self.image_tokenizer(raw_image)

        x = self.forward_with_fusion_and_adapter(text_embedding, image_embedding)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x


class MELDWrapper(models.meta_transformer.implementations.multimodal.emotion_recognition.models.CentralizedModel):
    """
    Model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """
    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes)

    def forward(self, *x):
        raw_audio, text_embedding = x

        raw_audio = raw_audio.to(self.device)
        audio = self.audio_tokenizer(raw_audio)

        x = self.forward_with_fusion(audio, text_embedding)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x


class MELDWrapperWithAdapter(models.meta_transformer.implementations.multimodal.emotion_recognition.models.CentralizedModelWithAdapter):
    """
    Adapter model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """
    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes, include_text_adapter, include_audio_adapter, include_unified_adapter):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, trainable_params_key, use_clip_encoder_for_text_embeddings, device, fusion_type, audio_time_dimension, num_classes, include_text_adapter, include_audio_adapter, include_unified_adapter)

    def forward(self, *x):
        raw_audio, text_embedding = x

        raw_audio = raw_audio.to(self.device)
        audio = self.audio_tokenizer(raw_audio)

        x = self.forward_with_fusion_and_adapter(audio, text_embedding)

        x = self.post_layer_norm(x)
        x = self.classifier(x)

        return x


class ImageTextRetrievalWrapper(models.meta_transformer.implementations.multimodal.image_text_retrieval.models.CentralizedModel):
    """
    Model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """
    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, freeze_encoder, nr_of_encoder_blocks_to_finetune, trainable_params_key, use_clip_encoder_for_text_embeddings, device)

    def forward(self, *x):
        raw_image, text_embedding = x

        raw_image = raw_image.to(self.device)
        image_embedding = self.image_tokenizer(raw_image)

        text = self.pre_layer_norm(text_embedding)
        image = self.pre_layer_norm(image_embedding)

        image = self.encoder(image)
        text = self.encoder(text)

        image_cls_patch_embedding = self.to_latent(image[:, 0])

        image = self.post_layer_norm(image_cls_patch_embedding)
        text = self.post_layer_norm(text)

        return image, text


class ImageTextRetrievalWrapperWithAdapter(models.meta_transformer.implementations.multimodal.image_text_retrieval.models.CentralizedModelWithAdapter):
    """
    Adapter model implementation that skips text tokenization, and also accepts *x as input instead of x.
    Both of which are needed to support profiling FLOPs via fvcore for models that use the text modality.
    """
    def __init__(self, use_pre_layer_norm, use_post_layer_norm, use_large_encoder, use_clip_encoder_for_text_embeddings, device, args):
        super().__init__(use_pre_layer_norm, use_post_layer_norm, use_large_encoder, use_clip_encoder_for_text_embeddings, device, args)

    def forward(self, *x):
        raw_image, text_embedding = x

        raw_image = raw_image.to(self.device)
        image = self.image_tokenizer(raw_image)

        text = self.pre_layer_norm(text_embedding)
        image = self.pre_layer_norm(image)

        image = self.encoder(image)
        text = self.encoder(text)

        image_cls_patch_embedding = self.to_latent(self.include_adapter_attention(image[:, 0], InputModality.IMAGE))
        text = self.include_adapter_attention(text, InputModality.TEXT)

        image = self.post_layer_norm(image_cls_patch_embedding)
        text = self.post_layer_norm(text)

        return image, text


def should_override_text_tokenization(dataset):
    """Returns True for any dataset that uses the text modality"""

    return dataset in ['meld', 'coco-retrieval', 'flickr30k', 't4sa', 'coco-qa']


def embed_text(model, text):
    text = model.text_tokenizer(text)
    text = text.to(model.device)

    return text


def embed_text_for_image_text_retrieval(model, text):
    text_embeddings = []

    for image_captions in text:
        text_embeddings.append(model.text_tokenizer(image_captions).reshape(1, len(image_captions), model.embed_dim))

    text = torch.cat(text_embeddings)
    text = text.to(model.device)

    return text


def build_model_wrapper(global_args):
    dataset = global_args.dataset

    # Note that we're solely evaluating the centralized version of each model implementation, as we can deduce the number of
    # parameters and GFlops for the Split Learning counterpart from the result as well, by reasoning about what layers are present on the client-side and server-side.
    if dataset == 'coco-qa' or dataset == 't4sa':
        if dataset == 'coco-qa':
            from mpsl.available_datasets.multimodal.vqa.coco_qa import NR_OF_CLASSES
        else:
            from mpsl.available_datasets.multimodal.sentiment_analysis.t4sa import NR_OF_CLASSES

        if global_args.use_adapter_approach:
            return VQAWrapperWithAdapter(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.trainable_params_key,
                global_args.use_clip_encoder_for_text_embeddings,
                device,
                global_args.fusion_type,
                global_args.include_image_adapter,
                global_args.include_text_adapter,
                global_args.include_unified_adapter
            ), InputModality.IMAGE
        else:
            return VQAWrapper(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.freeze_encoder,
                global_args.nr_of_last_encoder_blocks_to_finetune,
                global_args.trainable_params_key,
                global_args.use_clip_encoder_for_text_embeddings,
                device,
                global_args.fusion_type,
                NR_OF_CLASSES
            ), InputModality.IMAGE
    elif dataset == 'meld':
        from mpsl.available_datasets.multimodal.emotion_recognition.meld import NR_OF_CLASSES

        if global_args.use_adapter_approach:
            return MELDWrapperWithAdapter(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.trainable_params_key,
                global_args.use_clip_encoder_for_text_embeddings,
                device,
                global_args.fusion_type,
                global_args.audio_time_dimension,
                NR_OF_CLASSES,
                global_args.include_text_adapter,
                global_args.include_audio_adapter,
                global_args.include_unified_adapter
            ), InputModality.AUDIO
        else:
            return MELDWrapper(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.freeze_encoder,
                global_args.nr_of_last_encoder_blocks_to_finetune,
                global_args.trainable_params_key,
                global_args.use_clip_encoder_for_text_embeddings,
                device,
                global_args.fusion_type,
                global_args.audio_time_dimension,
                NR_OF_CLASSES
            ), InputModality.AUDIO
    elif dataset == 'flickr30k' or dataset == 'coco-retrieval':
        if global_args.use_adapter_approach:
            return ImageTextRetrievalWrapperWithAdapter(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.use_clip_encoder_for_text_embeddings,
                device,
                global_args
            ), InputModality.IMAGE
        else:
            return ImageTextRetrievalWrapper(
                global_args.use_pre_layer_norm,
                global_args.use_post_layer_norm,
                global_args.use_large_encoder,
                global_args.freeze_encoder,
                global_args.nr_of_last_encoder_blocks_to_finetune,
                global_args.trainable_params_key,
                global_args.use_clip_encoder_for_text_embeddings,
                device
            ), InputModality.IMAGE
    return None


if __name__ == '__main__':
    global_args = setup_arguments()
    global_args.batch_size = 1

    print(global_args)

    set_random_seed(global_args.random_seed)

    device = torch.device(get_free_cuda_device_name(global_args) if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    full_dataset = datasets.load_data(name=global_args.dataset, num_partitions=1, split='noniid', seed=global_args.random_seed, global_args=global_args)
    train_dataloader = datasets.DataLoader(full_dataset.load_partition(partition_id=0), batch_size=global_args.batch_size, shuffle=True, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn())

    if should_override_text_tokenization(global_args.dataset):
        # Any model implementation that uses the text modality should exclude the tokenization of text embeddings in the model itself because fvcore seems to have issues with this in particular.
        # Providing the model with text embeddings in forms of tensors instead, works.
        model, second_modality = build_model_wrapper(global_args)
        model = model.switch_to_device(device)

        is_image_text_retrieval_task = global_args.dataset in ['flickr30k', 'coco-retrieval']

        if is_image_text_retrieval_task:
            batch = next(iter(train_dataloader))

            raw_input_other_modality, raw_text_input = batch
        else:
            X, y = next(iter(train_dataloader))
            raw_input_other_modality, raw_text_input = X[second_modality], X[InputModality.TEXT]

        raw_text_input = embed_text_for_image_text_retrieval(model, raw_text_input) if is_image_text_retrieval_task else embed_text(model, raw_text_input)
        X = (raw_input_other_modality, raw_text_input)
    else:
        X, y = next(iter(train_dataloader))

        model, _ = get_centralized_model_and_trainer(global_args, device)
        model = model.switch_to_device(device)

    flops_analysis = FlopCountAnalysis(model, X)
    as_table = flop_count_table(flops_analysis, max_depth=4)
    as_str = flop_count_str(flops_analysis)

    print(as_table)
    print(as_str)

    print('============ PRINTED FLOPS ARE ACTUALLY MACs. To obtain FLOPS, divide by 2 ============')
    print('============ flops scale with the batch_sizes so if you want the flops for e.g. a batch_size of 500, just multiply by 500 ============')

    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} | Number of total parameters: {sum(p.numel() for p in model.parameters())}')
