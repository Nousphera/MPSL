import argparse
import os
from pathlib import Path

def strtobool(value):
    value = str(value).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


from mpsl.available_datasets import dataloaders
from mpsl.models import SupportedModel
from mpsl.models.meta_transformer.base.meta_transformer import trainable_params_options


def validate_base_argument_constraints(argument_parser):
    if argument_parser.audio_time_dimension is None:
        argument_parser.audio_time_dimension = '3.88s' if argument_parser.dataset == 'meld' else '10.24s'
    if argument_parser.vit_base != 'None':
        if argument_parser.vit_base == 'ViT-H/14' and not (argument_parser.dataset == 'coco-qa' or argument_parser.dataset == 't4sa'):
            raise NotImplementedError('ViT-H/14 base is currently only supported for image, text pairs (coco-qa & t4sa datasets)')

        if argument_parser.vit_base == 'ViT-Ti/16' or argument_parser.vit_base == 'ViT-S/16':
            # When using encoders that are smaller than ViT-B/16, we cannot use the CLIP encoder to encode text embeddings prior, as their dimension has a minimum of 512.
            # Thus, we need to manually override the chosen setting by disabling this altogether.
            argument_parser.use_clip_encoder_for_text_embeddings = False


def set_env_variables(arguments):
    directories = {
        'TORCH_DATA_DIR': arguments.torch_data_dir,
        'PRE_PROCESSORS_CACHE_DIR': arguments.pre_processors_cache_dir,
        'TOKENIZER_WEIGHTS_CACHE_DIR': arguments.tokenizer_weights_cache_dir,
        'MODEL_WEIGHTS_DIR': arguments.model_weights_dir,
    }
    directories = {key: Path(value) for key, value in directories.items()}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(value) for key, value in directories.items()})
    os.environ['VIT_BASE'] = arguments.vit_base


def build_base_argument_parser():
    """
    Builds and returns an instance of ArgumentParser, with all shared parameters across the possible configurations (centralized, split-learning, and federated-learning with and without the adapter approach).
    """
    parser = argparse.ArgumentParser()

    # == Data directories ==
    parser.add_argument('--torch_data_dir', type=str, default='data', help='The directory for the data of all datasets (default: data).')
    parser.add_argument('--pre_processors_cache_dir', type=str, default='cache/preprocessors', help='The directory for all pre-processor weights (default: cache/preprocessors).')
    parser.add_argument('--tokenizer_weights_cache_dir', type=str, default='cache/tokenizers', help='The directory for all tokenizer weights (default: cache/tokenizers).')
    parser.add_argument('--model_weights_dir', type=str, default='checkpoints', help='The directory of all model weights, including the required Meta-Transformer encoder checkpoint (default: checkpoints).')

    # == Dataset and model ==
    parser.add_argument('--dataset', type=str, required=True, choices=dataloaders.keys(), help='The dataset that should be used.')

    parser.add_argument('--model', type=str, help='The model that should be used.', choices=[x.lower() for x in SupportedModel.__members__.keys()], default='meta_transformer')
    parser.add_argument('--fusion_type', type=str, default='default', help='The type of fusion that should be used, if applicable. The \'default\' type has been used for the reported experiments. For available options, please refer to the model implementation of the chosen task.')
    parser.add_argument('--trainable_params_key', type=str, default='default', help='The key for the dictionary of trainable parameters. See MetaTransformerBase for all possible options. If no choice is made, the default is used, which is different per model implementation.', choices=trainable_params_options.keys())
    parser.add_argument('--use_pre_layer_norm', dest='use_pre_layer_norm', type=lambda x: bool(strtobool(x)), default=True, help='Whether pre-layer normalization should be used.')
    parser.add_argument('--use_post_layer_norm', dest='use_post_layer_norm', type=lambda x: bool(strtobool(x)), default=False, help='Whether post-layer normalization should be used.')
    parser.add_argument('--use_large_encoder', dest='use_large_encoder', type=lambda x: bool(strtobool(x)), default=False, help='Whether the large encoder version of Meta-Transformer should be used.')

    parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=lambda x: bool(strtobool(x)), default=False, help='Whether all blocks of the encoder should be frozen.')
    parser.add_argument('--nr_of_last_encoder_blocks_to_finetune', type=int, default=-1, help='If not freezing the entire encoder with freeze_encoder, we can optionally train the last X encoder blocks.')

    parser.add_argument('--use_clip_encoder_for_text_embeddings', dest='use_clip_encoder_for_text_embeddings', type=lambda x: bool(strtobool(x)), default=True, help='Whether the clip encoder should be used for encoding text, after text tokenization.')
    parser.add_argument('--vit_base', type=str, default='None', help='For manually overriding the ViT base that is used. All default experiments (those not in the ablation studies), are executed without touching this setting.')

    # == Misc ==
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size on the server. This means that with a batch_size of 500 with 25 clients, each client would have a mini-batch size of 20.')
    parser.add_argument('--nr_of_epochs', type=int, default=15)
    parser.add_argument('--start_lr', type=float, default=1e-4)
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers provided to the Dataloader.')

    parser.add_argument('--save_model_after_each_epoch', dest='save_model_after_each_epoch', type=lambda x: bool(strtobool(x)), default=False, help='Whether model(s) weights should be saved after each training epoch.')
    parser.add_argument('--save_final_model', dest='save_final_model', type=lambda x: bool(strtobool(x)), default=True, help='Whether the final obtained model weights after training should be saved.')
    parser.add_argument('--save_file_name', type=str, required=True, help='Unique output name for model weights and recorded metrics.')
    parser.add_argument('--gpu_id', type=str, default=None, help='If desired, this parameter can be used to explicitly define which CUDA gpu to use.')

    # Specifically for datasets with the audio modality
    parser.add_argument('--audio_time_dimension', type=str, default=None, help='This parameter is explicitly used for datasets with the audio modality. It defines the duration/dimension of a single audio input sample. For available options, please refer to the model implementation of the chosen task.')

    return parser


def expand_argument_parser_with_adapter_approach_parameters(argument_parser):
    # == For enabling adapter approach as taken in FedCLIP ==
    argument_parser.add_argument('--use_adapter_approach', dest='use_adapter_approach', type=lambda x: bool(strtobool(x)), default=False, help='Whether the adapter approach -- as taken in FedCLIP -- should be used. If so, ensure at least one of include_image_adapter, include_text_adapter, include_audio_adapter or include_unified_adapter is set to True.')
    argument_parser.add_argument('--include_image_adapter', dest='include_image_adapter', type=lambda x: bool(strtobool(x)), default=False, help='Whether an adapter should be used for the image modality.')
    argument_parser.add_argument('--include_text_adapter', dest='include_text_adapter', type=lambda x: bool(strtobool(x)), default=False, help='Whether an adapter should be used for the text modality.')
    argument_parser.add_argument('--include_audio_adapter', dest='include_audio_adapter', type=lambda x: bool(strtobool(x)), default=False, help='Whether an adapter should be used for the audio modality.')
    argument_parser.add_argument('--include_unified_adapter', dest='include_unified_adapter', type=lambda x: bool(strtobool(x)), default=False, help='Whether a single unified adapter should be used across all modalities.')

    return argument_parser


def expand_argument_parser_with_distributed_learning_parameters(argument_parser):
    argument_parser.add_argument("--dirichlet_alpha", type=float, default=0.1, help="Class concentration for noniid partitions (default: 0.1).")
    argument_parser.add_argument('--nr_of_clients', type=int, required=True, help='The number of clients to use during training.')
    argument_parser.add_argument('--dataset_split_type', type=str, required=True, choices=['iid', 'noniid'], help='The type of data distribution that should be used for splitting the original dataset into all separate client-side datasets.')

    return argument_parser
