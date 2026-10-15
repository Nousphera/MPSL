import os

import torch

supported_image_tokenizer_checkpoints = {
    'ViT-Ti/16': {
        'huggingface': 'WinKawaks/vit-tiny-patch16-224',
        'pretrained_model': 'WinKawaks/vit-tiny-patch16-224'
    },
    'ViT-S/16': {
        'huggingface': 'WinKawaks/vit-small-patch16-224',
        'pretrained_model': 'WinKawaks/vit-small-patch16-224'
    },
    'ViT-B/16': {
        'huggingface': 'laion/CLIP-MetaTransformer-B-16-laion2B',
        'pretrained_model': 'laion2b_s34b_b88k'
    },
    'ViT-L/14': {
        'huggingface': 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        'pretrained_model': 'laion2b_s32b_b82k'
    },
    'ViT-H/14': {
        'huggingface': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'pretrained_model': 'laion2b_s32b_b79k'
    }
}

supported_audio_tokenizer_checkpoints = {
    'audioset-0.4593': 'MIT/ast-finetuned-audioset-10-10-0.4593'
}


def get_image_tokenizer_weights(base_encoder_name):
    import open_clip
    from transformers import AutoModelForImageClassification
    checkpoint_data = supported_image_tokenizer_checkpoints[base_encoder_name]

    if base_encoder_name not in supported_image_tokenizer_checkpoints.keys():
        raise NotImplementedError(f"Pre-trained image tokenizer weights for {base_encoder_name} is not implemented.")

    if base_encoder_name == 'ViT-Ti/16' or base_encoder_name == 'ViT-S/16':
        from mpsl.models.meta_transformer.base.meta_transformer import vit_base_encoder_options
        _loaded_dict = AutoModelForImageClassification.from_pretrained(supported_image_tokenizer_checkpoints[base_encoder_name]['pretrained_model'], cache_dir=os.path.join(os.environ['MODEL_WEIGHTS_DIR'], base_encoder_name)).state_dict()
        embed_dim = vit_base_encoder_options[base_encoder_name]['embed_dim']

        tokenizer_weights = {
            'class_embedding': _loaded_dict['vit.embeddings.cls_token'].reshape((embed_dim, )),
            'positional_embedding': _loaded_dict['vit.embeddings.position_embeddings'].reshape((197, embed_dim)),
            'conv1.weight': _loaded_dict['vit.embeddings.patch_embeddings.projection.weight'],
            'ln_post.weight': _loaded_dict['vit.layernorm.weight'],
            'ln_post.bias': _loaded_dict['vit.layernorm.bias']
        }

        return tokenizer_weights

    return open_clip.create_model_and_transforms(
        base_encoder_name,
        pretrained=checkpoint_data['pretrained_model'],
        cache_dir=os.path.join(os.environ['TOKENIZER_WEIGHTS_CACHE_DIR'], base_encoder_name)
    )[0].visual.state_dict()


def get_audio_tokenizer_weights(model_key, model_name):
    from transformers import AutoModelForAudioClassification
    if model_key not in supported_audio_tokenizer_checkpoints:
        raise NotImplementedError(f"Pre-trained audio tokenizer weights for {model_name} is not implemented.")

    return AutoModelForAudioClassification.from_pretrained(model_name).state_dict()


def get_encoder_weights(target_model_state_dict, use_large_encoder=False, include_classifier=True):
    import open_clip
    from transformers import AutoModelForImageClassification
    from mpsl.models.meta_transformer.base.meta_transformer import vit_base_encoder_options
    should_override_base_encoder_setting = 'VIT_BASE' in os.environ and os.environ['VIT_BASE'] != 'None'

    if should_override_base_encoder_setting:
        if os.environ['VIT_BASE'] == 'ViT-Ti/16' or os.environ['VIT_BASE'] == 'ViT-S/16':
            _loaded_dict = AutoModelForImageClassification.from_pretrained(supported_image_tokenizer_checkpoints[os.environ['VIT_BASE']]['pretrained_model'], cache_dir=os.path.join(
                                                                     os.environ['MODEL_WEIGHTS_DIR'],
                                                                     os.environ['VIT_BASE'])).vit.encoder.state_dict()

            # Merge all separate q, k & v layer weights into single qkv layer weights
            _loaded_dict = merge_qkv(_loaded_dict, q_name='query', k_name='key', v_name='value')
            encoder_state = {}

            for i in range(vit_base_encoder_options[os.environ['VIT_BASE']]['nr_of_blocks']):
                encoder_state[f'encoder.{i}.norm1.weight'] = _loaded_dict[f'layer.{i}.layernorm_before.weight']
                encoder_state[f'encoder.{i}.norm1.bias'] = _loaded_dict[f'layer.{i}.layernorm_before.bias']

                encoder_state[f'encoder.{i}.norm2.weight'] = _loaded_dict[f'layer.{i}.layernorm_after.weight']
                encoder_state[f'encoder.{i}.norm2.bias'] = _loaded_dict[f'layer.{i}.layernorm_after.bias']

                encoder_state[f'encoder.{i}.attn.qkv.weight'] = _loaded_dict[f'layer.{i}.attention.attention.qkv.weight']
                encoder_state[f'encoder.{i}.attn.qkv.bias'] = _loaded_dict[f'layer.{i}.attention.attention.qkv.bias']

                encoder_state[f'encoder.{i}.attn.proj.weight'] = _loaded_dict[f'layer.{i}.attention.output.dense.weight']
                encoder_state[f'encoder.{i}.attn.proj.bias'] = _loaded_dict[f'layer.{i}.attention.output.dense.bias']

                encoder_state[f'encoder.{i}.mlp.fc1.weight'] = _loaded_dict[f'layer.{i}.intermediate.dense.weight']
                encoder_state[f'encoder.{i}.mlp.fc1.bias'] = _loaded_dict[f'layer.{i}.intermediate.dense.bias']

                encoder_state[f'encoder.{i}.mlp.fc2.weight'] = _loaded_dict[f'layer.{i}.output.dense.weight']
                encoder_state[f'encoder.{i}.mlp.fc2.bias'] = _loaded_dict[f'layer.{i}.output.dense.bias']
        elif os.environ['VIT_BASE'] == 'ViT-H/14':
            _loaded_dict = open_clip.create_model_and_transforms('ViT-H-14', pretrained=supported_image_tokenizer_checkpoints['ViT-H/14']['pretrained_model'],
                                                                 cache_dir=os.path.join(
                                                                     os.environ['MODEL_WEIGHTS_DIR'],
                                                                     'ViT-H-14'))[
                0].visual.transformer.resblocks.state_dict()

            encoder_state = {}

            for i in range(32):
                encoder_state[f'encoder.{i}.norm1.weight'] = _loaded_dict[f'{i}.ln_1.weight']
                encoder_state[f'encoder.{i}.norm1.bias'] = _loaded_dict[f'{i}.ln_1.bias']

                encoder_state[f'encoder.{i}.norm2.weight'] = _loaded_dict[f'{i}.ln_2.weight']
                encoder_state[f'encoder.{i}.norm2.bias'] = _loaded_dict[f'{i}.ln_2.bias']

                encoder_state[f'encoder.{i}.attn.qkv.weight'] = _loaded_dict[f'{i}.attn.in_proj_weight']
                encoder_state[f'encoder.{i}.attn.qkv.bias'] = _loaded_dict[f'{i}.attn.in_proj_bias']

                encoder_state[f'encoder.{i}.attn.proj.weight'] = _loaded_dict[f'{i}.attn.out_proj.weight']
                encoder_state[f'encoder.{i}.attn.proj.bias'] = _loaded_dict[f'{i}.attn.out_proj.bias']

                encoder_state[f'encoder.{i}.mlp.fc1.weight'] = _loaded_dict[f'{i}.mlp.c_fc.weight']
                encoder_state[f'encoder.{i}.mlp.fc1.bias'] = _loaded_dict[f'{i}.mlp.c_fc.bias']

                encoder_state[f'encoder.{i}.mlp.fc2.weight'] = _loaded_dict[f'{i}.mlp.c_proj.weight']
                encoder_state[f'encoder.{i}.mlp.fc2.bias'] = _loaded_dict[f'{i}.mlp.c_proj.bias']
    else:
        encoder_weights_file = 'Meta-Transformer_large_patch14_encoder.pth' if use_large_encoder else 'Meta-Transformer_base_patch16_encoder.pth'
        encoder_state = add_prefix(torch.load(os.path.join(os.environ['MODEL_WEIGHTS_DIR'], encoder_weights_file)), prefix='encoder.')

    if include_classifier:
        encoder_state['classifier.weight'] = target_model_state_dict['classifier.weight']
        encoder_state['classifier.bias'] = target_model_state_dict['classifier.bias']

    return encoder_state


def add_prefix(state_dict, prefix='encoder.'):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[prefix + k] = v
    return new_state_dict


def merge_qkv(state_dict, k_name='k_proj', v_name='v_proj', q_name='q_proj'):
    """Can be used to merge weights of a model that uses separate query, key and value layers, into weights for a single combined qkv layer."""

    new_state_dict = {}
    for k, v in state_dict.items():
        if f'{k_name}.weight' in k:
            base_key = k.rsplit('.', 2)[0]
            k_weight = state_dict[base_key + f'.{k_name}.weight']
            v_weight = state_dict[base_key + f'.{v_name}.weight']
            q_weight = state_dict[base_key + f'.{q_name}.weight']
            new_state_dict[base_key + '.qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        elif f'{k_name}.bias' in k:
            base_key = k.rsplit('.', 2)[0]
            k_bias = state_dict[base_key + f'.{k_name}.bias']
            v_bias = state_dict[base_key + f'.{v_name}.bias']
            q_bias = state_dict[base_key + f'.{q_name}.bias']
            new_state_dict[base_key + '.qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        elif k_name in k or v_name in k or q_name in k:
            continue
        else:
            new_state_dict[k] = v
    return new_state_dict
