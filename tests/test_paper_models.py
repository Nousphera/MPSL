"""Offline validation must retain the published Meta-Transformer architecture."""

import torch
from mpsl import smoke


def test_smoke_builds_full_paper_encoder_and_modality_tokenizers():
    torch.set_num_threads(1)
    assert hasattr(smoke, "create_system"), (
        "Smoke must construct the paper model explicitly"
    )
    clients, server = smoke.create_system("early")
    assert len(server.encoder) == 12
    assert server.embed_dim == 768
    assert server.encoder[0].attn.num_heads == 12
    assert server.encoder[0].mlp.fc1.out_features == 3072
    client = clients[0]
    assert client.image_tokenizer.tokenizer.patch_embedding.proj.weight.shape == (
        768,
        3,
        16,
        16,
    )
    assert client.audio_tokenizer.tokenizer.patch_embedding.proj.weight.shape == (
        768,
        1,
        16,
        16,
    )
    with torch.no_grad():
        features = client(smoke.sample_inputs(1))
        output = server(features)
    assert features.shape == (1, 306, 768)
    assert output.shape == (1, 51)
