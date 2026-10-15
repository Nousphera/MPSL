"""Behavior checks for the optional maintained text and media environment."""

from fractions import Fraction
import importlib
import ssl

import numpy as np
import pytest
import torch

open_clip = pytest.importorskip("open_clip")
av = pytest.importorskip("av")


def test_openai_token_ids_and_truncation_without_loading_weights(monkeypatch):
    from mpsl.models.meta_transformer.base.tokenizers.text_tokenizer import (
        TextTokenizer,
    )

    def no_download(*args, **kwargs):
        raise AssertionError("Token-only input must not load a pretrained encoder")

    monkeypatch.setattr(open_clip, "create_model_and_transforms", no_download)
    tokenizer = TextTokenizer(
        96,
        device="cpu",
        use_clip_encoder_for_text_embeddings=False,
        use_large_encoder=False,
    )
    out = tokenizer(["a photo of a dog", " ".join(["sample"] * 100)])
    # Captured from the original OpenAI tokenizer, including truncate=True.
    expected = torch.zeros((2, 96))
    expected[0, :7] = torch.tensor([49406, 320, 1125, 539, 320, 1929, 49407])
    expected[1, :77] = torch.tensor([49406, *([9542] * 75), 49407])
    torch.testing.assert_close(out, expected)


def test_cifar_import_does_not_disable_https_certificate_checks():
    original = ssl._create_default_https_context
    module = importlib.import_module(
        "mpsl.available_datasets.unimodal.image_classification.cifar100"
    )
    importlib.reload(module)
    assert ssl._create_default_https_context is original


def media_fixture(path):
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=8)
        video.width = video.height = 16
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("pcm_s16le", rate=16000)
        audio.layout = "mono"
        for i in range(8):
            pixels = np.full((16, 16, 3), 20 + i * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 8)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        waveform = (
            np.sin(np.arange(16000) * (2 * np.pi * 440 / 16000)) * 16000
        ).astype(np.int16)[None]
        frame = av.AudioFrame.from_ndarray(waveform, format="s16", layout="mono")
        frame.sample_rate = 16000
        frame.pts = 0
        frame.time_base = Fraction(1, 16000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return waveform.astype(np.float32) / 32768


def test_actual_video_and_audio_decoding(tmp_path):
    from mpsl.available_datasets.utils.media import read_audio, read_video_audio

    path = tmp_path / "media.mkv"
    expected = media_fixture(path)
    frames, waveform, info = read_video_audio(path)
    assert frames.shape == (8, 3, 16, 16)
    assert frames.dtype == torch.uint8
    assert info == {"video_fps": 8.0, "audio_fps": 16000}
    torch.testing.assert_close(waveform, torch.from_numpy(expected))
    audio, rate = read_audio(path)
    assert rate == 16000
    torch.testing.assert_close(audio, waveform)
    assert frames[-1].float().mean() > frames[0].float().mean()


def test_kaldi_audio_frontend_still_executes():
    from mpsl.available_datasets.utils.dataset_utils import load_transform

    torch.manual_seed(1)
    result = load_transform("audio", sample_rate=16000, duration=16000)(
        torch.randn(1, 16000)
    )
    assert result.ndim == 2 and result.shape[-1] == 128
    assert torch.isfinite(result).all()


def test_frozen_openai_encoder_activation_padding_and_cache(monkeypatch):
    from mpsl.models.meta_transformer.base.tokenizers import text_tokenizer as module

    calls = []

    created = []
    def create(name, **kwargs):
        calls.append((name, kwargs))
        # Exercise the complete OpenAI CLIP text architecture without checkpoints.
        model = open_clip.create_model(name, pretrained=None, force_quick_gelu=True)
        model.visual = None  # Unused vision branch is not part of the text tokenizer.
        created.append(model)
        return model, None, None

    monkeypatch.setattr(module, "_ENCODERS", {})
    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)

    def tokenizer(width, device="cpu"):
        return module.TextTokenizer(
            width, device=device, use_clip_encoder_for_text_embeddings=True
        )

    small = tokenizer(768)
    first = small(["a dog"])
    with torch.no_grad():
        expected = created[0].encode_text(open_clip.tokenize(["a dog"]), normalize=False)
    torch.testing.assert_close(first[:, :512], expected)
    assert len(created[0].transformer.resblocks) == 12
    assert all(not p.requires_grad for p in created[0].parameters())
    assert not created[0].training
    assert first[:, 512:].count_nonzero() == 0
    assert first.requires_grad is False
    assert not small.state_dict()
    torch.testing.assert_close(tokenizer(768)(["a dog"]), first)
    assert len(calls) == 1
    assert tokenizer(1024)(["a cat"]).shape == (1, 1024)
    tokenizer(768, "cpu:0")(["a dog"])
    assert len(calls) == 3
    assert [name for name, _ in calls] == ["ViT-B-16", "ViT-L-14", "ViT-B-16"]
    for _, kwargs in calls:
        assert kwargs["pretrained"] == "openai"
        assert kwargs["force_quick_gelu"] is True
    assert calls[0][1]["device"] != calls[2][1]["device"]
