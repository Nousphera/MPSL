"""PyAV decoding independent of removed torchvision/torchaudio media APIs."""

from pathlib import Path

import av
import numpy as np
import torch


def _decode_audio(path):
    blocks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}")
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        if rate is None or rate <= 0:
            raise ValueError(f"Missing or invalid audio sample rate in {path}")
        resampler = av.AudioResampler(
            format="fltp", layout=stream.codec_context.layout, rate=rate
        )
        for frame in container.decode(stream):
            blocks.extend(out.to_ndarray() for out in resampler.resample(frame))
        blocks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not blocks:
        raise ValueError(f"Empty audio stream in {path}")
    return torch.from_numpy(np.concatenate(blocks, axis=1).copy()), rate


def read_audio(path):
    """Return float32 planar samples in [-1, 1] and the original sample rate."""
    return _decode_audio(Path(path))


def read_video_audio(path):
    """Return uint8 TCHW frames, planar audio and source stream rates."""
    path = Path(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or rate <= 0:
            raise ValueError(f"Missing or invalid video frame rate in {path}")
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
        ]
    if not frames:
        raise ValueError(f"Empty video stream in {path}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    audio, audio_rate = _decode_audio(path)
    return video, audio, {"video_fps": float(rate), "audio_fps": audio_rate}
