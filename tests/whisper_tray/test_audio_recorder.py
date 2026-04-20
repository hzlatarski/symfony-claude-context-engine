import io
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_tray.audio_recorder import AudioRecorder, _encode_wav


def test_encode_wav_produces_valid_wav_header():
    samples = np.zeros(1600, dtype=np.float32)
    wav_bytes = _encode_wav(samples, sample_rate=16000)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_encode_wav_correct_length():
    n = 1600
    samples = np.zeros(n, dtype=np.float32)
    wav_bytes = _encode_wav(samples, sample_rate=16000)
    # 44-byte header + 2 bytes per sample (int16)
    assert len(wav_bytes) == 44 + n * 2


def test_encode_wav_clips_float32_correctly():
    samples = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
    wav_bytes = _encode_wav(samples, sample_rate=16000)
    # Read int16 samples (skip 44-byte header)
    int16_samples = np.frombuffer(wav_bytes[44:], dtype=np.int16)
    assert int16_samples[0] == 32767   # clipped positive
    assert int16_samples[1] == -32767  # clipped negative (limit of int16 with 32767 scale)


@patch("whisper_tray.audio_recorder.sd")
def test_recorder_start_opens_stream(mock_sd):
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
    mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

    recorder = AudioRecorder(device="auto")
    recorder.start()
    assert mock_sd.InputStream.called


@patch("whisper_tray.audio_recorder.sd")
def test_recorder_stop_returns_bytes(mock_sd):
    recorder = AudioRecorder(device="auto")
    recorder._chunks = [np.zeros(160, dtype=np.float32)]
    recorder._recording = True
    result = recorder.stop()
    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"


@patch("whisper_tray.audio_recorder.sd")
def test_recorder_stop_without_start_returns_empty(mock_sd):
    recorder = AudioRecorder(device="auto")
    result = recorder.stop()
    assert result == b""
