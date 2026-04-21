from __future__ import annotations

import io
import queue as _queue
import struct
import threading
from typing import Any

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = np.float32


def _encode_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = pcm.nbytes
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, CHANNELS, sample_rate,
                          sample_rate * CHANNELS * 2, CHANNELS * 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class AudioRecorder:
    def __init__(self, device: str | int = "auto") -> None:
        self._device: str | int | None = None if device == "auto" else device
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._stream: Any = None
        self._lock = threading.Lock()
        # Real-time RMS amplitude stream, drained by the pill's waveform animation.
        # Bounded so a slow UI can't cause memory growth.
        self.level_queue: _queue.Queue[float] = _queue.Queue(maxsize=150)

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if self._recording:
            chunk = indata[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            try:
                self.level_queue.put_nowait(rms)
            except _queue.Full:
                pass
            with self._lock:
                self._chunks.append(chunk)

    def _drain_levels(self) -> None:
        try:
            while True:
                self.level_queue.get_nowait()
        except _queue.Empty:
            pass

    def start(self) -> None:
        self._chunks = []
        self._drain_levels()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if not self._recording:
            return b""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            chunks = list(self._chunks)
            self._chunks = []
        if not chunks:
            return b""
        audio = np.concatenate(chunks)
        return _encode_wav(audio)

    def cancel(self) -> None:
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._chunks = []
