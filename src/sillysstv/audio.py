"""Audio file I/O helpers (thin wrapper over ``soundfile``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import DecodeError

DEFAULT_SAMPLE_RATE = 44100


def write_wav(
    path: str | Path,
    signal: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_16",
    peak: float = 0.89,
) -> None:
    """Write a mono signal, normalised to ``peak`` to leave clipping headroom."""
    signal = np.asarray(signal, dtype=np.float32)
    top = float(np.max(np.abs(signal))) if signal.size else 0.0
    if top > 0:
        signal = signal * (peak / top)
    sf.write(str(path), signal, sample_rate, subtype=subtype)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any libsndfile-supported file as mono ``float32``."""
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - surface libsndfile messages verbatim
        raise DecodeError(f"cannot read audio file {path!r}: {exc}") from exc
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def describe(path: str | Path) -> str:
    info = sf.info(str(path))
    return (
        f"{info.format}/{info.subtype} {info.samplerate} Hz "
        f"{info.channels}ch {info.frames / info.samplerate:.2f} s"
    )
