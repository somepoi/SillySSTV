"""Low-level DSP primitives shared by the digital and analog codecs.

Everything here works on 1-D ``float32``/``float64`` mono signals.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage as sp_ndimage
from scipy import signal as sp_signal

__all__ = [
    "tone_matrix",
    "window_magnitudes",
    "sliding_tone_magnitudes",
    "synth_from_frequencies",
    "apply_fade",
    "instantaneous_frequency",
    "windowed_means",
]


def tone_matrix(bins: np.ndarray, n: int) -> np.ndarray:
    """Return the ``(len(bins), n)`` DFT kernel for the given integer FFT bins.

    Because the tone frequencies are exact multiples of ``sample_rate / n`` the
    resulting projections are perfectly orthogonal over one symbol, which is
    what makes non-coherent M-FSK detection reliable without any equaliser.
    """
    k = np.asarray(bins, dtype=np.float64).reshape(-1, 1)
    t = np.arange(n, dtype=np.float64).reshape(1, -1)
    return np.exp(-2j * np.pi * k * t / n).astype(np.complex64)


def window_magnitudes(windows: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Project ``windows`` (``(k, n)``) onto ``kernel`` (``(m, n)``) -> ``(k, m)``."""
    return np.abs(np.asarray(windows, dtype=np.float32) @ kernel.T)


def sliding_tone_magnitudes(
    x: np.ndarray,
    n: int,
    hop: int,
    bins: np.ndarray,
    chunk_windows: int = 4096,
) -> np.ndarray:
    """Magnitude spectrogram restricted to ``bins``, computed with ``rfft``.

    Returns an array of shape ``(n_windows, len(bins))``; window ``w`` starts at
    sample ``w * hop``.  Only the requested bins are kept, so memory stays small
    even for long recordings.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) < n:
        return np.zeros((0, len(bins)), dtype=np.float32)

    n_windows = 1 + (len(x) - n) // hop
    out = np.empty((n_windows, len(bins)), dtype=np.float32)
    strided = np.lib.stride_tricks.sliding_window_view(x, n)
    bins = np.asarray(bins, dtype=np.intp)

    for start in range(0, n_windows, chunk_windows):
        stop = min(start + chunk_windows, n_windows)
        block = strided[start * hop : (stop - 1) * hop + 1 : hop]
        spectrum = sp_fft.rfft(block, axis=-1, workers=-1)
        out[start:stop] = np.abs(spectrum[:, bins])
    return out


def synth_from_frequencies(freqs: np.ndarray, sample_rate: int) -> np.ndarray:
    """Continuous-phase synthesis of a per-sample instantaneous frequency track."""
    phase = np.cumsum(np.asarray(freqs, dtype=np.float64)) * (2.0 * np.pi / sample_rate)
    return np.sin(phase, dtype=np.float64).astype(np.float32)


def apply_fade(x: np.ndarray, n: int) -> np.ndarray:
    """Raised-cosine fade in/out, in place, to avoid clicks at the file edges."""
    n = min(n, len(x) // 2)
    if n <= 0:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def instantaneous_frequency(
    x: np.ndarray,
    sample_rate: int,
    band: tuple[float, float] | None = None,
    smooth: int = 0,
) -> np.ndarray:
    """FM-demodulate ``x`` into a per-sample frequency estimate in Hz.

    ``band`` applies a Butterworth band-pass first, which is what keeps the
    analog decoder usable at low SNR: out-of-band noise otherwise dominates the
    phase derivative.
    """
    x = np.asarray(x, dtype=np.float64)
    if band is not None:
        low, high = band
        nyq = sample_rate / 2.0
        wn = [max(low / nyq, 1e-4), min(high / nyq, 0.999)]
        sos = sp_signal.butter(4, wn, btype="bandpass", output="sos")
        x = sp_signal.sosfiltfilt(sos, x)

    analytic = sp_signal.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    freq = np.empty_like(phase)
    freq[1:] = np.diff(phase) * (sample_rate / (2.0 * np.pi))
    freq[0] = freq[1] if len(freq) > 1 else 0.0

    if smooth > 1:
        freq = sp_ndimage.uniform_filter1d(freq, size=smooth, mode="nearest")
    return np.clip(freq, 0.0, sample_rate / 2.0)


def windowed_means(cumsum: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Mean of a signal over ``[start, start + length)`` for many starts at once.

    ``cumsum`` must be ``np.concatenate([[0], np.cumsum(signal)])``.
    """
    starts = np.clip(np.asarray(starts, dtype=np.int64), 0, len(cumsum) - 1)
    stops = np.clip(starts + length, 0, len(cumsum) - 1)
    span = np.maximum(stops - starts, 1)
    return (cumsum[stops] - cumsum[starts]) / span
