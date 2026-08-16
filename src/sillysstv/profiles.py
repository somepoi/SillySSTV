"""Modem profiles for the digital mode.

A profile pins down the M-FSK geometry.  Tone *k* sits exactly on FFT bin
``bin_base + k * bin_step`` of a ``symbol_samples``-point transform, so tones
are orthogonal by construction and the demodulator is a plain bin argmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    """M-FSK geometry, expressed in FFT bins so it scales with the sample rate."""

    name: str
    symbol_samples: int
    tones: int
    bin_base: int
    bin_step: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.tones not in (2, 4, 8, 16, 32, 64):
            raise ValueError("tones must be a power of two between 2 and 64")
        top = self.bin_base + (self.tones - 1) * self.bin_step
        if top >= self.symbol_samples // 2:
            raise ValueError(
                f"profile {self.name!r}: top tone (bin {top}) exceeds Nyquist "
                f"for a {self.symbol_samples}-sample symbol"
            )

    @property
    def bits_per_symbol(self) -> int:
        return int(math.log2(self.tones))

    @property
    def tone_bins(self) -> np.ndarray:
        return self.bin_base + self.bin_step * np.arange(self.tones, dtype=np.intp)

    def tone_freqs(self, sample_rate: int) -> np.ndarray:
        return self.tone_bins * (sample_rate / self.symbol_samples)

    def baud(self, sample_rate: int) -> float:
        return sample_rate / self.symbol_samples

    def bitrate(self, sample_rate: int) -> float:
        return self.baud(sample_rate) * self.bits_per_symbol

    def band(self, sample_rate: int) -> tuple[float, float]:
        freqs = self.tone_freqs(sample_rate)
        return float(freqs[0]), float(freqs[-1])

    def summary(self, sample_rate: int = 44100) -> str:
        low, high = self.band(sample_rate)
        return (
            f"{self.name:<8} {self.baud(sample_rate):7.1f} baud  "
            f"{self.tones:>2}-FSK  {low:6.0f}-{high:6.0f} Hz  "
            f"{self.bitrate(sample_rate) / 8:7.1f} B/s raw   {self.note}"
        )


#: Ordered slowest (most robust) to fastest.  Bin bases are chosen so the
#: lowest tone lands around 500-700 Hz regardless of symbol length.
PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("robust", 1024, 16, 12, 2, "narrow band, survives heavy noise"),
        Profile("normal", 512, 16, 6, 2, "fits a 3 kHz voice channel"),
        Profile("fast", 256, 16, 3, 2, "good default for files"),
        Profile("turbo", 128, 16, 2, 2, "needs a clean wideband path"),
        Profile("hyper", 64, 16, 1, 2, "up to ~21 kHz, lossy codecs will kill it"),
    )
}

DEFAULT_PROFILE = "fast"


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {name!r} (known: {known})") from None
