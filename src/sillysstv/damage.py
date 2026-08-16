"""Deliberate signal degradation, for testing how much abuse a file survives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DamageSpec:
    snr_db: float | None = None
    dropouts: int = 0
    dropout_ms: float = 200.0
    cuts: int = 0
    cut_ms: float = 200.0
    clip: float | None = None
    hum_hz: float | None = None
    hum_level: float = 0.2
    bit_depth: int | None = None
    seed: int = 0

    def describe(self) -> str:
        parts = []
        if self.snr_db is not None:
            parts.append(f"noise @ {self.snr_db:g} dB SNR")
        if self.dropouts:
            parts.append(f"{self.dropouts} dropouts of {self.dropout_ms:g} ms")
        if self.cuts:
            parts.append(f"{self.cuts} cuts of {self.cut_ms:g} ms")
        if self.clip is not None:
            parts.append(f"clipping at {self.clip:g}")
        if self.hum_hz is not None:
            parts.append(f"{self.hum_hz:g} Hz hum at {self.hum_level:g}")
        if self.bit_depth is not None:
            parts.append(f"{self.bit_depth}-bit crush")
        return ", ".join(parts) or "nothing"


def apply(signal: np.ndarray, sample_rate: int, spec: DamageSpec) -> np.ndarray:
    """Apply every requested degradation, in a fixed, reproducible order."""
    rng = np.random.default_rng(spec.seed)
    x = np.array(signal, dtype=np.float32, copy=True)

    if spec.dropouts > 0 and len(x):
        width = max(1, int(spec.dropout_ms * sample_rate / 1000.0))
        for start in rng.integers(0, max(1, len(x) - width), size=spec.dropouts):
            x[int(start) : int(start) + width] = 0.0

    if spec.cuts > 0 and len(x):
        width = max(1, int(spec.cut_ms * sample_rate / 1000.0))
        starts = sorted(
            int(s) for s in rng.integers(0, max(1, len(x) - width), size=spec.cuts)
        )
        keep = np.ones(len(x), dtype=bool)
        for start in starts:
            keep[start : start + width] = False
        x = x[keep]

    if spec.hum_hz is not None and len(x):
        t = np.arange(len(x), dtype=np.float32) / sample_rate
        x += spec.hum_level * np.sin(2 * np.pi * spec.hum_hz * t, dtype=np.float32)

    if spec.snr_db is not None and len(x):
        power = float(np.mean(x.astype(np.float64) ** 2)) or 1e-12
        noise_power = power / (10.0 ** (spec.snr_db / 10.0))
        x += rng.normal(0.0, np.sqrt(noise_power), size=len(x)).astype(np.float32)

    if spec.clip is not None:
        x = np.clip(x, -spec.clip, spec.clip)

    if spec.bit_depth is not None and len(x):
        levels = float(2 ** (spec.bit_depth - 1))
        peak = float(np.max(np.abs(x))) or 1.0
        x = np.round(x / peak * levels) / levels * peak

    return x.astype(np.float32)
