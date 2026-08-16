"""Non-coherent M-FSK modem with self-locating frames.

Each frame is ``[preamble][255-byte RS codeword]``.  Frames are found by
correlating a known preamble against a cheap tone-bin spectrogram, so the
decoder never relies on a global timing reference: a frame that survives is
decoded wherever it happens to sit in the file.  Because every codeword carries
its own block index, cuts, splices and reordering are all tolerated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from . import bitpack, dsp
from .fec import CODEWORD_SIZE
from .profiles import Profile

PREAMBLE_LEN = 24
_PREAMBLE_SEED = 0x5111_5577

#: Symbols of steady tone padded around the transmission.  The click-suppressing
#: fade lives entirely inside these, so it can never eat a preamble or a data
#: symbol -- a fade over the first symbol shifts frame sync by half a symbol and
#: destroys the whole first frame.
LEAD_SYMBOLS = 4

#: Correlation score below which a peak is not considered a frame at all.
DEFAULT_SYNC_THRESHOLD = 0.30


def preamble_symbols(tones: int, length: int = PREAMBLE_LEN) -> np.ndarray:
    """Deterministic pseudo-random preamble with no repeated adjacent tones."""
    rng = random.Random(_PREAMBLE_SEED ^ tones)
    seq: list[int] = []
    previous = -1
    for _ in range(length):
        choice = rng.randrange(tones)
        while choice == previous:
            choice = rng.randrange(tones)
        seq.append(choice)
        previous = choice
    return np.array(seq, dtype=np.int64)


def frame_symbol_count(profile: Profile) -> int:
    """Symbols per frame: preamble plus one full RS codeword."""
    return PREAMBLE_LEN + bitpack.symbols_per_bytes(
        CODEWORD_SIZE, profile.bits_per_symbol
    )


def frame_sample_count(profile: Profile) -> int:
    return frame_symbol_count(profile) * profile.symbol_samples


# --------------------------------------------------------------------------- #
# Modulation
# --------------------------------------------------------------------------- #


def modulate_symbols(
    symbols: np.ndarray, profile: Profile, sample_rate: int
) -> np.ndarray:
    """Continuous-phase M-FSK synthesis of a symbol stream."""
    freqs = profile.tone_freqs(sample_rate)[np.asarray(symbols, dtype=np.intp)]
    per_sample = np.repeat(freqs, profile.symbol_samples)
    return dsp.synth_from_frequencies(per_sample, sample_rate)


def modulate_codewords(
    codewords: list[bytes], profile: Profile, sample_rate: int
) -> np.ndarray:
    """Turn a list of 255-byte codewords into one continuous waveform."""
    if not codewords:
        return np.zeros(0, dtype=np.float32)

    bits = profile.bits_per_symbol
    preamble = preamble_symbols(profile.tones)
    lead = np.zeros(LEAD_SYMBOLS, dtype=np.int64)

    chunks: list[np.ndarray] = [lead]
    for codeword in codewords:
        chunks.append(preamble)
        chunks.append(bitpack.bytes_to_symbols(codeword, bits))
    chunks.append(lead)

    signal = modulate_symbols(np.concatenate(chunks), profile, sample_rate)
    return dsp.apply_fade(signal, LEAD_SYMBOLS * profile.symbol_samples)


# --------------------------------------------------------------------------- #
# Demodulation
# --------------------------------------------------------------------------- #


@dataclass
class FrameHit:
    """One frame located in the recording."""

    start: int
    score: float
    codeword: bytes
    confidence: np.ndarray


class Demodulator:
    """Locates and demodulates frames in a recording for one profile."""

    #: Spectrogram hop as a fraction of the symbol length. 1/4 keeps the search
    #: grid cheap; the residual timing error is removed by a fine search.
    HOP_DIVISOR = 4

    def __init__(self, signal: np.ndarray, profile: Profile, sample_rate: int) -> None:
        self.profile = profile
        self.sample_rate = sample_rate
        self.symbol_samples = profile.symbol_samples
        self.hop = max(1, profile.symbol_samples // self.HOP_DIVISOR)
        self.preamble = preamble_symbols(profile.tones)
        self.frame_symbols = frame_symbol_count(profile)
        self.frame_samples = frame_sample_count(profile)
        self._kernel = dsp.tone_matrix(profile.tone_bins, profile.symbol_samples)

        signal = np.asarray(signal, dtype=np.float32)
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if peak > 0:
            signal = signal / peak
        # Pad so the last frame can be read without bounds juggling.
        self.signal = np.concatenate(
            [signal, np.zeros(self.frame_samples + self.symbol_samples, np.float32)]
        )
        self.length = len(signal)

    # -- frame search ------------------------------------------------------- #

    def sync_scores(self) -> np.ndarray:
        """Normalised preamble correlation on the hop grid."""
        mags = dsp.sliding_tone_magnitudes(
            self.signal, self.symbol_samples, self.hop, self.profile.tone_bins
        )
        if not len(mags):
            return np.zeros(0, dtype=np.float32)

        norm = mags / (mags.sum(axis=1, keepdims=True) + 1e-9)
        step = self.symbol_samples // self.hop
        span = len(norm) - (len(self.preamble) - 1) * step
        if span <= 0:
            return np.zeros(0, dtype=np.float32)

        score = np.zeros(span, dtype=np.float32)
        for i, tone in enumerate(self.preamble):
            score += norm[i * step : i * step + span, tone]
        return score / len(self.preamble)

    def find_frames(self, threshold: float = DEFAULT_SYNC_THRESHOLD) -> list[int]:
        """Sample offsets of every plausible frame start, best score first."""
        score = self.sync_scores()
        if not len(score):
            return []
        min_distance = max(1, int(self.frame_samples * 0.5) // self.hop)
        peaks, _ = sp_signal.find_peaks(score, height=threshold, distance=min_distance)
        starts = [int(p) * self.hop for p in peaks]
        return [self._refine(s) for s in starts if s + self.frame_samples <= len(self.signal)]

    def _refine(self, start: int) -> int:
        """Sample-accurate re-alignment of a coarse frame start."""
        radius = self.hop
        step = max(1, self.symbol_samples // 16)
        candidates = np.arange(start - radius, start + radius + 1, step)
        candidates = candidates[(candidates >= 0)]
        if not len(candidates):
            return start

        best, best_score = start, -1.0
        for candidate in candidates:
            if candidate + len(self.preamble) * self.symbol_samples > len(self.signal):
                continue
            mags = self._symbol_magnitudes(candidate, len(self.preamble))
            norm = mags / (mags.sum(axis=1, keepdims=True) + 1e-9)
            value = float(norm[np.arange(len(self.preamble)), self.preamble].mean())
            if value > best_score:
                best, best_score = int(candidate), value
        return best

    # -- symbol recovery ---------------------------------------------------- #

    def _symbol_magnitudes(self, start: int, count: int) -> np.ndarray:
        offsets = start + np.arange(count, dtype=np.intp) * self.symbol_samples
        index = offsets[:, None] + np.arange(self.symbol_samples, dtype=np.intp)[None, :]
        return dsp.window_magnitudes(self.signal[index], self._kernel)

    def demodulate_frame(self, start: int) -> FrameHit | None:
        """Recover the codeword carried by the frame starting at ``start``."""
        if start < 0 or start + self.frame_samples > len(self.signal):
            return None

        mags = self._symbol_magnitudes(start, self.frame_symbols)
        ordered = np.sort(mags, axis=1)
        strongest = ordered[:, -1]
        runner_up = ordered[:, -2]
        symbols = mags.argmax(axis=1)
        # Confidence = how much the winning tone beats its closest rival.
        confidence = 1.0 - (runner_up / (strongest + 1e-9))

        norm = mags / (mags.sum(axis=1, keepdims=True) + 1e-9)
        score = float(norm[np.arange(len(self.preamble)), self.preamble].mean())

        payload_symbols = symbols[len(self.preamble) :]
        payload_conf = confidence[len(self.preamble) :]
        bits = self.profile.bits_per_symbol
        codeword = bitpack.symbols_to_bytes(payload_symbols, bits, CODEWORD_SIZE)
        byte_conf = bitpack.symbol_confidence_to_bytes(
            payload_conf, bits, CODEWORD_SIZE
        )
        return FrameHit(start=start, score=score, codeword=codeword, confidence=byte_conf)

    def frames(self, threshold: float = DEFAULT_SYNC_THRESHOLD) -> list[FrameHit]:
        hits = []
        for start in self.find_frames(threshold):
            hit = self.demodulate_frame(start)
            if hit is not None:
                hits.append(hit)
        return hits
