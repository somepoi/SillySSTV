"""Classic SSTV-style analog mode.

Pixels are sent as frequency, not as bits: 1500 Hz is black, 2300 Hz is white,
and every line opens with a 1200 Hz sync pulse.  There is no FEC and none is
wanted -- noise on the wire becomes noise in the picture, a dropout becomes a
smear, and the image always comes out whole.  That is the failure mode people
actually want from a photo.

The header is the one part that must survive verbatim, so it is sent as slow
FSK, protected by a CRC and repeated three times.
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as sp_ndimage

from . import dsp
from .errors import DecodeError

FREQ_SYNC = 1200.0
FREQ_BLACK = 1500.0
FREQ_WHITE = 2300.0
FREQ_LEADER = 1900.0
FREQ_BIT_ONE = 1100.0
FREQ_BIT_ZERO = 1300.0

SYNC_MS = 4.862
PORCH_MS = 0.572
LEADER_MS = 300.0
BREAK_MS = 10.0
HEADER_BIT_MS = 20.0
HEADER_REPEATS = 3

MODE_MONO = 0
MODE_RGB = 1
MODE_NAMES = {MODE_MONO: "mono", MODE_RGB: "rgb"}
NAME_MODES = {v: k for k, v in MODE_NAMES.items()}

HEADER_MAGIC = 0x5A
_HEADER = struct.Struct(">BBHHH")  # magic, version|mode, width, height, scan (0.1 ms)
HEADER_BYTES = _HEADER.size + 2  # + CRC-16
HEADER_BITS = HEADER_BYTES * 8
VERSION = 1

#: Default scan time per channel, in ms, for a 320 px wide line (Martin M1-ish).
DEFAULT_SCAN_MS = 146.432


#: The header carries scan time as a uint16 in 0.1 ms units.
SCAN_QUANTUM_MS = 0.1


@dataclass(frozen=True)
class AnalogFormat:
    mode: int
    width: int
    height: int
    scan_ms: float

    def __post_init__(self) -> None:
        # Snap to the grid the header can actually represent, so the encoder
        # generates exactly the timing the decoder will read back.
        snapped = round(self.scan_ms / SCAN_QUANTUM_MS) * SCAN_QUANTUM_MS
        object.__setattr__(self, "scan_ms", round(snapped, 1))

    @property
    def channels(self) -> int:
        return 3 if self.mode == MODE_RGB else 1

    def line_ms(self) -> float:
        return SYNC_MS + PORCH_MS + self.channels * (self.scan_ms + PORCH_MS)

    def duration_s(self) -> float:
        return (LEADER_MS * 2 + BREAK_MS) / 1000.0 + (
            HEADER_REPEATS * HEADER_BITS * HEADER_BIT_MS / 1000.0
        ) + self.height * self.line_ms() / 1000.0

    def pack(self) -> bytes:
        body = _HEADER.pack(
            HEADER_MAGIC,
            (VERSION << 4) | (self.mode & 0x0F),
            self.width,
            self.height,
            max(1, min(65535, round(self.scan_ms * 10))),
        )
        return body + struct.pack(">H", binascii.crc_hqx(body, 0xFFFF))

    @classmethod
    def unpack(cls, raw: bytes) -> "AnalogFormat | None":
        if len(raw) != HEADER_BYTES:
            return None
        body, crc = raw[: _HEADER.size], raw[_HEADER.size :]
        magic, version_mode, width, height, scan = _HEADER.unpack(body)
        if magic != HEADER_MAGIC:
            return None
        if struct.unpack(">H", crc)[0] != binascii.crc_hqx(body, 0xFFFF):
            return None
        if (version_mode >> 4) != VERSION:
            return None
        mode = version_mode & 0x0F
        if mode not in MODE_NAMES or not (0 < width <= 4096) or not (0 < height <= 4096):
            return None
        return cls(mode=mode, width=width, height=height, scan_ms=scan / 10.0)


def _ms(samples_per_ms: float, milliseconds: float) -> int:
    return max(1, int(round(samples_per_ms * milliseconds)))


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def encode(
    image: Image.Image,
    mode: str = "rgb",
    scan_ms: float | None = None,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, AnalogFormat]:
    mode_id = NAME_MODES[mode]
    image = image.convert("RGB" if mode_id == MODE_RGB else "L")
    width, height = image.size
    if scan_ms is None:
        scan_ms = DEFAULT_SCAN_MS * width / 320.0

    fmt = AnalogFormat(mode_id, width, height, scan_ms)
    scan_ms = fmt.scan_ms  # quantised to the header's 0.1 ms grid
    spms = sample_rate / 1000.0
    sync_n = _ms(spms, SYNC_MS)
    porch_n = _ms(spms, PORCH_MS)
    scan_n = _ms(spms, scan_ms)
    # Which source pixel each scan sample belongs to.
    pixel_index = np.minimum((np.arange(scan_n) * width) // scan_n, width - 1)

    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]

    segments: list[np.ndarray] = [
        np.full(_ms(spms, LEADER_MS), FREQ_LEADER, dtype=np.float32),
        np.full(_ms(spms, BREAK_MS), FREQ_SYNC, dtype=np.float32),
        np.full(_ms(spms, LEADER_MS), FREQ_LEADER, dtype=np.float32),
    ]

    bit_n = _ms(spms, HEADER_BIT_MS)
    header_bits = np.unpackbits(np.frombuffer(fmt.pack(), dtype=np.uint8))
    bit_freqs = np.where(header_bits == 1, FREQ_BIT_ONE, FREQ_BIT_ZERO).astype(np.float32)
    for _ in range(HEADER_REPEATS):
        segments.append(np.repeat(bit_freqs, bit_n))

    porch = np.full(porch_n, FREQ_BLACK, dtype=np.float32)
    sync = np.full(sync_n, FREQ_SYNC, dtype=np.float32)
    span = FREQ_WHITE - FREQ_BLACK
    for row in range(height):
        segments.append(sync)
        segments.append(porch)
        for channel in range(fmt.channels):
            values = pixels[row, :, channel] / 255.0
            segments.append((FREQ_BLACK + span * values[pixel_index]).astype(np.float32))
            segments.append(porch)

    freqs = np.concatenate(segments)
    signal = dsp.synth_from_frequencies(freqs, sample_rate)
    return dsp.apply_fade(signal, int(spms * 5)), fmt


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


@dataclass
class AnalogDecodeStats:
    format: AnalogFormat
    header_offset: int
    lines_synced: int
    lines_total: int

    def render(self) -> str:
        return "\n".join(
            [
                f"mode             : {MODE_NAMES[self.format.mode]}",
                f"size             : {self.format.width}x{self.format.height}",
                f"scan per channel : {self.format.scan_ms:.3f} ms",
                f"line syncs found : {self.lines_synced}/{self.lines_total}",
            ]
        )


def _find_header(
    freq: np.ndarray, sample_rate: int, search_s: float = 60.0
) -> tuple[AnalogFormat, int] | None:
    """Brute-force the header start offset, validated by its CRC.

    Scanning every 2 ms is cheap once the per-bit means are computed from a
    prefix sum, and it means we do not depend on detecting the leader tone.
    """
    spms = sample_rate / 1000.0
    bit_n = _ms(spms, HEADER_BIT_MS)
    guard = bit_n // 5  # ignore the transition edges of each bit slot
    total = HEADER_BITS * bit_n
    horizon = min(len(freq) - total, int(search_s * sample_rate))
    if horizon <= 0:
        return None

    cumsum = np.concatenate([[0.0], np.cumsum(freq, dtype=np.float64)])
    step = max(1, _ms(spms, 2.0))
    starts = np.arange(0, horizon, step, dtype=np.int64)
    slot = np.arange(HEADER_BITS, dtype=np.int64) * bit_n + guard
    means = dsp.windowed_means(
        cumsum, (starts[:, None] + slot[None, :]).ravel(), bit_n - 2 * guard
    ).reshape(len(starts), HEADER_BITS)

    bits = (means < (FREQ_BIT_ONE + FREQ_BIT_ZERO) / 2).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    for row, start in enumerate(starts):
        if packed[row, 0] != HEADER_MAGIC:
            continue
        fmt = AnalogFormat.unpack(packed[row].tobytes()[:HEADER_BYTES])
        if fmt is not None:
            return fmt, int(start)
    return None


def _line_starts(
    freq: np.ndarray, sample_rate: int, fmt: AnalogFormat, first_guess: int
) -> tuple[np.ndarray, int]:
    """Track the 1200 Hz sync pulse of every line, predicting through damage."""
    spms = sample_rate / 1000.0
    sync_n = _ms(spms, SYNC_MS)
    period = fmt.line_ms() * spms

    is_sync = (np.abs(freq - FREQ_SYNC) < 90.0).astype(np.float32)
    matched = sp_ndimage.uniform_filter1d(is_sync, size=sync_n, mode="constant")

    search = int(period * 0.25)
    window = matched[first_guess : first_guess + int(period * 2)]
    anchor = first_guess + (int(np.argmax(window)) if len(window) else 0)
    anchor -= sync_n // 2  # uniform_filter1d peaks at the pulse centre

    starts = np.empty(fmt.height, dtype=np.int64)
    synced = 0
    for row in range(fmt.height):
        predicted = int(round(anchor + row * period))
        lo = max(0, min(predicted, len(matched) - 1) - search)
        hi = min(len(matched), predicted + search + sync_n)
        if hi - lo > sync_n:
            local = matched[lo:hi]
            best = int(np.argmax(local))
            if local[best] > 0.55:
                position = lo + best - sync_n // 2
                synced += 1
            else:
                position = predicted
        else:
            position = predicted
        starts[row] = position
    return starts, synced


def decode(
    signal: np.ndarray, sample_rate: int, search_s: float = 60.0
) -> tuple[Image.Image, AnalogDecodeStats]:
    freq = dsp.instantaneous_frequency(
        signal,
        sample_rate,
        band=(FREQ_BIT_ONE - 300.0, FREQ_WHITE + 400.0),
        smooth=max(1, sample_rate // 8000),
    )

    found = _find_header(freq, sample_rate, search_s)
    if found is None:
        raise DecodeError(
            "no SillySSTV analog header found - wrong mode, or the header "
            "region of the recording is destroyed"
        )
    fmt, header_offset = found

    spms = sample_rate / 1000.0
    header_end = header_offset + HEADER_REPEATS * HEADER_BITS * _ms(spms, HEADER_BIT_MS)
    starts, synced = _line_starts(freq, sample_rate, fmt, header_end)

    sync_n = _ms(spms, SYNC_MS)
    porch_n = _ms(spms, PORCH_MS)
    scan_n = _ms(spms, fmt.scan_ms)
    pixel_n = max(1, scan_n // fmt.width)

    cumsum = np.concatenate([[0.0], np.cumsum(freq, dtype=np.float64)])
    # Sample each pixel from the middle of its dwell time.
    pixel_offsets = (np.arange(fmt.width) * scan_n) // fmt.width
    out = np.zeros((fmt.height, fmt.width, fmt.channels), dtype=np.uint8)

    for channel in range(fmt.channels):
        base = sync_n + porch_n + channel * (scan_n + porch_n)
        origins = starts[:, None] + base + pixel_offsets[None, :]
        means = dsp.windowed_means(cumsum, origins.ravel(), pixel_n)
        values = (means.reshape(fmt.height, fmt.width) - FREQ_BLACK) / (
            FREQ_WHITE - FREQ_BLACK
        )
        out[:, :, channel] = np.clip(values * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(out if fmt.channels == 3 else out[:, :, 0])
    return image, AnalogDecodeStats(fmt, header_offset, synced, fmt.height)
