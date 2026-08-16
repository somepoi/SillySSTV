"""The digital mode: arbitrary payloads -> FEC blocks -> M-FSK audio -> back."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import container, modem, payloads
from .container import (
    KIND_DATA,
    KIND_MANIFEST,
    MANIFEST_INTERVAL,
    MANIFEST_LEAD_COPIES,
    Manifest,
)
from .errors import DecodeError, EncodeError, ManifestError
from .fec import ECC_CANDIDATES, BlockCodec
from .modem import DEFAULT_SYNC_THRESHOLD, Demodulator
from .profiles import Profile

Progress = Callable[[str, int, int], None]


def _noop(_stage: str, _done: int, _total: int) -> None:
    return None


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


@dataclass
class EncodeResult:
    signal: np.ndarray
    sample_rate: int
    manifest: Manifest
    n_frames: int

    @property
    def duration(self) -> float:
        return len(self.signal) / self.sample_rate


def encode(
    payload: payloads.EncodedPayload,
    profile: Profile,
    sample_rate: int,
    *,
    ecc: int = 64,
    name: str = "",
    progress: Progress = _noop,
) -> EncodeResult:
    """Turn an encoded payload into a waveform."""
    codec = BlockCodec(ecc)
    data_size = codec.k - container.BLOCK_HEADER
    if data_size <= 0:
        raise EncodeError(f"ecc={ecc} leaves no room for data in a block")

    stream = payload.stream
    n_blocks = max(1, -(-len(stream) // data_size))
    manifest = Manifest(
        payload_kind=payload.kind,
        total_len=len(stream),
        n_blocks=n_blocks,
        data_size=data_size,
        sha256=hashlib.sha256(stream).digest(),
        name=name,
        flags=payload.flags,
        extra=payload.extra,
    )

    manifest_raw = manifest.pack()
    if len(manifest_raw) > data_size:
        raise EncodeError(
            f"manifest needs {len(manifest_raw)} bytes but a block only holds "
            f"{data_size}; use a smaller --ecc or a shorter file name"
        )

    def manifest_block(copy: int) -> bytes:
        return container.pack_block(KIND_MANIFEST, copy, manifest_raw, codec.k)

    raw_blocks: list[bytes] = [manifest_block(i) for i in range(MANIFEST_LEAD_COPIES)]
    copy = MANIFEST_LEAD_COPIES
    for index in range(n_blocks):
        chunk = stream[index * data_size : (index + 1) * data_size]
        raw_blocks.append(container.pack_block(KIND_DATA, index, chunk, codec.k))
        if (index + 1) % MANIFEST_INTERVAL == 0 and index + 1 < n_blocks:
            raw_blocks.append(manifest_block(copy))
            copy += 1

    codewords: list[bytes] = []
    for i, block in enumerate(raw_blocks):
        codewords.append(codec.encode(block))
        progress("fec", i + 1, len(raw_blocks))

    progress("modulate", 0, 1)
    signal = modem.modulate_codewords(codewords, profile, sample_rate)
    progress("modulate", 1, 1)
    return EncodeResult(signal, sample_rate, manifest, len(codewords))


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


@dataclass
class DecodeStats:
    frames_found: int = 0
    frames_decoded: int = 0
    blocks_expected: int = 0
    blocks_recovered: int = 0
    bytes_missing: int = 0
    ecc: int = 0
    profile: str = ""
    sync_score: float = 0.0
    sha_ok: bool | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"profile          : {self.profile}",
            f"ecc symbols      : {self.ecc}",
            f"frames located   : {self.frames_found} (mean sync {self.sync_score:.2f})",
            f"frames decoded   : {self.frames_decoded}",
            f"blocks recovered : {self.blocks_recovered}/{self.blocks_expected}",
            f"bytes missing    : {self.bytes_missing}",
        ]
        if self.sha_ok is not None:
            lines.append(f"sha256           : {'OK' if self.sha_ok else 'MISMATCH'}")
        lines.extend(f"note             : {n}" for n in self.notes)
        return "\n".join(lines)


@dataclass
class DecodeResult:
    manifest: Manifest | None
    stream: bytes
    valid: np.ndarray
    stats: DecodeStats


def detect_profile(
    signal: np.ndarray,
    sample_rate: int,
    candidates: Iterable[Profile],
    threshold: float = DEFAULT_SYNC_THRESHOLD,
) -> tuple[Profile, float]:
    """Pick the profile whose preamble correlates best with the recording."""
    best: tuple[Profile, float] | None = None
    for profile in candidates:
        demod = Demodulator(signal, profile, sample_rate)
        scores = demod.sync_scores()
        if not len(scores):
            continue
        # Median of the top peaks: a single lucky spike should not win.
        top = np.sort(scores)[-min(16, len(scores)) :]
        value = float(np.median(top))
        if best is None or value > best[1]:
            best = (profile, value)
    if best is None or best[1] < threshold * 0.6:
        raise DecodeError(
            "no SillySSTV digital signal found "
            f"(best preamble correlation {0.0 if best is None else best[1]:.2f})"
        )
    return best


def _decode_blocks(
    hits: list[modem.FrameHit], codec: BlockCodec
) -> tuple[dict[int, bytes], list[bytes], int]:
    """RS-decode every located frame and sort the survivors by kind."""
    data_blocks: dict[int, bytes] = {}
    manifests: list[bytes] = []
    decoded = 0
    for hit in hits:
        block, _ = codec.decode(hit.codeword, hit.confidence)
        if block is None:
            continue
        parsed = container.unpack_block(block)
        if parsed is None:
            continue
        decoded += 1
        kind, index, data = parsed
        if kind == KIND_MANIFEST:
            manifests.append(data)
        elif kind == KIND_DATA:
            data_blocks.setdefault(index, data)
    return data_blocks, manifests, decoded


def _pick_ecc(hits: list[modem.FrameHit], candidates: Iterable[int]) -> int:
    """Probe candidate ECC sizes and keep the one that validates most blocks."""
    sample = hits[: min(12, len(hits))]
    best_ecc, best_hits = 0, -1
    for ecc in candidates:
        codec = BlockCodec(ecc)
        good = 0
        for hit in sample:
            block, _ = codec.decode(hit.codeword, hit.confidence)
            if block is not None and container.unpack_block(block) is not None:
                good += 1
        if good > best_hits:
            best_ecc, best_hits = ecc, good
    if best_hits <= 0:
        raise DecodeError(
            "frames were found but none could be error-corrected; "
            "the recording may be too damaged, or --ecc/--profile are wrong"
        )
    return best_ecc


def decode(
    signal: np.ndarray,
    sample_rate: int,
    *,
    profile: Profile | None = None,
    profile_candidates: Iterable[Profile] = (),
    ecc: int | None = None,
    threshold: float = DEFAULT_SYNC_THRESHOLD,
    progress: Progress = _noop,
) -> DecodeResult:
    """Recover as much of the payload as the recording still carries."""
    stats = DecodeStats()

    if profile is None:
        progress("detect", 0, 1)
        profile, score = detect_profile(
            signal, sample_rate, profile_candidates, threshold
        )
        stats.notes.append(f"auto-detected profile {profile.name}")
        progress("detect", 1, 1)
    stats.profile = profile.name

    demod = Demodulator(signal, profile, sample_rate)
    starts = demod.find_frames(threshold)
    stats.frames_found = len(starts)
    if not starts:
        raise DecodeError(
            f"no frames found with profile {profile.name!r} "
            f"(sync threshold {threshold:.2f})"
        )

    hits: list[modem.FrameHit] = []
    for i, start in enumerate(starts):
        hit = demod.demodulate_frame(start)
        if hit is not None:
            hits.append(hit)
        progress("demod", i + 1, len(starts))
    stats.sync_score = float(np.mean([h.score for h in hits])) if hits else 0.0

    if ecc is None:
        ecc = _pick_ecc(hits, ECC_CANDIDATES)
        stats.notes.append(f"auto-detected ecc {ecc}")
    stats.ecc = ecc

    codec = BlockCodec(ecc)
    progress("fec", 0, len(hits))
    data_blocks, manifest_raws, decoded = _decode_blocks(hits, codec)
    stats.frames_decoded = decoded
    progress("fec", len(hits), len(hits))

    manifest: Manifest | None = None
    for raw in manifest_raws:
        try:
            manifest = Manifest.unpack(raw)
            break
        except ValueError:
            continue

    data_size = codec.k - container.BLOCK_HEADER
    if manifest is None:
        if not data_blocks:
            raise ManifestError("no manifest and no data blocks survived")
        stats.notes.append("manifest lost - falling back to a raw byte dump")
        n_blocks = max(data_blocks) + 1
        total_len = n_blocks * data_size
        kind, extra, flags = payloads.KIND_RAW, {}, 0
    else:
        n_blocks = manifest.n_blocks
        total_len = manifest.total_len
        data_size = manifest.data_size
        kind, extra, flags = manifest.payload_kind, manifest.extra, manifest.flags

    stats.blocks_expected = n_blocks
    stats.blocks_recovered = sum(1 for i in data_blocks if i < n_blocks)

    stream = bytearray(n_blocks * data_size)
    valid = np.zeros(n_blocks * data_size, dtype=bool)
    for index, data in data_blocks.items():
        if index >= n_blocks:
            continue
        offset = index * data_size
        chunk = data[:data_size]
        stream[offset : offset + len(chunk)] = chunk
        valid[offset : offset + len(chunk)] = True

    stream = bytes(stream[:total_len])
    valid = valid[:total_len]

    if flags & payloads.FLAG_INTERLEAVED:
        stream, valid = payloads.deinterleave(stream, valid)

    stats.bytes_missing = int((~valid).sum())
    if manifest is not None:
        stats.sha_ok = hashlib.sha256(stream).digest() == manifest.sha256

    result = DecodeResult(manifest, stream, valid, stats)
    result.stats.notes.append(f"payload kind {payloads.KIND_NAMES.get(kind, kind)}")
    if manifest is None:
        result.manifest = Manifest(
            payload_kind=kind,
            total_len=total_len,
            n_blocks=n_blocks,
            data_size=data_size,
            sha256=b"\x00" * 32,
            extra=extra,
            flags=flags,
        )
    return result
