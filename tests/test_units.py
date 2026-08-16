"""Unit tests for the building blocks: bit packing, FEC, framing, payloads."""

from __future__ import annotations

import numpy as np
import pytest

from sillysstv import bitpack, container, damage, payloads
from sillysstv.fec import CODEWORD_SIZE, BlockCodec


# --------------------------------------------------------------------------- #
# bitpack
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6])
def test_symbol_packing_round_trip(bits: int, rng: np.random.Generator) -> None:
    data = rng.integers(0, 256, 137, dtype=np.uint8).tobytes()
    symbols = bitpack.bytes_to_symbols(data, bits)
    assert symbols.max() < (1 << bits)
    assert len(symbols) == bitpack.symbols_per_bytes(len(data), bits)
    assert bitpack.symbols_to_bytes(symbols, bits, len(data)) == data


def test_symbol_confidence_folds_to_worst_bit() -> None:
    # 4 bits/symbol: two symbols per byte, so byte 0 sees symbols 0 and 1.
    confidence = np.array([0.9, 0.1, 0.8, 0.7], dtype=np.float32)
    folded = bitpack.symbol_confidence_to_bytes(confidence, 4, 2)
    assert folded == pytest.approx([0.1, 0.7], abs=1e-6)


# --------------------------------------------------------------------------- #
# Reed-Solomon
# --------------------------------------------------------------------------- #


def test_rs_round_trip_clean(rng: np.random.Generator) -> None:
    codec = BlockCodec(32)
    message = rng.integers(0, 256, codec.k, dtype=np.uint8).tobytes()
    codeword = codec.encode(message)
    assert len(codeword) == CODEWORD_SIZE
    assert codec.decode(codeword)[0] == message


def test_rs_repairs_errors_up_to_half_nsym(rng: np.random.Generator) -> None:
    codec = BlockCodec(32)  # corrects 16 unknown errors
    message = rng.integers(0, 256, codec.k, dtype=np.uint8).tobytes()
    damaged = bytearray(codec.encode(message))
    for position in rng.choice(CODEWORD_SIZE, 16, replace=False):
        damaged[int(position)] ^= 0xFF
    assert codec.decode(bytes(damaged))[0] == message


def test_rs_erasure_hints_beat_blind_correction(rng: np.random.Generator) -> None:
    """Flagging which bytes are suspect roughly doubles the tolerable damage."""
    codec = BlockCodec(32)
    message = rng.integers(0, 256, codec.k, dtype=np.uint8).tobytes()
    damaged = bytearray(codec.encode(message))
    positions = sorted(int(p) for p in rng.choice(CODEWORD_SIZE, 28, replace=False))
    for position in positions:
        damaged[position] ^= 0xFF

    assert codec.decode(bytes(damaged))[0] != message  # too many for blind repair

    confidence = np.ones(CODEWORD_SIZE, dtype=np.float32)
    confidence[positions] = 0.0
    assert codec.decode(bytes(damaged), confidence)[0] == message


def test_rs_rejects_wrong_payload_size() -> None:
    codec = BlockCodec(32)
    with pytest.raises(ValueError):
        codec.encode(b"too short")


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #


def test_manifest_round_trip() -> None:
    original = container.Manifest(
        payload_kind=payloads.KIND_IMAGE_STRIPS,
        total_len=12345,
        n_blocks=67,
        data_size=183,
        sha256=bytes(range(32)),
        name="пример.jpg",
        flags=1,
        extra={"w": 640, "h": 480, "strip": 32},
    )
    restored = container.Manifest.unpack(original.pack())
    assert restored == original


def test_manifest_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        container.Manifest.unpack(b"\x00" * 64)


def test_block_round_trip_and_crc_detection() -> None:
    block = container.pack_block(container.KIND_DATA, 4242, b"payload", k=64)
    assert len(block) == 64

    kind, index, data = container.unpack_block(block)
    assert (kind, index) == (container.KIND_DATA, 4242)
    assert data.startswith(b"payload")

    corrupted = bytearray(block)
    corrupted[20] ^= 0xFF
    assert container.unpack_block(bytes(corrupted)) is None


# --------------------------------------------------------------------------- #
# payloads
# --------------------------------------------------------------------------- #


def test_interleaver_round_trip(rng: np.random.Generator) -> None:
    payload = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    stream = payloads.interleave(payload)
    assert stream != payload
    restored, valid = payloads.deinterleave(stream, np.ones(len(stream), dtype=bool))
    assert restored == payload
    assert valid.all()


def test_interleaver_scatters_a_contiguous_loss(rng: np.random.Generator) -> None:
    """A lost block must not become a lost region of the picture."""
    length = 5000
    payload = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
    stream = payloads.interleave(payload)

    valid = np.ones(length, dtype=bool)
    valid[1000:1200] = False  # one contiguous block lost on the wire
    _, spread = payloads.deinterleave(stream, valid)

    lost = np.nonzero(~spread)[0]
    assert len(lost) == 200
    # Losses should land far apart rather than in one run.
    assert int(np.diff(lost).min()) > 1


def test_image_strips_round_trip(picture) -> None:
    encoded = payloads.encode_image_strips(picture, strip_height=16, quality=90)
    assert encoded.kind == payloads.KIND_IMAGE_STRIPS
    assert encoded.extra["n"] == 3

    valid = np.ones(len(encoded.stream), dtype=bool)
    decoded = payloads.decode_payload(
        encoded.stream, valid, encoded.kind, encoded.extra
    )
    assert decoded.image is not None
    assert decoded.image.size == picture.size
    error = np.abs(
        np.asarray(decoded.image, np.int16) - np.asarray(picture, np.int16)
    ).mean()
    assert error < 8.0  # JPEG loss only


def test_image_strips_survive_a_destroyed_strip(picture) -> None:
    encoded = payloads.encode_image_strips(picture, strip_height=16, quality=90)
    stream = bytearray(encoded.stream)
    valid = np.ones(len(stream), dtype=bool)

    # Wipe out the middle of the stream, which kills a strip but not the rest.
    start, stop = len(stream) // 3, len(stream) // 3 + 400
    stream[start:stop] = b"\x00" * (stop - start)
    valid[start:stop] = False

    decoded = payloads.decode_payload(
        bytes(stream), valid, encoded.kind, encoded.extra
    )
    assert decoded.image is not None
    assert decoded.image.size == picture.size
    assert "strips recovered" in decoded.detail


def test_image_raw_inpaints_scattered_losses(picture) -> None:
    encoded = payloads.encode_image_raw(picture)
    assert encoded.flags & payloads.FLAG_INTERLEAVED

    valid = np.ones(len(encoded.stream), dtype=bool)
    valid[500:900] = False  # a lost block, on the wire
    stream, spread = payloads.deinterleave(encoded.stream, valid)

    decoded = payloads.decode_payload(
        stream, spread, encoded.kind, encoded.extra, inpaint=True
    )
    assert decoded.image is not None
    error = np.abs(
        np.asarray(decoded.image, np.int16) - np.asarray(picture, np.int16)
    ).mean()
    # Scattered single-pixel holes filled from neighbours are nearly free.
    assert error < 3.0


def test_guess_kind() -> None:
    assert payloads.guess_kind("holiday.JPG", "auto") == "image-strips"
    assert payloads.guess_kind("archive.zip", "auto") == "raw"
    assert payloads.guess_kind("holiday.jpg", "raw") == "raw"


# --------------------------------------------------------------------------- #
# damage
# --------------------------------------------------------------------------- #


def test_damage_cuts_shorten_the_signal() -> None:
    signal = np.ones(44100, dtype=np.float32)
    spec = damage.DamageSpec(cuts=2, cut_ms=100.0, seed=3)
    out = damage.apply(signal, 44100, spec)
    assert len(out) == 44100 - 2 * 4410


def test_damage_is_reproducible() -> None:
    signal = np.sin(np.linspace(0, 500, 20000)).astype(np.float32)
    spec = damage.DamageSpec(snr_db=10.0, dropouts=3, seed=7)
    first = damage.apply(signal, 44100, spec)
    second = damage.apply(signal, 44100, spec)
    assert np.array_equal(first, second)
