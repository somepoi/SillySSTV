"""End-to-end tests: bytes and pictures through audio and back, intact and damaged."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sillysstv import analog, damage, digital, modem, payloads
from sillysstv.profiles import PROFILES, get_profile

from .conftest import FAST_TEST_PROFILE, SAMPLE_RATE, add_noise


def _encode_bytes(data: bytes, ecc: int = 16, profile=FAST_TEST_PROFILE):
    payload = payloads.encode_raw(data)
    return digital.encode(payload, profile, SAMPLE_RATE, ecc=ecc, name="test.bin")


# --------------------------------------------------------------------------- #
# Modem
# --------------------------------------------------------------------------- #


def test_modem_round_trip_is_exact(rng: np.random.Generator) -> None:
    profile = FAST_TEST_PROFILE
    codeword = rng.integers(0, 256, 255, dtype=np.uint8).tobytes()
    signal = modem.modulate_codewords([codeword] * 3, profile, SAMPLE_RATE)

    demod = modem.Demodulator(signal, profile, SAMPLE_RATE)
    hits = demod.frames()
    assert len(hits) == 3
    for hit in hits:
        assert hit.codeword == codeword
        assert hit.score > 0.9


def test_frames_land_on_an_exact_grid() -> None:
    """The fade must never eat into the first preamble (it desyncs frame 0)."""
    profile = FAST_TEST_PROFILE
    signal = modem.modulate_codewords([bytes(255)] * 4, profile, SAMPLE_RATE)
    starts = modem.Demodulator(signal, profile, SAMPLE_RATE).find_frames()
    assert len(starts) == 4
    assert np.all(np.diff(starts) == modem.frame_sample_count(profile))


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_profile_round_trips(name: str, rng: np.random.Generator) -> None:
    profile = get_profile(name)
    codeword = rng.integers(0, 256, 255, dtype=np.uint8).tobytes()
    signal = modem.modulate_codewords([codeword], profile, SAMPLE_RATE)
    hits = modem.Demodulator(signal, profile, SAMPLE_RATE).frames()
    assert len(hits) == 1
    assert hits[0].codeword == codeword


# --------------------------------------------------------------------------- #
# Digital pipeline
# --------------------------------------------------------------------------- #


def test_digital_round_trip_is_bit_exact(rng: np.random.Generator) -> None:
    data = rng.integers(0, 256, 1500, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data)

    result = digital.decode(
        encoded.signal, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=16
    )
    assert result.stream == data
    assert result.valid.all()
    assert result.stats.sha_ok is True
    assert result.manifest is not None
    assert result.manifest.name == "test.bin"


def test_digital_auto_detects_profile_and_ecc(rng: np.random.Generator) -> None:
    data = rng.integers(0, 256, 600, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data, ecc=32)

    result = digital.decode(
        encoded.signal,
        SAMPLE_RATE,
        profile=None,
        profile_candidates=list(PROFILES.values()),
        ecc=None,
    )
    assert result.stats.profile == FAST_TEST_PROFILE.name
    assert result.stats.ecc == 32
    assert result.stream == data


def test_digital_survives_noise(rng: np.random.Generator) -> None:
    data = rng.integers(0, 256, 1500, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data, ecc=32)
    noisy = add_noise(encoded.signal, snr_db=6.0, seed=5)

    result = digital.decode(noisy, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=32)
    assert result.stream == data
    assert result.stats.sha_ok is True


def test_digital_dropouts_cost_only_the_blocks_they_hit(
    rng: np.random.Generator,
) -> None:
    data = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data, ecc=32)
    broken = damage.apply(
        encoded.signal,
        SAMPLE_RATE,
        damage.DamageSpec(dropouts=3, dropout_ms=300.0, seed=11),
    )

    result = digital.decode(broken, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=32)
    assert 0 < result.stats.bytes_missing < len(data)

    recovered = np.frombuffer(result.stream, np.uint8)
    original = np.frombuffer(data, np.uint8)
    # Whatever survived must be exactly right: a CRC per block is what stops a
    # mis-corrected codeword from silently poisoning the output.
    assert np.array_equal(recovered[result.valid], original[result.valid])


def test_digital_survives_cuts_that_shift_every_later_frame(
    rng: np.random.Generator,
) -> None:
    """Self-locating frames mean a splice costs only the frames it destroys."""
    data = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data, ecc=32)
    broken = damage.apply(
        encoded.signal, SAMPLE_RATE, damage.DamageSpec(cuts=2, cut_ms=250.0, seed=2)
    )
    assert len(broken) < len(encoded.signal)

    result = digital.decode(broken, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=32)
    assert result.stats.blocks_recovered >= result.stats.blocks_expected - 4

    recovered = np.frombuffer(result.stream, np.uint8)
    original = np.frombuffer(data, np.uint8)
    assert np.array_equal(recovered[result.valid], original[result.valid])


def test_digital_survives_losing_the_leading_manifest_copies(
    rng: np.random.Generator,
) -> None:
    """The manifest is re-sent periodically, so a chopped start is recoverable."""
    data = rng.integers(0, 256, 8000, dtype=np.uint8).tobytes()
    encoded = _encode_bytes(data, ecc=32)

    frame = modem.frame_sample_count(FAST_TEST_PROFILE)
    truncated = encoded.signal[3 * frame :]  # drop all three leading manifests

    result = digital.decode(truncated, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=32)
    assert result.manifest is not None
    assert result.manifest.total_len == len(data)
    assert result.manifest.sha256 == hashlib.sha256(data).digest()


def test_digital_reports_missing_signal() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with pytest.raises(digital.DecodeError):
        digital.decode(
            silence,
            SAMPLE_RATE,
            profile=None,
            profile_candidates=list(PROFILES.values()),
        )


def test_image_through_the_whole_digital_pipeline(picture) -> None:
    payload = payloads.encode_image_strips(picture, strip_height=16, quality=90)
    encoded = digital.encode(
        payload, FAST_TEST_PROFILE, SAMPLE_RATE, ecc=16, name="pic.png"
    )
    result = digital.decode(
        encoded.signal, SAMPLE_RATE, profile=FAST_TEST_PROFILE, ecc=16
    )
    decoded = payloads.decode_payload(
        result.stream,
        result.valid,
        result.manifest.payload_kind,
        result.manifest.extra,
    )
    assert decoded.image is not None
    assert decoded.image.size == picture.size
    error = np.abs(
        np.asarray(decoded.image, np.int16) - np.asarray(picture, np.int16)
    ).mean()
    assert error < 8.0


# --------------------------------------------------------------------------- #
# Analog pipeline
# --------------------------------------------------------------------------- #


def _analog_error(decoded, original) -> float:
    return float(
        np.abs(
            np.asarray(decoded, np.int16) - np.asarray(original, np.int16)
        ).mean()
    )


def test_analog_header_round_trip() -> None:
    fmt = analog.AnalogFormat(analog.MODE_RGB, 320, 256, 146.432)
    assert analog.AnalogFormat.unpack(fmt.pack()) == fmt


def test_analog_format_snaps_scan_time_to_the_transmittable_grid() -> None:
    """Scan time travels as a uint16 in 0.1 ms units, so the type snaps to it.

    Without this the encoder would generate timing the header cannot describe.
    """
    assert analog.AnalogFormat(analog.MODE_RGB, 320, 256, 146.432).scan_ms == 146.4


def test_analog_header_rejects_a_flipped_bit() -> None:
    raw = bytearray(analog.AnalogFormat(analog.MODE_RGB, 320, 256, 100.0).pack())
    raw[3] ^= 0x01
    assert analog.AnalogFormat.unpack(bytes(raw)) is None


def test_analog_round_trip_rgb(picture) -> None:
    signal, fmt = analog.encode(picture, "rgb", scan_ms=24.0, sample_rate=SAMPLE_RATE)
    decoded, stats = analog.decode(signal, SAMPLE_RATE)

    assert (stats.format.width, stats.format.height) == picture.size
    assert stats.lines_synced == picture.height
    assert _analog_error(decoded, picture) < 12.0


def test_analog_round_trip_mono(picture) -> None:
    grey = picture.convert("L")
    signal, _ = analog.encode(grey, "mono", scan_ms=24.0, sample_rate=SAMPLE_RATE)
    decoded, stats = analog.decode(signal, SAMPLE_RATE)

    assert stats.format.mode == analog.MODE_MONO
    assert decoded.size == grey.size
    assert _analog_error(decoded, grey) < 12.0


def test_analog_degrades_instead_of_failing(picture) -> None:
    """Noise should show up as picture noise, never as a decode failure."""
    signal, _ = analog.encode(picture, "rgb", scan_ms=24.0, sample_rate=SAMPLE_RATE)
    noisy = add_noise(signal, snr_db=12.0, seed=4)

    decoded, stats = analog.decode(noisy, SAMPLE_RATE)
    assert decoded.size == picture.size  # whole image, always
    assert stats.lines_synced > picture.height * 0.8
    assert _analog_error(decoded, picture) < 30.0


def test_analog_tolerates_leading_silence(picture) -> None:
    signal, _ = analog.encode(picture, "rgb", scan_ms=24.0, sample_rate=SAMPLE_RATE)
    padded = np.concatenate([np.zeros(SAMPLE_RATE // 2, np.float32), signal])

    decoded, _ = analog.decode(padded, SAMPLE_RATE)
    assert _analog_error(decoded, picture) < 12.0


def test_analog_rejects_a_non_sstv_signal(rng: np.random.Generator) -> None:
    noise = rng.normal(0, 0.2, SAMPLE_RATE * 2).astype(np.float32)
    with pytest.raises(analog.DecodeError):
        analog.decode(noise, SAMPLE_RATE, search_s=2.0)
