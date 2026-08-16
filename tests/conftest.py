"""Shared fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sillysstv.profiles import get_profile

#: The shortest symbol length keeps the test suite quick; correctness of the
#: framing does not depend on which profile is used.
FAST_TEST_PROFILE = get_profile("hyper")
SAMPLE_RATE = 44100


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def picture() -> Image.Image:
    """A small image with gradients and hard edges."""
    height, width = 48, 64
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.stack(
        [
            (xx / width * 255),
            (yy / height * 255),
            np.full_like(xx, 90),
        ],
        axis=-1,
    ).astype(np.uint8)
    pixels[10:20, 15:40] = (240, 30, 30)
    return Image.fromarray(pixels, "RGB")


def add_noise(signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Additive white Gaussian noise at a given SNR."""
    generator = np.random.default_rng(seed)
    power = float(np.mean(signal.astype(np.float64) ** 2))
    sigma = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return (signal + generator.normal(0.0, sigma, len(signal))).astype(np.float32)
