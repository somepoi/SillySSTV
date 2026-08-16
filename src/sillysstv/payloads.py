"""Payload codecs: how a file becomes the byte stream that goes on the air.

The choice of codec is what decides *how* the result degrades:

``raw``
    The file verbatim.  Lost blocks become holes at known offsets.
``image-strips``
    The picture is cut into horizontal strips, each compressed independently
    and framed by a self-delimiting record.  Losing blocks costs you the
    strips they touched, never the whole picture.
``image-raw``
    Uncompressed pixels, scattered across the stream by an interleaver, so a
    lost block turns into isolated missing pixels that inpaint away almost
    invisibly.  Big, but the most graceful of the three.
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile
from scipy import ndimage as sp_ndimage

from .errors import EncodeError

# Salvage as much of a truncated strip as the JPEG decoder can give us.
ImageFile.LOAD_TRUNCATED_IMAGES = True

KIND_RAW = 0
KIND_IMAGE_STRIPS = 1
KIND_IMAGE_RAW = 2

KIND_NAMES = {
    KIND_RAW: "raw",
    KIND_IMAGE_STRIPS: "image-strips",
    KIND_IMAGE_RAW: "image-raw",
}
NAME_KINDS = {v: k for k, v in KIND_NAMES.items()}

FLAG_INTERLEAVED = 1 << 0

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
}

STRIP_MAGIC = b"SSTP"
_STRIP_HEADER = struct.Struct(">4sHII")  # magic, index, length, crc32
STRIP_HEADER_SIZE = 16  # padded from 14 to a round number
STRIP_ALIGN = 64


@dataclass
class EncodedPayload:
    stream: bytes
    kind: int
    extra: dict
    flags: int = 0


# --------------------------------------------------------------------------- #
# Interleaver
# --------------------------------------------------------------------------- #


def _interleave_stride(length: int) -> int:
    """A stride coprime with ``length``, near the golden ratio for even spread."""
    if length < 3:
        return 1
    stride = max(1, int(length * 0.6180339887))
    for candidate in range(stride, stride + length):
        value = candidate % length
        if value > 1 and math.gcd(value, length) == 1:
            return value
    return 1


def interleave(payload: bytes) -> bytes:
    """Scatter payload bytes so that a lost contiguous block spreads out."""
    length = len(payload)
    if length < 3:
        return payload
    stride = _interleave_stride(length)
    order = (np.arange(length, dtype=np.int64) * stride) % length
    return np.frombuffer(payload, dtype=np.uint8)[order].tobytes()


def deinterleave(stream: bytes, valid: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Undo :func:`interleave`, carrying the validity mask along."""
    length = len(stream)
    if length < 3:
        return stream, valid
    stride = _interleave_stride(length)
    order = (np.arange(length, dtype=np.int64) * stride) % length
    out = np.empty(length, dtype=np.uint8)
    out[order] = np.frombuffer(stream, dtype=np.uint8)
    out_valid = np.empty(length, dtype=bool)
    out_valid[order] = valid
    return out.tobytes(), out_valid


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def guess_kind(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = path.lower()
    return "image-strips" if any(lowered.endswith(s) for s in IMAGE_SUFFIXES) else "raw"


def load_image(path: str, max_size: int | None, grayscale: bool) -> Image.Image:
    image = Image.open(path)
    image = image.convert("L" if grayscale else "RGB")
    if max_size and max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def encode_raw(data: bytes) -> EncodedPayload:
    return EncodedPayload(stream=data, kind=KIND_RAW, extra={})


def encode_image_strips(
    image: Image.Image,
    strip_height: int = 32,
    quality: int = 80,
    fmt: str = "jpeg",
) -> EncodedPayload:
    """Cut the image into independently decodable horizontal strips."""
    width, height = image.size
    n_strips = -(-height // strip_height)
    chunks: list[bytes] = []

    for index in range(n_strips):
        top = index * strip_height
        box = (0, top, width, min(top + strip_height, height))
        buffer = io.BytesIO()
        if fmt == "jpeg":
            image.crop(box).save(buffer, "JPEG", quality=quality, optimize=True)
        else:
            image.crop(box).save(buffer, "PNG", optimize=True)
        body = buffer.getvalue()
        header = _STRIP_HEADER.pack(
            STRIP_MAGIC, index, len(body), zlib.crc32(body) & 0xFFFFFFFF
        ).ljust(STRIP_HEADER_SIZE, b"\x00")
        record = header + body
        pad = (-len(record)) % STRIP_ALIGN
        chunks.append(record + b"\x00" * pad)

    return EncodedPayload(
        stream=b"".join(chunks),
        kind=KIND_IMAGE_STRIPS,
        extra={
            "w": width,
            "h": height,
            "strip": strip_height,
            "fmt": fmt,
            "mode": image.mode,
            "n": n_strips,
        },
    )


def encode_image_raw(image: Image.Image) -> EncodedPayload:
    pixels = np.asarray(image, dtype=np.uint8)
    return EncodedPayload(
        stream=interleave(pixels.tobytes()),
        kind=KIND_IMAGE_RAW,
        extra={"w": image.width, "h": image.height, "mode": image.mode},
        flags=FLAG_INTERLEAVED,
    )


def encode_file(
    path: str,
    kind: str = "auto",
    *,
    strip_height: int = 32,
    quality: int = 80,
    strip_format: str = "jpeg",
    max_size: int | None = None,
    grayscale: bool = False,
) -> EncodedPayload:
    kind = guess_kind(path, kind)
    if kind == "raw":
        with open(path, "rb") as handle:
            return encode_raw(handle.read())
    if kind == "image-strips":
        image = load_image(path, max_size, grayscale)
        return encode_image_strips(image, strip_height, quality, strip_format)
    if kind == "image-raw":
        image = load_image(path, max_size, grayscale)
        return encode_image_raw(image)
    raise EncodeError(f"unknown payload kind {kind!r}")


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


@dataclass
class DecodedPayload:
    data: bytes | None = None
    image: Image.Image | None = None
    detail: str = ""
    suggested_suffix: str = ""


def _fill_missing(pixels: np.ndarray, valid: np.ndarray, inpaint: bool) -> np.ndarray:
    """Replace invalid pixels with the nearest good one (or flat grey)."""
    if valid.all():
        return pixels
    if not inpaint:
        pixels = pixels.copy()
        pixels[~valid] = 128
        return pixels
    indices = sp_ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = pixels[tuple(indices)]
    if not valid.any():
        filled[...] = 128
    return filled


def decode_image_raw(
    stream: bytes, valid: np.ndarray, extra: dict, inpaint: bool = True
) -> DecodedPayload:
    width, height, mode = int(extra["w"]), int(extra["h"]), extra.get("mode", "RGB")
    channels = 3 if mode == "RGB" else 1
    need = width * height * channels

    raw = np.frombuffer(stream.ljust(need, b"\x00")[:need], dtype=np.uint8)
    mask = np.zeros(need, dtype=bool)
    mask[: min(need, len(valid))] = valid[:need]

    shape = (height, width, channels) if channels > 1 else (height, width)
    pixels = raw.reshape(shape)
    pixel_valid = mask.reshape(shape)
    if channels > 1:
        pixel_valid = pixel_valid.all(axis=2)

    filled = _fill_missing(pixels, pixel_valid, inpaint)
    lost = int((~pixel_valid).sum())
    image = Image.fromarray(filled, mode)
    return DecodedPayload(
        image=image,
        detail=f"{lost}/{width * height} pixels missing ({100 * lost / (width * height):.2f}%)",
        suggested_suffix=".png",
    )


def _iter_strip_records(stream: bytes, valid: np.ndarray):
    """Scan the stream for strip records on the alignment grid."""
    position = 0
    limit = len(stream)
    while position + STRIP_HEADER_SIZE <= limit:
        if stream[position : position + 4] != STRIP_MAGIC:
            position += STRIP_ALIGN
            continue
        _, index, length, crc = _STRIP_HEADER.unpack(
            stream[position : position + _STRIP_HEADER.size]
        )
        body_start = position + STRIP_HEADER_SIZE
        body_end = body_start + length
        if length == 0 or body_end > limit:
            position += STRIP_ALIGN
            continue
        body = stream[body_start:body_end]
        intact = bool(valid[body_start:body_end].all()) and (
            zlib.crc32(body) & 0xFFFFFFFF == crc
        )
        yield index, body, intact
        record = STRIP_HEADER_SIZE + length
        position += record + ((-record) % STRIP_ALIGN)


def decode_image_strips(stream: bytes, valid: np.ndarray, extra: dict) -> DecodedPayload:
    width, height = int(extra["w"]), int(extra["h"])
    strip_height = int(extra["strip"])
    mode = extra.get("mode", "RGB")
    expected = int(extra.get("n", -(-height // strip_height)))

    canvas = Image.new(mode, (width, height), color=128 if mode == "L" else (128, 128, 128))
    recovered = 0
    partial = 0

    for index, body, intact in _iter_strip_records(stream, valid):
        if index >= expected:
            continue
        try:
            strip = Image.open(io.BytesIO(body))
            strip.load()
            strip = strip.convert(mode)
        except Exception:  # noqa: BLE001 - a broken strip is expected, not fatal
            continue
        canvas.paste(strip, (0, index * strip_height))
        recovered += 1
        if not intact:
            partial += 1

    return DecodedPayload(
        image=canvas,
        detail=(
            f"{recovered}/{expected} strips recovered"
            + (f" ({partial} partially damaged)" if partial else "")
        ),
        suggested_suffix=".png",
    )


def decode_payload(
    stream: bytes,
    valid: np.ndarray,
    kind: int,
    extra: dict,
    *,
    inpaint: bool = True,
) -> DecodedPayload:
    if kind == KIND_IMAGE_RAW:
        return decode_image_raw(stream, valid, extra, inpaint)
    if kind == KIND_IMAGE_STRIPS:
        return decode_image_strips(stream, valid, extra)
    lost = int((~valid).sum())
    return DecodedPayload(
        data=stream,
        detail=f"{lost}/{len(stream)} bytes missing ({100 * lost / max(len(stream), 1):.2f}%)",
    )
