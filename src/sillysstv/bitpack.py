"""Conversion between byte strings and M-ary symbol streams.

Bits are laid out MSB-first, which keeps the mapping trivially reversible and
makes the symbol -> byte confidence mapping a plain reshape.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bytes_to_symbols",
    "symbols_to_bytes",
    "symbol_confidence_to_bytes",
    "symbols_per_bytes",
]


def bytes_to_symbols(data: bytes | bytearray, bits: int) -> np.ndarray:
    """Pack ``data`` into symbols of ``bits`` bits each (zero-padded at the end)."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    bit_array = np.unpackbits(arr)
    pad = (-len(bit_array)) % bits
    if pad:
        bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])
    groups = bit_array.reshape(-1, bits).astype(np.uint32)
    weights = (1 << np.arange(bits - 1, -1, -1, dtype=np.uint32))
    return (groups * weights).sum(axis=1).astype(np.int64)


def symbols_to_bytes(symbols: np.ndarray, bits: int, n_bytes: int) -> bytes:
    """Inverse of :func:`bytes_to_symbols`, truncated to ``n_bytes``."""
    symbols = np.asarray(symbols, dtype=np.uint32)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_array = ((symbols[:, None] >> shifts[None, :]) & 1).astype(np.uint8).ravel()
    need = n_bytes * 8
    if len(bit_array) < need:
        bit_array = np.concatenate(
            [bit_array, np.zeros(need - len(bit_array), dtype=np.uint8)]
        )
    return np.packbits(bit_array[:need]).tobytes()


def symbol_confidence_to_bytes(
    confidence: np.ndarray, bits: int, n_bytes: int
) -> np.ndarray:
    """Fold per-symbol confidences into a per-byte confidence (worst bit wins)."""
    per_bit = np.repeat(np.asarray(confidence, dtype=np.float32), bits)
    need = n_bytes * 8
    if len(per_bit) < need:
        per_bit = np.concatenate(
            [per_bit, np.zeros(need - len(per_bit), dtype=np.float32)]
        )
    return per_bit[:need].reshape(n_bytes, 8).min(axis=1)


def symbols_per_bytes(n_bytes: int, bits: int) -> int:
    """How many symbols are needed to carry ``n_bytes`` bytes."""
    return -(-n_bytes * 8 // bits)
