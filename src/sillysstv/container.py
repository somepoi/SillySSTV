"""Block and manifest layout for the digital mode.

Block (before Reed-Solomon, ``k = 255 - nsym`` bytes)::

    0        kind      u8   0 = manifest, 1 = data
    1..4     index     u24  block index (or manifest copy number)
    4..8     crc32     u32  checksum of the data field
    8..k     data           payload slice, zero padded

The CRC matters because Reed-Solomon can *mis*-correct a block that is damaged
beyond its capacity; without it a silently wrong block would poison the output.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

BLOCK_HEADER = 8
KIND_MANIFEST = 0
KIND_DATA = 1

MANIFEST_MAGIC = b"SSTV"
MANIFEST_VERSION = 1
_MANIFEST_FIXED = struct.Struct(">4sBBBIIH32s")

#: A manifest copy is repeated this many times up front...
MANIFEST_LEAD_COPIES = 3
#: ...and re-sent every N data blocks, so a cut at the start is survivable.
MANIFEST_INTERVAL = 24


@dataclass
class Manifest:
    """Self-describing header for a transmission."""

    payload_kind: int
    total_len: int
    n_blocks: int
    data_size: int
    sha256: bytes
    name: str = ""
    flags: int = 0
    extra: dict = field(default_factory=dict)

    def pack(self) -> bytes:
        name = self.name.encode("utf-8")[:255]
        extra = json.dumps(self.extra, separators=(",", ":")).encode("utf-8")
        return (
            _MANIFEST_FIXED.pack(
                MANIFEST_MAGIC,
                MANIFEST_VERSION,
                self.payload_kind,
                self.flags,
                self.total_len,
                self.n_blocks,
                self.data_size,
                self.sha256,
            )
            + bytes([len(name)])
            + name
            + struct.pack(">H", len(extra))
            + extra
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "Manifest":
        size = _MANIFEST_FIXED.size
        if len(raw) < size + 3:
            raise ValueError("manifest too short")
        magic, version, kind, flags, total_len, n_blocks, data_size, digest = (
            _MANIFEST_FIXED.unpack(raw[:size])
        )
        if magic != MANIFEST_MAGIC:
            raise ValueError("bad manifest magic")
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version {version}")

        pos = size
        name_len = raw[pos]
        pos += 1
        name = raw[pos : pos + name_len].decode("utf-8", "replace")
        pos += name_len
        (extra_len,) = struct.unpack(">H", raw[pos : pos + 2])
        pos += 2
        extra_raw = raw[pos : pos + extra_len]
        try:
            extra = json.loads(extra_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            extra = {}
        return cls(
            payload_kind=kind,
            total_len=total_len,
            n_blocks=n_blocks,
            data_size=data_size,
            sha256=digest,
            name=name,
            flags=flags,
            extra=extra,
        )


def pack_block(kind: int, index: int, data: bytes, k: int) -> bytes:
    """Build one pre-FEC block of exactly ``k`` bytes."""
    capacity = k - BLOCK_HEADER
    if len(data) > capacity:
        raise ValueError(f"block data too large: {len(data)} > {capacity}")
    padded = data.ljust(capacity, b"\x00")
    header = bytes([kind]) + index.to_bytes(3, "big") + struct.pack(
        ">I", zlib.crc32(padded) & 0xFFFFFFFF
    )
    return header + padded


def unpack_block(block: bytes) -> tuple[int, int, bytes] | None:
    """Validate and split a block; ``None`` when the CRC does not match."""
    if len(block) <= BLOCK_HEADER:
        return None
    kind = block[0]
    index = int.from_bytes(block[1:4], "big")
    (crc,) = struct.unpack(">I", block[4:8])
    data = block[BLOCK_HEADER:]
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        return None
    return kind, index, data
