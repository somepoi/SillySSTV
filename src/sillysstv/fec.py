"""Reed-Solomon block coding with soft-decision erasure hints.

Every block is an RS(255, 255-nsym) codeword over GF(256).  The codec can fix
``nsym // 2`` unknown byte errors, or up to ``nsym`` erasures when the modem
tells us *which* bytes it was unsure about -- that soft information roughly
doubles the tolerable damage.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional Cython accelerator
    from creedsolo import RSCodec, ReedSolomonError  # type: ignore
except ImportError:  # pragma: no cover
    from reedsolo import RSCodec, ReedSolomonError

CODEWORD_SIZE = 255
#: Candidate ECC sizes probed by ``--ecc auto`` on decode.
ECC_CANDIDATES = (16, 32, 48, 64, 96, 128)


class BlockCodec:
    """RS(255, 255-nsym) codec operating on fixed-size blocks."""

    def __init__(self, nsym: int = 64) -> None:
        if not 2 <= nsym <= 200 or nsym % 2:
            raise ValueError("nsym must be an even number in [2, 200]")
        self.nsym = int(nsym)
        self.k = CODEWORD_SIZE - self.nsym
        self._rs = RSCodec(self.nsym, nsize=CODEWORD_SIZE)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BlockCodec(nsym={self.nsym}, k={self.k})"

    def encode(self, payload: bytes | bytearray) -> bytes:
        """Encode exactly ``k`` bytes into a 255-byte codeword."""
        if len(payload) != self.k:
            raise ValueError(f"payload must be {self.k} bytes, got {len(payload)}")
        return bytes(self._rs.encode(bytearray(payload)))

    def decode(
        self,
        codeword: bytes | bytearray,
        confidence: np.ndarray | None = None,
    ) -> tuple[bytes | None, int]:
        """Try to repair a codeword.

        Returns ``(message, n_corrected)`` or ``(None, -1)`` if the block is
        beyond repair.  ``confidence`` is an optional per-byte reliability in
        ``[0, 1]``; the least reliable bytes are declared erasures.
        """
        if len(codeword) != CODEWORD_SIZE:
            return None, -1

        attempts: list[list[int] | None] = [None]
        if confidence is not None:
            weak = self._weak_positions(confidence)
            # Try progressively more aggressive erasure sets. Fewer erasures
            # leaves more room for the decoder to also fix unflagged errors.
            for count in (self.nsym // 2, self.nsym):
                subset = weak[:count]
                if len(subset):
                    attempts.append(sorted(int(i) for i in subset))

        for erase_pos in attempts:
            try:
                message, _, errata = self._rs.decode(
                    bytearray(codeword), erase_pos=erase_pos
                )
            except (ReedSolomonError, ZeroDivisionError, IndexError):
                continue
            return bytes(message), len(errata)
        return None, -1

    def _weak_positions(self, confidence: np.ndarray) -> np.ndarray:
        conf = np.asarray(confidence, dtype=np.float32)
        if len(conf) < CODEWORD_SIZE:
            conf = np.concatenate(
                [conf, np.zeros(CODEWORD_SIZE - len(conf), dtype=np.float32)]
            )
        order = np.argsort(conf[:CODEWORD_SIZE], kind="stable")
        # Only bytes that are genuinely ambiguous are worth erasing.
        return order[conf[order] < 0.5]
