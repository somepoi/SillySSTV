"""Exception hierarchy for SillySSTV."""

from __future__ import annotations


class SillyError(Exception):
    """Base class for all SillySSTV errors."""


class DecodeError(SillyError):
    """Raised when a signal cannot be decoded at all."""


class EncodeError(SillyError):
    """Raised when the requested payload cannot be encoded."""


class ManifestError(DecodeError):
    """Raised when no usable manifest survived in the recording."""
