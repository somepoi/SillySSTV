"""SillySSTV - photos and files to sound and back, damage tolerated.

Two transports share one CLI:

``digital``
    Non-coherent M-FSK carrying Reed-Solomon protected blocks.  Works on any
    file.  Damage costs you individual blocks; everything else is bit-exact.
``analog``
    A classic SSTV-style scan-line format for pictures.  Damage becomes visible
    noise rather than missing data -- the image is always whole.
"""

from __future__ import annotations

from .errors import DecodeError, EncodeError, ManifestError, SillyError
from .profiles import DEFAULT_PROFILE, PROFILES, Profile, get_profile

__version__ = "0.1.0"

__all__ = [
    "DecodeError",
    "EncodeError",
    "ManifestError",
    "SillyError",
    "Profile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "get_profile",
    "__version__",
]
