"""Command line interface for SillySSTV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from . import __version__, analog, audio, damage, digital, payloads
from .errors import SillyError
from .profiles import DEFAULT_PROFILE, PROFILES, get_profile


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


class Reporter:
    """Single-line progress on stderr, silenced with --quiet."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._last = 0.0
        self._dirty = False

    def __call__(self, stage: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if done < total and now - self._last < 0.1:
            return
        self._last = now
        pct = 100.0 * done / total if total else 100.0
        sys.stderr.write(f"\r  {stage:<9} {pct:5.1f}%  ")
        sys.stderr.flush()
        self._dirty = True

    def done(self) -> None:
        if self._dirty:
            sys.stderr.write("\r" + " " * 32 + "\r")
            sys.stderr.flush()
            self._dirty = False


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(message)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #


def cmd_encode(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_file():
        raise SillyError(f"input file not found: {source}")
    output = Path(args.output) if args.output else source.with_suffix(".wav")
    reporter = Reporter(not args.quiet)

    if args.mode == "analog":
        image = payloads.load_image(str(source), args.max_size, args.gray)
        signal, fmt = analog.encode(
            image, args.analog_mode, args.scan_ms, args.sample_rate
        )
        audio.write_wav(output, signal, args.sample_rate, args.subtype)
        info(
            f"analog {analog.MODE_NAMES[fmt.mode]}  {fmt.width}x{fmt.height}  "
            f"{len(signal) / args.sample_rate:.1f} s  -> {output}",
            args.quiet,
        )
        return 0

    kind = payloads.guess_kind(str(source), args.payload)
    payload = payloads.encode_file(
        str(source),
        kind,
        strip_height=args.strip_height,
        quality=args.quality,
        strip_format=args.strip_format,
        max_size=args.max_size,
        grayscale=args.gray,
    )
    profile = get_profile(args.profile)
    result = digital.encode(
        payload,
        profile,
        args.sample_rate,
        ecc=args.ecc,
        name=source.name,
        progress=reporter,
    )
    reporter.done()
    audio.write_wav(output, result.signal, args.sample_rate, args.subtype)

    overhead = 100.0 * (result.n_frames * 255) / max(len(payload.stream), 1) - 100.0
    info(
        f"digital {profile.name}  payload {kind}  {len(payload.stream)} B  "
        f"{result.n_frames} frames  ecc {args.ecc}  "
        f"+{overhead:.0f}% wire overhead\n"
        f"{result.duration:.1f} s @ {args.sample_rate} Hz  -> {output}",
        args.quiet,
    )
    return 0


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def _write_decoded(
    decoded: payloads.DecodedPayload, output: Path | None, fallback_name: str, quiet: bool
) -> None:
    if decoded.image is not None:
        target = output or Path(fallback_name).with_suffix(".png")
        if target.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".webp"}:
            target = target.with_suffix(".png")
        decoded.image.save(target)
    else:
        target = output or Path(fallback_name or "recovered.bin")
        target.write_bytes(decoded.data or b"")
    info(f"wrote {target}", quiet)


def cmd_decode(args: argparse.Namespace) -> int:
    signal, sample_rate = audio.read_wav(args.input)
    reporter = Reporter(not args.quiet)
    output = Path(args.output) if args.output else None

    if args.mode in ("analog", "auto"):
        try:
            image, stats = analog.decode(signal, sample_rate, args.search)
        except SillyError:
            if args.mode == "analog":
                raise
        else:
            info(stats.render(), args.quiet)
            target = output or Path(args.input).with_suffix(".png")
            image.save(target)
            info(f"wrote {target}", args.quiet)
            return 0

    candidates = (
        [get_profile(args.profile)] if args.profile != "auto" else list(PROFILES.values())
    )
    result = digital.decode(
        signal,
        sample_rate,
        profile=None if args.profile == "auto" else candidates[0],
        profile_candidates=candidates,
        ecc=None if args.ecc == "auto" else int(args.ecc),
        threshold=args.threshold,
        progress=reporter,
    )
    reporter.done()
    info(result.stats.render(), args.quiet)

    manifest = result.manifest
    decoded = payloads.decode_payload(
        result.stream,
        result.valid,
        manifest.payload_kind if manifest else payloads.KIND_RAW,
        manifest.extra if manifest else {},
        inpaint=not args.no_inpaint,
    )
    if decoded.detail:
        info(f"payload          : {decoded.detail}", args.quiet)

    fallback = (manifest.name if manifest and manifest.name else "recovered.bin")
    _write_decoded(decoded, output, fallback, args.quiet)
    return 0


# --------------------------------------------------------------------------- #
# damage / info / profiles
# --------------------------------------------------------------------------- #


def cmd_damage(args: argparse.Namespace) -> int:
    signal, sample_rate = audio.read_wav(args.input)
    spec = damage.DamageSpec(
        snr_db=args.snr,
        dropouts=args.dropouts,
        dropout_ms=args.dropout_ms,
        cuts=args.cuts,
        cut_ms=args.cut_ms,
        clip=args.clip,
        hum_hz=args.hum,
        bit_depth=args.bits,
        seed=args.seed,
    )
    damaged = damage.apply(signal, sample_rate, spec)
    audio.write_wav(args.output, damaged, sample_rate, args.subtype, peak=0.98)
    info(f"applied {spec.describe()}\nwrote {args.output}", args.quiet)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print(f"{args.input}: {audio.describe(args.input)}")
    signal, sample_rate = audio.read_wav(args.input)
    try:
        profile, score = digital.detect_profile(
            signal, sample_rate, list(PROFILES.values())
        )
        print(f"digital signal   : profile {profile.name} (correlation {score:.2f})")
    except SillyError as exc:
        print(f"digital signal   : {exc}")
    try:
        _, stats = analog.decode(signal, sample_rate, args.search)
        print("analog signal    :")
        print(stats.render())
    except SillyError as exc:
        print(f"analog signal    : {exc}")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    print(f"digital modem profiles at {args.sample_rate} Hz:\n")
    for profile in PROFILES.values():
        print("  " + profile.summary(args.sample_rate))
    print(
        "\nEffective payload rate is roughly the raw rate times "
        "(255 - ecc - 8) / 255, minus ~4% preamble."
    )
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sillysstv",
        description="Convert photos and files to sound and back, damage tolerated.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  sillysstv encode photo.jpg -o photo.wav\n"
            "  sillysstv encode photo.jpg -o photo.wav --mode analog\n"
            "  sillysstv encode archive.zip -o archive.wav --profile turbo --ecc 96\n"
            "  sillysstv damage photo.wav -o broken.wav --snr 6 --dropouts 8\n"
            "  sillysstv decode broken.wav -o recovered.png\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"sillysstv {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Accepted on either side of the subcommand. SUPPRESS keeps an unused
    # subcommand flag from clobbering a --quiet given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress reports",
    )

    # -- encode ------------------------------------------------------------- #
    enc = subparsers.add_parser("encode", parents=[common], help="turn a file into audio")
    enc.add_argument("input")
    enc.add_argument("-o", "--output", help="output WAV (default: input name + .wav)")
    enc.add_argument(
        "--mode",
        choices=["digital", "analog"],
        default="digital",
        help="digital = FEC-protected bits (any file); analog = SSTV scan lines (images)",
    )
    enc.add_argument(
        "--profile",
        choices=list(PROFILES),
        default=DEFAULT_PROFILE,
        help="digital modem speed/robustness trade-off",
    )
    enc.add_argument(
        "--ecc",
        type=_positive_int,
        default=64,
        help="Reed-Solomon symbols per 255-byte block (default 64 = 25%% redundancy)",
    )
    enc.add_argument(
        "--payload",
        choices=["auto", "raw", "image-strips", "image-raw"],
        default="auto",
        help="how the file is turned into a byte stream",
    )
    enc.add_argument("--strip-height", type=_positive_int, default=32)
    enc.add_argument("--strip-format", choices=["jpeg", "png"], default="jpeg")
    enc.add_argument("--quality", type=int, default=80, help="JPEG quality for strips")
    enc.add_argument("--max-size", type=int, default=None, help="downscale images to fit")
    enc.add_argument("--gray", action="store_true", help="convert images to grayscale")
    enc.add_argument("--analog-mode", choices=["mono", "rgb"], default="rgb")
    enc.add_argument("--scan-ms", type=float, default=None, help="analog scan time/line")
    enc.add_argument("--sample-rate", type=_positive_int, default=44100)
    enc.add_argument("--subtype", default="PCM_16", help="WAV sample format")
    enc.set_defaults(func=cmd_encode)

    # -- decode ------------------------------------------------------------- #
    dec = subparsers.add_parser(
        "decode", parents=[common], help="recover a file from audio"
    )
    dec.add_argument("input")
    dec.add_argument("-o", "--output")
    dec.add_argument("--mode", choices=["auto", "digital", "analog"], default="auto")
    dec.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    dec.add_argument("--ecc", default="auto")
    dec.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="preamble correlation needed to accept a frame (0-1)",
    )
    dec.add_argument(
        "--search", type=float, default=60.0, help="seconds to scan for an analog header"
    )
    dec.add_argument(
        "--no-inpaint",
        action="store_true",
        help="leave holes grey instead of filling from neighbours (image-raw)",
    )
    dec.set_defaults(func=cmd_decode)

    # -- damage ------------------------------------------------------------- #
    dmg = subparsers.add_parser(
        "damage", parents=[common], help="degrade a recording on purpose"
    )
    dmg.add_argument("input")
    dmg.add_argument("-o", "--output", required=True)
    dmg.add_argument("--snr", type=float, default=None, help="target SNR in dB")
    dmg.add_argument("--dropouts", type=int, default=0, help="number of silent gaps")
    dmg.add_argument("--dropout-ms", type=float, default=200.0)
    dmg.add_argument("--cuts", type=int, default=0, help="number of removed chunks")
    dmg.add_argument("--cut-ms", type=float, default=200.0)
    dmg.add_argument("--clip", type=float, default=None, help="hard clip threshold")
    dmg.add_argument("--hum", type=float, default=None, help="add a hum at this Hz")
    dmg.add_argument("--bits", type=int, default=None, help="requantise to N bits")
    dmg.add_argument("--seed", type=int, default=0)
    dmg.add_argument("--subtype", default="PCM_16")
    dmg.set_defaults(func=cmd_damage)

    # -- info / profiles ---------------------------------------------------- #
    nfo = subparsers.add_parser("info", help="inspect a recording")
    nfo.add_argument("input")
    nfo.add_argument("--search", type=float, default=60.0)
    nfo.set_defaults(func=cmd_info)

    pro = subparsers.add_parser("profiles", help="list digital modem profiles")
    pro.add_argument("--sample-rate", type=_positive_int, default=44100)
    pro.set_defaults(func=cmd_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SillyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, Image.UnidentifiedImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
