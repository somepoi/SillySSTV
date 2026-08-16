"""CLI-level tests covering the paths a user actually types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sillysstv.cli import main


def _run(*args: str) -> int:
    return main(list(args))


def test_profiles_and_help(capsys) -> None:
    assert _run("profiles") == 0
    assert "robust" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        _run("--help")
    assert exc.value.code == 0


def test_file_round_trip_through_the_cli(tmp_path: Path, capsys) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(bytes(range(256)) * 6)
    wav = tmp_path / "payload.wav"
    out = tmp_path / "recovered.bin"

    assert _run("encode", str(source), "-o", str(wav), "--profile", "hyper",
                "--ecc", "16") == 0
    assert wav.is_file()

    assert _run("decode", str(wav), "-o", str(out), "--mode", "digital",
                "--profile", "hyper", "--ecc", "16") == 0
    assert out.read_bytes() == source.read_bytes()
    assert "sha256           : OK" in capsys.readouterr().out


def test_image_round_trip_and_damage_through_the_cli(
    tmp_path: Path, picture: Image.Image, capsys
) -> None:
    source = tmp_path / "photo.png"
    picture.save(source)
    wav = tmp_path / "photo.wav"
    broken = tmp_path / "broken.wav"
    out = tmp_path / "out.png"

    assert _run("encode", str(source), "-o", str(wav), "--profile", "hyper",
                "--ecc", "32", "--strip-height", "16") == 0
    assert _run("damage", str(wav), "-o", str(broken), "--snr", "9",
                "--dropouts", "2", "--dropout-ms", "150", "--seed", "3") == 0
    assert _run("decode", str(broken), "-o", str(out), "--mode", "digital",
                "--profile", "hyper", "--ecc", "32") == 0

    recovered = Image.open(out)
    assert recovered.size == picture.size  # damaged, but whole
    assert "strips recovered" in capsys.readouterr().out


def test_analog_round_trip_through_the_cli(
    tmp_path: Path, picture: Image.Image
) -> None:
    source = tmp_path / "photo.png"
    picture.save(source)
    wav = tmp_path / "analog.wav"
    out = tmp_path / "analog.png"

    assert _run("encode", str(source), "-o", str(wav), "--mode", "analog",
                "--scan-ms", "24") == 0
    assert _run("decode", str(wav), "-o", str(out), "--mode", "analog") == 0

    recovered = np.asarray(Image.open(out), np.int16)
    error = np.abs(recovered - np.asarray(picture, np.int16)).mean()
    assert error < 12.0


def test_decode_auto_mode_finds_a_digital_signal(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"auto-detect me" * 40)
    wav = tmp_path / "payload.wav"
    out = tmp_path / "out.bin"

    assert _run("encode", str(source), "-o", str(wav), "--profile", "hyper",
                "--ecc", "16") == 0
    # No --mode, --profile or --ecc: everything is recovered from the signal.
    assert _run("decode", str(wav), "-o", str(out)) == 0
    assert out.read_bytes() == source.read_bytes()


def test_quiet_works_on_either_side_of_the_subcommand(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "x.bin"
    source.write_bytes(b"hello" * 100)
    wav = tmp_path / "x.wav"

    assert _run("encode", str(source), "-o", str(wav), "--profile", "hyper", "-q") == 0
    assert capsys.readouterr().out == ""

    assert _run("-q", "encode", str(source), "-o", str(wav), "--profile", "hyper") == 0
    assert capsys.readouterr().out == ""


def test_missing_input_is_a_clean_error(tmp_path: Path, capsys) -> None:
    assert _run("encode", str(tmp_path / "nope.bin")) == 2
    assert "error:" in capsys.readouterr().err


def test_info_reports_on_a_recording(tmp_path: Path, capsys) -> None:
    source = tmp_path / "x.bin"
    source.write_bytes(b"payload" * 60)
    wav = tmp_path / "x.wav"
    assert _run("encode", str(source), "-o", str(wav), "--profile", "hyper") == 0

    assert _run("info", str(wav), "--search", "2") == 0
    output = capsys.readouterr().out
    assert "digital signal   : profile hyper" in output
