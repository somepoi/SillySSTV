# SillySSTV

Convert photos — and arbitrary files — into sound, and back again. Built so that
**damaged audio still yields whole, merely damaged data**, rather than nothing at all.

Two transports share one CLI:

| mode | carries | how it fails |
|---|---|---|
| **digital** (default) | any file | Reed-Solomon repairs the damage it can; what it can't becomes localised holes. Everything else is bit-exact. |
| **analog** | images only | Classic SSTV scan lines. Noise becomes visible noise, dropouts become smears. The picture always comes out whole. |

## Requirements

**Python 3.11 – 3.14.** Developed and tested on 3.13.

That range is not arbitrary, and the floor matters more than it looks. `requires-python`
decides which numpy and scipy the resolver is *allowed* to pick: it must find versions
that support the entire declared range. Declare `>=3.10` and the newest candidates
become numpy 2.2.x and scipy 1.15.x — the last releases that still support 3.10 — and
those predate CPython 3.14, so they ship no `cp314` wheels. Install on 3.14 and pip
falls back to the sdist and starts **compiling numpy and scipy from source**, which
needs a C and Fortran toolchain and takes many minutes.

Raising the floor to `>=3.11` lets the resolver reach numpy 2.4+ and scipy 1.17+, which
do have `cp311`–`cp314` wheels. The upper end is set by scipy, which has no 3.15 wheels yet.

As a second line of defence, `pyproject.toml` declares:

```toml
[tool.pdm.resolution]
only-binary = ["numpy", "scipy", "pillow"]
```

so `pdm.lock` contains **zero source distributions** for the compiled dependencies. A
missing wheel now produces an immediate resolution error naming the package, instead of
a surprise half-hour build.

If you change `requires-python`, delete `pdm.lock` before re-locking. PDM records the
target range in `[[metadata.targets]]` and reuses it, so an in-place `pdm lock` can
leave the lock targeting the old range while `pyproject.toml` claims the new one.

## Install

```bash
pdm install
```

Then run either `pdm run sillysstv ...` or, inside the venv, `sillysstv ...`.

## Quick start

```bash
sillysstv encode photo.jpg -o photo.wav
```

```bash
sillysstv damage photo.wav -o broken.wav --snr 8 --dropouts 10 --cuts 3
```

```bash
sillysstv decode broken.wav -o recovered.png
```

Decoding is fully self-describing: profile, ECC size, payload type, dimensions and
the original file name all travel inside the signal.

## How the digital mode works

```
file
 └─ payload codec        raw bytes | independent image strips | interleaved raw pixels
     └─ blocks           [kind|index|crc32|data]  →  RS(255, 255-ecc) codeword
         └─ frames       [24-symbol preamble][255-byte codeword]
             └─ M-FSK    16 tones on exact FFT bins, continuous phase
```

Every design choice here exists to keep damage *local*:

- **Self-locating frames.** Each frame carries its own preamble and its own block
  index. The decoder correlates for preambles across the whole file and decodes
  whatever it finds, wherever it finds it — so cuts, splices, added silence and
  reordering cost you only the frames they actually destroy. There is no global
  timing reference to lose.
- **Reed-Solomon per block.** `--ecc 64` (the default) fixes up to 32 corrupted
  bytes in each 255-byte block, or up to 64 when the modem flags them.
- **Soft-decision erasures.** The demodulator scores every symbol by how far the
  winning tone beat its nearest rival. Bytes it was unsure about are handed to
  the RS decoder as erasures, which roughly doubles what it can repair.
- **CRC on every block.** Reed-Solomon can *mis*-correct a block that is damaged
  past its capacity. The CRC catches that, so a broken block becomes a known
  hole instead of silently wrong bytes.
- **Repeated manifest.** The header is sent three times up front and again every
  24 blocks, so losing the start of the recording is survivable.
- **Orthogonal tones.** Tone *k* sits exactly on FFT bin `base + k·step` of a
  symbol-length transform, so demodulation is a plain bin argmax with no
  equaliser, no AGC and no carrier recovery.

### Payload codecs

`--payload` decides *how* the damage shows up.

- `raw` — the file verbatim. Lost blocks are holes at known offsets. Default for
  non-images.
- `image-strips` — the picture is cut into horizontal strips, each compressed
  independently and framed by a self-delimiting record that the decoder scans
  for. A lost block costs you the strips it touched, never the whole picture.
  Default for images.
- `image-raw` — uncompressed pixels, scattered across the stream by an
  interleaver. A lost block becomes isolated missing pixels spread over the
  whole image, which the decoder inpaints from their nearest surviving
  neighbours. Much larger, but by far the most graceful.

Note that `image-strips` is the only option that both compresses *and* keeps
damage local — a plain JPEG sent as `raw` would be destroyed from the first bad
byte onward.

### Modem profiles

```
sillysstv profiles
```

```
robust     43.1 baud  16-FSK     517-  1809 Hz     21.5 B/s raw   narrow band, survives heavy noise
normal     86.1 baud  16-FSK     517-  3101 Hz     43.1 B/s raw   fits a 3 kHz voice channel
fast      172.3 baud  16-FSK     517-  5685 Hz     86.1 B/s raw   good default for files
turbo     344.5 baud  16-FSK     689- 11025 Hz    172.3 B/s raw   needs a clean wideband path
hyper     689.1 baud  16-FSK     689- 21361 Hz    344.5 B/s raw   up to ~21 kHz, lossy codecs will kill it
```

Effective payload rate is the raw rate × `(255 - ecc - 8) / 255`, minus ~4% preamble.
Pick `robust`/`normal` if the audio will pass through anything analog or lossy;
`turbo`/`hyper` only for file-to-file work.

## How the analog mode works

A real SSTV-style format: 1200 Hz line sync, 1500 Hz black, 2300 Hz white,
sequential R/G/B scans (or a single luminance scan with `--analog-mode mono`).
There is no FEC and none is wanted — the whole point is that degradation stays
proportional.

The one part that must survive verbatim is the header (mode, dimensions, scan
timing), so it goes out as slow 1100/1300 Hz FSK, CRC-16 protected, repeated
three times. The decoder brute-forces the header offset against the CRC rather
than relying on leader-tone detection, then tracks each line's sync pulse,
*predicting* through regions where the sync is destroyed so the image stays
geometrically intact instead of tearing.

```bash
sillysstv encode photo.jpg -o photo.wav --mode analog --max-size 320
sillysstv decode photo.wav -o out.png
```

Default timing is Martin M1-ish: about 114 s for a 320×256 colour image.

## Damage simulator

```bash
sillysstv damage in.wav -o out.wav \
    --snr 6 \            # white noise to this SNR in dB
    --dropouts 10 \      # silent gaps
    --dropout-ms 250 \
    --cuts 3 \           # chunks removed entirely (shifts everything after)
    --cut-ms 500 \
    --clip 0.3 \         # hard clipping
    --hum 50 \           # mains hum
    --bits 4             # bit crush
```

`--cuts` is the interesting one: it shifts every later frame, which is exactly
what a global timing reference cannot survive and self-locating frames can.

## Inspecting a file

```bash
sillysstv info recording.wav
```

Reports what libsndfile sees, which digital profile correlates best, and whether
an analog header can be found.

## Limitations

- The digital decoder assumes the recording is at its original sample rate;
  resampling changes the symbol length in samples and will break frame sync.
- `hyper` puts tones up near 21 kHz. It is fine for WAV-to-WAV, but MP3/Opus and
  most real audio paths will remove them.
- Reed-Solomon here is pure Python. Installing the optional `creedsolo` Cython
  extension speeds up encode and decode considerably; SillySSTV uses it
  automatically when present. Note that `creedsolo` *is* a source build — it is
  deliberately left out of the dependency list for that reason.
- scipy has no CPython 3.15 wheels yet, so 3.15 is out of range until it does.

## Development

```bash
pdm run pytest
```

## Licence

MIT.
