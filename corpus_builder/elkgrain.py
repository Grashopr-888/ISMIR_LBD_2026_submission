# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/elkgrain.py (changed: spec path)
""".elkgrain v1 — single-file, mmap-friendly corpus pack for the Elk Stomp latent-granular port.

Byte-exact specification: docs/pack-format.md
(the C++ loader, plugin/src/ElkgrainPack.cpp, is written from that document; keep the two in lockstep).

Layout (all little-endian, every binary block 128-byte aligned — alignment is LOAD-BEARING:
Apple Accelerate's sgemv gives bit-different (1 ULP) sums for matrix base pointers that are
not 0 mod 128, so a 64-aligned mmap'd matching table would not reproduce the picks of the
same table in a fresh numpy allocation; mmap bases are page-aligned on macOS and Linux, so
128-aligned file offsets keep mmap'd matching bit-identical to malloc'd matching):

    [0..128)   fixed header  "<8s14I8Q"
    [128..)    JSON metadata (UTF-8)
    ...        grain table    N x {audio_pos u64, file_idx u32, frame_idx u32}  (16 B/grain)
    ...        centre         f32 [grain_dim]
    ...        embeddings     f32 [N, grain_dim]   mean-centred + L2-normalised (matching table)
    ...        raw latents    f32 [N, grain_dim]   raw grains (rung-C / debug)
    ...        audio          i16 [audio_samples]  48 kHz mono, deduplicated regions

Per grain, `audio_pos` indexes the audio block (in samples, not bytes) at the start of that
grain's stored window of exactly `total_len48 = unit*hop48 + 2*margin48` samples:
margin | core (grain_len48 = unit*hop48) | margin. Windows of neighbouring grains OVERLAP
inside shared regions — the audio block stores each source sample at most once.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"ELKGRAIN"
VERSION = 1
HEADER_BYTES = 128
HEADER_FMT = "<8s14I8Q"          # magic + 14 u32 + 8 u64 = 8 + 56 + 64 = 128
FLAG_HAS_RAW = 1 << 0
ALIGN = 128            # see module docstring: BLAS bit-parity requires 0 mod 128 pointers

GRAIN_REC_DTYPE = np.dtype([("audio_pos", "<u8"), ("file_idx", "<u4"), ("frame_idx", "<u4")])
assert GRAIN_REC_DTYPE.itemsize == 16

assert struct.calcsize(HEADER_FMT) == HEADER_BYTES


def _align(off: int, a: int = ALIGN) -> int:
    return (off + a - 1) // a * a


@dataclass
class Header:
    version: int
    flags: int
    n_grains: int
    latent_dim: int
    unit: int
    grain_dim: int
    hop48: int
    margin48: int
    grain_len48: int
    total_len48: int
    sample_rate: int
    n_files: int
    json_off: int
    json_len: int
    emb_off: int
    raw_off: int
    centre_off: int
    audio_off: int
    audio_samples: int
    grains_off: int


def write_pack(path: str | Path,
               emb: np.ndarray,            # f32 [N, grain_dim] mean-centred + L2 (matching table)
               raw: np.ndarray,            # f32 [N, grain_dim] raw latent grains
               centre: np.ndarray,         # f32 [grain_dim]
               audio: np.ndarray,          # i16 [audio_samples] 48 kHz mono
               grain_records: np.ndarray,  # GRAIN_REC_DTYPE [N]
               *, unit: int, hop48: int, margin48: int, n_files: int,
               latent_dim: int = 64, sample_rate: int = 48000,
               meta: dict | None = None) -> Header:
    """Write a v1 .elkgrain pack. Arrays are written bit-exactly (no rescaling)."""
    emb = np.ascontiguousarray(emb, dtype="<f4")
    raw = np.ascontiguousarray(raw, dtype="<f4")
    centre = np.ascontiguousarray(centre, dtype="<f4")
    audio = np.ascontiguousarray(audio, dtype="<i2")
    grain_records = np.ascontiguousarray(grain_records, dtype=GRAIN_REC_DTYPE)

    n = int(emb.shape[0])
    grain_dim = int(latent_dim * unit)
    if emb.shape != (n, grain_dim) or raw.shape != (n, grain_dim):
        raise ValueError(f"emb/raw must be [N,{grain_dim}], got {emb.shape}/{raw.shape}")
    if centre.shape != (grain_dim,):
        raise ValueError(f"centre must be [{grain_dim}], got {centre.shape}")
    if grain_records.shape != (n,):
        raise ValueError(f"grain_records must be [N], got {grain_records.shape}")
    total_len48 = unit * hop48 + 2 * margin48
    end_ok = (grain_records["audio_pos"].astype(np.uint64) + np.uint64(total_len48)
              <= np.uint64(audio.size)) if n else np.array([True])
    if not bool(np.all(end_ok)):
        raise ValueError("grain audio_pos + total_len48 overruns the audio block")

    js = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_off = HEADER_BYTES
    grains_off = _align(json_off + len(js))
    centre_off = _align(grains_off + grain_records.nbytes)
    emb_off = _align(centre_off + centre.nbytes)
    raw_off = _align(emb_off + emb.nbytes)
    audio_off = _align(raw_off + raw.nbytes)

    hdr = Header(version=VERSION, flags=FLAG_HAS_RAW, n_grains=n, latent_dim=latent_dim,
                 unit=unit, grain_dim=grain_dim, hop48=hop48, margin48=margin48,
                 grain_len48=unit * hop48, total_len48=total_len48, sample_rate=sample_rate,
                 n_files=n_files, json_off=json_off, json_len=len(js), emb_off=emb_off,
                 raw_off=raw_off, centre_off=centre_off, audio_off=audio_off,
                 audio_samples=int(audio.size), grains_off=grains_off)

    packed = struct.pack(HEADER_FMT, MAGIC, hdr.version, HEADER_BYTES, hdr.flags, hdr.n_grains,
                         hdr.latent_dim, hdr.unit, hdr.grain_dim, hdr.hop48, hdr.margin48,
                         hdr.grain_len48, hdr.total_len48, hdr.sample_rate, hdr.n_files, 0,
                         hdr.json_off, hdr.json_len, hdr.emb_off, hdr.raw_off, hdr.centre_off,
                         hdr.audio_off, hdr.audio_samples, hdr.grains_off)

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(packed)
        f.write(js)
        for off, blob in ((grains_off, grain_records.tobytes()),
                          (centre_off, centre.tobytes()),
                          (emb_off, emb.tobytes()),
                          (raw_off, raw.tobytes()),
                          (audio_off, audio.tobytes())):
            f.write(b"\x00" * (off - f.tell()))
            f.write(blob)
    tmp.replace(path)
    return hdr


class ElkGrainPack:
    """Read/mmap a .elkgrain pack. All arrays are zero-copy read-only views into the mapping."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        buf = self._mm
        fields = struct.unpack_from(HEADER_FMT, buf, 0)
        if fields[0] != MAGIC:
            raise ValueError(f"{path}: bad magic {fields[0]!r}")
        if fields[1] != VERSION:
            raise ValueError(f"{path}: unsupported version {fields[1]}")
        if fields[2] != HEADER_BYTES:
            raise ValueError(f"{path}: unexpected header size {fields[2]}")
        (self.version, _hb, self.flags, self.n, self.latent_dim, self.unit, self.grain_dim,
         self.hop48, self.margin48, self.grain_len48, self.total_len48, self.sample_rate,
         self.n_files, _pad) = fields[1:15]
        (self.json_off, self.json_len, self.emb_off, self.raw_off, self.centre_off,
         self.audio_off, self.audio_samples, self.grains_off) = fields[15:23]

        self.meta = json.loads(bytes(buf[self.json_off:self.json_off + self.json_len]) or b"{}")

        def view(off, dtype, count):
            a = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
            a.flags.writeable = False
            return a

        gd, n = self.grain_dim, self.n
        self.grains = view(self.grains_off, GRAIN_REC_DTYPE, n)
        self.centre = view(self.centre_off, "<f4", gd)
        self.emb = view(self.emb_off, "<f4", n * gd).reshape(n, gd)
        self.raw = (view(self.raw_off, "<f4", n * gd).reshape(n, gd)
                    if self.flags & FLAG_HAS_RAW else None)
        self.audio = view(self.audio_off, "<i2", self.audio_samples)

    # -- convenience ---------------------------------------------------------
    def grain_window(self, i: int) -> np.ndarray:
        """int16 view: margin | core | margin (total_len48 samples)."""
        p = int(self.grains["audio_pos"][i])
        return self.audio[p:p + self.total_len48]

    def grain_core(self, i: int) -> np.ndarray:
        """int16 view of the core grain_len48 samples (frame-grid aligned)."""
        p = int(self.grains["audio_pos"][i]) + self.margin48
        return self.audio[p:p + self.grain_len48]

    def close(self):
        self._mm = None  # views die with the mapping when GC'd

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        mb = self.path.stat().st_size / 1e6
        return (f"ElkGrainPack({self.path.name}: {self.n} grains, unit={self.unit}, "
                f"hop48={self.hop48}, margin48={self.margin48}, {self.n_files} files, "
                f"{self.audio_samples / self.sample_rate:.1f}s audio, {mb:.1f} MB)")
