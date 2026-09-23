#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/build_corpus.py (changed: imports, cache path, usage lines)
"""Build a .elkgrain corpus pack from a folder of recordings (T05, offline laptop tool).

    python -m corpus_builder.build_corpus <corpus_folder> -o <name>.elkgrain --unit 1 --stride 1

Pipeline (mirrors the desktop instrument, latent/granular.py):
    load mono @44.1k, per-file peak-norm      (granular.load_mono_44k — imported, not copied)
    encode with music2latent at native 44.1k  (granular.Codec, MPS with CPU fallback; cached)
    grain windows: unit in {1,2,4} frames     (granular.grains_from_latents via build_codebook)
    drop near-silent grains                   (build_codebook's RMS gate, default -50 dBFS)
    matching table + raw latents              (granular.build_codebook — bit-identical to live)
    audio grains @48k int16                   (board is 48k fixed; scipy polyphase resample,
                                               deduplicated overlap regions + crossfade margins)
    single mmap-friendly pack                 (elkgrain.write_pack; spec in docs/pack-format.md)

Latents are content-hash cached (path|mtime|size|params), CorpusManager-style, so rebuilds with
different unit/stride/margin do not re-encode.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from corpus_builder import elkgrain
from latent import granular as _granular

AUDIO_EXT = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a")  # == live_engine.AUDIO_EXT
TOOL_VERSION = "0.1.0"
# latent cache: $LBD_WORK/cache/latents (default <repo>/work, git-ignored)
DEFAULT_CACHE = Path(os.environ.get("LBD_WORK", Path(__file__).resolve().parents[1] / "work")) / "cache" / "latents"


# --------------------------------------------------------------------------- helpers
def find_audio(folder: Path) -> list[Path]:
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXT]
    return files


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(path: Path, codec_sr: int, normalize: bool, m2l_version: str) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{int(st.st_mtime)}|{st.st_size}|{codec_sr}|norm{int(normalize)}|m2l{m2l_version}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def keep_mask(y44: np.ndarray, lat: np.ndarray, unit: int, stride: int,
              silence_db: float | None) -> np.ndarray:
    """EXACTLY build_codebook's silence gate (granular.py:99-105) — do not modify.

    Same expressions, same dtypes (float64 rms/db), so the mask matches the codebook the
    original code builds bit for bit. Returns keep flags over grain start frames
    range(0, T-unit+1, stride)."""
    T = lat.shape[1]
    n_starts = len(range(0, T - unit + 1, stride))
    if silence_db is None:
        return np.ones(n_starts, dtype=bool)
    a = np.asarray(y44, dtype=np.float32)
    hop = len(a) / lat.shape[1]                                    # samples per latent (~4106)
    rms = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(T)])
    db = 20 * np.log10(rms + 1e-9)
    return np.array([np.max(db[i:i + unit]) > silence_db for i in range(0, T - unit + 1, stride)])


def to_int16(y: np.ndarray) -> np.ndarray:
    """float [-1,1] -> int16, round-half-away like C lrintf on the write side is NOT what numpy
    does; we use rint (banker's) and document decode as x = s / 32768.0 (see pack-format.md)."""
    return np.clip(np.rint(np.asarray(y, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def slice_zero_fill(y: np.ndarray, a: int, b: int) -> np.ndarray:
    """y[a:b] with zeros outside [0, len(y)) — margins at file edges."""
    out = np.zeros(b - a, dtype=y.dtype)
    lo, hi = max(a, 0), min(b, len(y))
    if hi > lo:
        out[lo - a:hi - a] = y[lo:hi]
    return out


# --------------------------------------------------------------------------- build
def build_pack(corpus_dir: str | Path, out_path: str | Path, unit: int = 1, stride: int = 1,
               silence_db: float | None = -50.0, margin_ms: float = 10.0, device: str = "auto",
               cache_dir: str | Path = DEFAULT_CACHE, log=print) -> tuple[elkgrain.Header, dict]:
    t_start = time.perf_counter()
    corpus_dir, out_path = Path(corpus_dir), Path(out_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    granular = _granular
    from importlib.metadata import version as pkg_version  # noqa: PLC0415
    m2l_version = pkg_version("music2latent")

    files = find_audio(corpus_dir)
    if not files:
        raise SystemExit(f"no audio files under {corpus_dir} (extensions: {AUDIO_EXT})")
    log(f"corpus: {len(files)} file(s) under {corpus_dir}")

    codecs: dict[str, object] = {}   # lazy: only constructed on a cache miss
    codec_device = None

    def get_codec(dev):
        nonlocal codec_device
        if dev not in codecs:
            log(f"loading music2latent codec (requested device '{dev}') …")
            codecs[dev] = granular.Codec(device=dev)
        codec_device = codecs[dev].device
        return codecs[dev]

    # ---- per-file: load, encode (cached), gate, resample --------------------
    y44_list, lat_list, y48_list, file_meta = [], [], [], []
    encode_s = 0.0
    for k, path in enumerate(files):
        y44 = granular.load_mono_44k(str(path))                    # per-file peak-norm to 0.9
        if len(y44) < granular.CODEC_SR // 2:
            log(f"  [{k}] SKIP (<0.5 s): {path.name}")
            continue
        ck = cache_key(path, granular.CODEC_SR, True, m2l_version)
        cp = cache_dir / f"{path.stem[:24].replace(' ', '_')}__{ck}.npz"
        if cp.exists():
            lat = np.load(cp)["lat"]
            log(f"  [{k}] cached latents  {path.name}  [{lat.shape[0]},{lat.shape[1]}]")
        else:
            t0 = time.perf_counter()
            try:
                lat = get_codec(device).encode(y44)
            except Exception as e:  # MPS misbehaving -> CPU (slow but correct)
                if (device in ("auto", "mps")) and codec_device != "cpu":
                    log(f"  [{k}] encode failed on '{codec_device}' ({e}); retrying on cpu")
                    lat = get_codec("cpu").encode(y44)
                else:
                    raise
            lat = np.asarray(lat, dtype=np.float32)
            dt = time.perf_counter() - t0
            encode_s += dt
            np.savez(cp, lat=lat)
            log(f"  [{k}] encoded {len(y44)/granular.CODEC_SR:6.1f}s -> [{lat.shape[0]},{lat.shape[1]}] "
                f"in {dt:.1f}s ({codec_device})  {path.name}")
        y48 = to_int16(granular.resample(y44, granular.CODEC_SR, 48000))
        y44_list.append(y44)
        lat_list.append(lat)
        y48_list.append(y48)
        file_meta.append({"i": len(file_meta), "name": str(path.relative_to(corpus_dir)),
                          "sha1": file_sha1(path), "seconds44": round(len(y44) / granular.CODEC_SR, 3),
                          "n_frames": int(lat.shape[1]), "len48": int(len(y48))})
    if not lat_list:
        raise SystemExit("no usable audio files (all too short?)")

    # ---- codebook via the ORIGINAL build_codebook (bit-identical to the live engine) ----
    cb = granular.build_codebook(lat_list, unit, stride, name=out_path.stem,
                                 sources=[m["name"] for m in file_meta],
                                 seconds=sum(m["seconds44"] for m in file_meta),
                                 audio_list=y44_list, silence_db=silence_db)
    log(f"codebook: {cb.n} grains (unit={unit}, stride={stride}), "
        f"{cb.dropped_silent} silent dropped, built in {cb.build_s:.2f}s")
    if cb.n == 0:
        raise SystemExit("codebook is empty — corpus all silence? lower --silence-db")

    # ---- provenance mapping: reproduce the gate per file, verify bit-equality ----
    keeps = [keep_mask(y44, lat, unit, stride, silence_db) for y44, lat in zip(y44_list, lat_list)]
    my_raw = np.concatenate([granular.grains_from_latents(lat, unit, stride)[kp]
                             for lat, kp in zip(lat_list, keeps)], axis=0) \
        if any(kp.any() for kp in keeps) else np.zeros((0, 64 * unit), np.float32)
    if my_raw.shape != cb.raw.shape or my_raw.tobytes() != cb.raw.tobytes():
        raise AssertionError("provenance gate diverged from build_codebook — pack would misattribute grains")

    # ---- audio block: per-grain windows on the 48k grid, deduplicated regions ----
    hop48 = int(round(sum(m["len48"] for m in file_meta) / sum(m["n_frames"] for m in file_meta)))
    margin48 = int(round(margin_ms * 48000.0 / 1000.0))
    total_len48 = unit * hop48 + 2 * margin48
    audio_parts, records = [], []
    base = 0
    n_regions = 0
    for fi, (y48, lat, kp, fm) in enumerate(zip(y48_list, lat_list, keeps, file_meta)):
        T = lat.shape[1]
        hop48_eff = len(y48) / T                       # per-file effective hop (no drift)
        fm["hop48_eff"] = round(hop48_eff, 4)
        frames = np.arange(0, T - unit + 1, stride)[kp]
        fm["n_grains_kept"] = int(len(frames))
        fm["n_grains_dropped"] = int((~kp).sum())
        if len(frames) == 0:
            continue
        starts48 = np.round(frames * hop48_eff).astype(np.int64)   # core starts, exact per file
        wins = starts48 - margin48                                  # stored-window starts (ascending)
        regions: list[list[int]] = []                               # [a, b) in file timeline
        for w in wins:
            w = int(w)
            if regions and w <= regions[-1][1]:
                regions[-1][1] = max(regions[-1][1], w + total_len48)
            else:
                regions.append([w, w + total_len48])
        region_base = {}
        for a, b in regions:
            audio_parts.append(slice_zero_fill(y48, a, b))
            region_base[(a, b)] = base
            base += b - a
        n_regions += len(regions)
        ri = 0
        for w, fr in zip(wins, frames):
            w = int(w)
            while regions[ri][1] < w + total_len48:
                ri += 1
            a, b = regions[ri]
            records.append((region_base[(a, b)] + (w - a), fi, int(fr)))
    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, np.int16)
    grain_records = np.array(records, dtype=elkgrain.GRAIN_REC_DTYPE)
    assert len(grain_records) == cb.n, (len(grain_records), cb.n)

    naive = cb.n * total_len48
    meta = {
        "format": "elkgrain", "format_version": elkgrain.VERSION,
        "tool": {"name": "corpus_builder", "version": TOOL_VERSION,
                 "music2latent": m2l_version, "numpy": np.__version__,
                 "codec_device": codec_device or "cache-only"},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build": {"corpus_dir": str(corpus_dir.resolve()), "unit": unit, "stride": stride,
                  "silence_db": silence_db, "margin_ms": margin_ms, "normalize": True,
                  "codec_sr": granular.CODEC_SR, "resampler": "scipy.signal.resample_poly 160/147"},
        "files": file_meta,
        "stats": {"n_grains": cb.n, "dropped_silent": int(cb.dropped_silent),
                  "corpus_seconds44": round(cb.seconds, 2),
                  "audio_samples": int(audio.size),
                  "audio_seconds48": round(audio.size / 48000.0, 2),
                  "n_regions": n_regions,
                  "dedup_ratio": round(audio.size / naive, 4) if naive else 1.0,
                  "encode_s": round(encode_s, 2)},
    }
    hdr = elkgrain.write_pack(out_path, cb.book, cb.raw, cb.centre, audio, grain_records,
                              unit=unit, hop48=hop48, margin48=margin48, n_files=len(file_meta),
                              meta=meta)
    total_s = time.perf_counter() - t_start
    sz = out_path.stat().st_size
    log(f"\nwrote {out_path}  ({sz/1e6:.1f} MB, {total_s:.1f}s total, {encode_s:.1f}s encoding)")
    log(f"  grains        : {cb.n} (unit={unit} => {unit*hop48/48000*1000:.0f} ms core, "
        f"margin {margin48/48.0:.1f} ms each side)")
    log(f"  matching table: {cb.book.nbytes/1e6:6.2f} MB   raw latents: {cb.raw.nbytes/1e6:6.2f} MB")
    log(f"  audio block   : {audio.nbytes/1e6:6.2f} MB ({audio.size/48000.0:.1f}s, "
        f"dedup x{naive/max(audio.size,1):.2f} vs per-grain copies)")
    log(f"  grain table   : {grain_records.nbytes/1e6:6.2f} MB   json: {hdr.json_len/1e3:.1f} kB")
    per_min = (sz / max(cb.seconds, 1e-9)) * 60 / 1e6
    log(f"  density       : {per_min:.1f} MB per corpus-minute -> 30 min ~ {per_min*30:.0f} MB")
    return hdr, meta


# --------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("corpus", help="folder of recordings (wav/aif/flac/mp3/ogg/m4a, recursive)")
    ap.add_argument("-o", "--out", required=True, help="output pack path (.elkgrain)")
    ap.add_argument("--unit", type=int, default=1, choices=[1, 2, 4],
                    help="latent frames per grain (~93/186/372 ms)")
    ap.add_argument("--stride", type=int, default=1, help="grain window stride in frames")
    ap.add_argument("--silence-db", default="-50",
                    help="drop grains whose per-frame max dBFS is below this; 'none' disables")
    ap.add_argument("--margin-ms", type=float, default=10.0,
                    help="crossfade margin stored on each side of every grain")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"],
                    help="music2latent device (auto = MPS if available, CPU fallback on error)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="latent cache (content-hash keyed, CorpusManager-style)")
    args = ap.parse_args(argv)
    silence_db = None if str(args.silence_db).lower() == "none" else float(args.silence_db)
    build_pack(args.corpus, args.out, unit=args.unit, stride=args.stride, silence_db=silence_db,
               margin_ms=args.margin_ms, device=args.device, cache_dir=args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
