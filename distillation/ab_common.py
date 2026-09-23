# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/ab_common.py (changed: work directory from $LBD_WORK, package imports)
"""Shared plumbing for Experiment 1 (retrieval fidelity of the query encoders).

Every script runs from the repository root as a module, e.g. `python -m distillation.evaluate`.
The desktop instrument is `latent.granular`; the pack format and selection are `corpus_builder`.
All intermediates live under $LBD_WORK/ab (default <repo>/work/ab, git-ignored): they contain
latents and audio derived from the corpora, so they never go into the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from corpus_builder import elkgrain

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("LBD_WORK", HERE.parent / "work"))
AB = SCRATCH / "ab"                      # all intermediates for Experiment 1
CACHE = SCRATCH / "cache" / "latents"    # content-hash latent cache (shared with corpus_builder)

for d in (AB, CACHE):
    d.mkdir(parents=True, exist_ok=True)

_granular = None
_codec = None


def granular():
    global _granular
    if _granular is None:
        from latent import granular as g
        _granular = g
    return _granular


def codec(device: str = "auto"):
    """Lazy shared music2latent codec (MPS with CPU fallback happens at call sites)."""
    global _codec
    if _codec is None:
        _codec = granular().Codec(device=device)
        print(f"music2latent codec on '{_codec.device}'")
    return _codec


def encode_cached(path: str | Path, *, offset: float = 0.0, seconds: float | None = None,
                  tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """load_mono_44k(+peak-norm) then music2latent-encode, content-hash cached.

    Returns (y44 float32, lat float32 [64, T]).  Cache key mirrors build_corpus.cache_key
    but adds offset/seconds so clips and full files never collide.
    """
    g = granular()
    path = Path(path)
    st = path.stat()
    from importlib.metadata import version as pkg_version
    m2l = pkg_version("music2latent")
    raw = (f"{path.resolve()}|{int(st.st_mtime)}|{st.st_size}|{g.CODEC_SR}|norm1"
           f"|off{offset}|sec{seconds}|m2l{m2l}")
    key = hashlib.sha1(raw.encode()).hexdigest()[:16]
    cp = CACHE / f"{path.stem[:24].replace(' ', '_')}{tag}__{key}.npz"
    y44 = g.load_mono_44k(str(path), offset=offset, seconds=seconds)
    if cp.exists():
        lat = np.load(cp)["lat"]
    else:
        c = codec()
        try:
            lat = c.encode(y44)
        except Exception as e:                      # MPS misbehaving -> CPU (slow but correct)
            print(f"  encode failed on '{c.device}' ({e}); retrying on cpu")
            lat = granular().Codec(device="cpu").encode(y44)
        lat = np.asarray(lat, dtype=np.float32)
        np.savez(cp, lat=lat)
    return y44, np.asarray(lat, dtype=np.float32)


def load_pack(path: str | Path) -> elkgrain.ElkGrainPack:
    return elkgrain.ElkGrainPack(path)


def pack_pair_table(pack: elkgrain.ElkGrainPack, corpus_dir: Path, feat_fn) -> dict:
    """Per-grain (features -> raw-latent) pairs for a pack, using pack provenance.

    feat_fn(y44, T) -> dict of per-frame feature arrays [T, d].  Features are computed for
    every latent frame of every source file, then selected by each grain's (file_idx,
    frame_idx) so rows align exactly with pack.raw / pack.emb (unit==1 packs only).
    """
    assert pack.unit == 1, "T06 works on unit=1 packs"
    per_file = []
    for fm in pack.meta["files"]:
        y44 = granular().load_mono_44k(str(corpus_dir / fm["name"]))
        per_file.append(feat_fn(y44, fm["n_frames"]))
    keys = per_file[0].keys()
    fi = pack.grains["file_idx"]
    fr = pack.grains["frame_idx"]
    out = {}
    for k in keys:
        rows = np.empty((pack.n, per_file[0][k].shape[1]), dtype=np.float32)
        for i in range(pack.n):
            rows[i] = per_file[fi[i]][k][fr[i]]
        out[k] = rows
    out["file_idx"] = fi.copy()
    return out


def save_npz(path: Path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))
    print(f"wrote {path}")
