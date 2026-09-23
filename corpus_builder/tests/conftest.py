# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/tests/conftest.py (changed: synthetic corpus only, imports, paths)
"""Fixtures: a small synthesized test corpus, session-built .elkgrain packs, and target latents.

The original fixtures copied three dry recordings from the author's hackathon repository; the public
version always synthesizes (chirps, noise bursts, tones, each with a silent middle for the gate).
Everything heavyweight (music2latent encodes) is content-hash cached in $LBD_WORK/cache/latents,
so the first session pays a few seconds of encoding and later runs are faster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corpus_builder import build_corpus
from corpus_builder import elkgrain  # noqa: F401
from latent import granular as _granular

DATA = build_corpus.DEFAULT_CACHE.parents[1] / "testdata"

SYNTH_CORPUS = {"chirp.wav": ("chirp", 1), "noise.wav": ("noise", 2), "tones.wav": ("tones", 3)}


def _synth(path: Path, kind: str, seconds: float = 8.0, sr: int = 44100, seed: int = 0):
    """Test audio: chirps + noise bursts with a silent gap (exercises the gate)."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    if kind == "chirp":
        y = 0.7 * np.sin(2 * np.pi * (120 + 40 * t) * t)
    elif kind == "noise":
        y = np.zeros(n, dtype=np.float64)
        for s in np.linspace(0.2, seconds - 1.2, 8):
            i = int(s * sr)
            burst = rng.standard_normal(sr // 3) * np.hanning(sr // 3)
            y[i:i + len(burst)] += 0.6 * burst
    else:  # tones
        y = 0.5 * (np.sin(2 * np.pi * 330 * t) + 0.5 * np.sin(2 * np.pi * 495 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    gap = slice(int(0.45 * n), int(0.55 * n))
    y[gap] = 0.0                                        # silent middle: silence gate has work to do
    sf.write(path, y.astype(np.float32), sr)


@pytest.fixture(scope="session")
def granular():
    return _granular


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    d = DATA / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    for dst, (kind, seed) in SYNTH_CORPUS.items():
        if not (d / dst).exists():
            _synth(d / dst, kind=kind, seed=seed)
    return d


@pytest.fixture(scope="session")
def target_wav() -> Path:
    d = DATA / "target"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "target.wav"
    if not p.exists():
        _synth(p, kind="chirp", seed=99)
    return p


@pytest.fixture(scope="session")
def packs(corpus_dir) -> dict:
    """unit -> (pack_path, header, meta); unit=2 exercises overlapping-window dedup."""
    out = {}
    for unit in (1, 2):
        p = DATA / f"test_u{unit}.elkgrain"
        hdr, meta = build_corpus.build_pack(corpus_dir, p, unit=unit, stride=1,
                                            silence_db=-50.0, margin_ms=10.0)
        out[unit] = (p, hdr, meta)
    return out


def encode_cached(path: Path, granular) -> np.ndarray:
    """Cache-aware single-file encode using the builder's own cache scheme."""
    from importlib.metadata import version as pkg_version
    m2l = pkg_version("music2latent")
    ck = build_corpus.cache_key(path, granular.CODEC_SR, True, m2l)
    cp = build_corpus.DEFAULT_CACHE / f"{path.stem[:24].replace(' ', '_')}__{ck}.npz"
    if cp.exists():
        return np.load(cp)["lat"]
    y = granular.load_mono_44k(str(path))
    try:
        codec = granular.Codec(device="auto")
        lat = codec.encode(y)
    except Exception:
        codec = granular.Codec(device="cpu")
        lat = codec.encode(y)
    lat = np.asarray(lat, dtype=np.float32)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cp, lat=lat)
    return lat


@pytest.fixture(scope="session")
def target_lat(target_wav, granular) -> np.ndarray:
    return encode_cached(target_wav, granular)


def rebuild_codebook(granular, corpus_dir: Path, unit: int, stride: int = 1,
                     silence_db: float | None = -50.0):
    """Independently rebuild the codebook with the ORIGINAL build_codebook over the exact
    latents the builder used (from its cache) — the ground truth the pack must match."""
    files = build_corpus.find_audio(corpus_dir)
    y44s = [granular.load_mono_44k(str(p)) for p in files]
    lats = [encode_cached(p, granular) for p in files]
    return granular.build_codebook(lats, unit, stride, name="test",
                                   sources=[p.name for p in files],
                                   seconds=sum(len(y) / granular.CODEC_SR for y in y44s),
                                   audio_list=y44s, silence_db=silence_db)
