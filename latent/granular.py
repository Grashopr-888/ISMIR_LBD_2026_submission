# Provenance: Grashopr-888/mutek-hackathon@2db1ce0:src/latent/granular.py (verbatim)
"""
Latent granular resynthesis core (Tokui & Baker 2025) — shared by the offline bench and the live renderer.

    corpus audio ─► codec.encode ─► latents [dim, T] (~10.7 Hz for music2latent) ─► grains [N, dim*unit]
    target audio ─► codec.encode ─► grains ─► cosine (mean-centred) + softmax(τ) ─► substituted latents ─► codec.decode

Design notes from listening/benchmarks (docs/experiment_log.md):
  * mean-centre before cosine (raw cosine is dominated by a shared offset);
  * gain-match substituted grains to the target grain's latent norm (dynamics survive);
  * per-file peak normalisation on load (loudness disparity between corpus files ⇒ norm disparity);
  * drop near-silent grains from the codebook (windchime "presence" lesson) — optional;
  * report hubness (share of picks taken by the 5 most-used grains) — windchime "hubness" lesson.
Torch/music2latent are imported lazily; numpy does the matching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

CODEC_SR = 44100      # music2latent native
LATENT_DIM = 64


# --------------------------------------------------------------------------- codec wrapper
class Codec:
    """Thin wrapper so the rest of the code never touches torch directly.
    All GPU calls are serialised with a lock: Metal (MPS) aborts the process if two threads submit
    command buffers concurrently ("commit an already committed command buffer", seen 2026-08-17)."""
    def __init__(self, device: str = "auto"):
        import threading
        import torch
        from music2latent import EncoderDecoder
        self._lock = threading.Lock()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self.encdec = EncoderDecoder(device=torch.device(device))
        self.sr = CODEC_SR

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """mono float32 @44.1k → latents [dim, T]"""
        with self._lock:
            lat = self.encdec.encode(np.asarray(audio, dtype=np.float32), max_waveform_length=CODEC_SR * 10)
            if hasattr(lat, "cpu"):
                lat = lat.cpu().numpy()
            if self.device == "mps":
                self._torch.mps.synchronize()
        return np.asarray(lat)[0]

    def decode(self, lat: np.ndarray, denoising_steps: int = 1) -> np.ndarray:
        with self._lock:
            y = self.encdec.decode(np.asarray(lat, dtype=np.float32)[None, ...], denoising_steps=denoising_steps,
                                   max_waveform_length=CODEC_SR * 10)
            if hasattr(y, "cpu"):
                y = y.cpu().numpy()
            if self.device == "mps":
                self._torch.mps.synchronize()
        return np.asarray(y).reshape(-1)


# --------------------------------------------------------------------------- codebook
def grains_from_latents(lat: np.ndarray, unit: int, stride: int = 1) -> np.ndarray:
    dim, T = lat.shape
    idx = np.arange(0, T - unit + 1, stride)
    if len(idx) == 0:
        return np.zeros((0, dim * unit), dtype=np.float32)
    return np.stack([lat[:, i:i + unit].reshape(-1) for i in idx], axis=0)


@dataclass
class Codebook:
    name: str
    unit: int
    raw: np.ndarray                 # [N, dim*unit] raw grains (what gets written into the output)
    book: np.ndarray                # [N, dim*unit] mean-centred + L2-normalised (what gets matched)
    centre: np.ndarray              # [dim*unit]
    sources: list[str] = field(default_factory=list)
    seconds: float = 0.0
    build_s: float = 0.0
    dropped_silent: int = 0

    @property
    def n(self) -> int:
        return int(self.raw.shape[0])


def build_codebook(latents_list: Sequence[np.ndarray], unit: int, stride: int = 1, name: str = "corpus",
                   sources: Sequence[str] = (), seconds: float = 0.0,
                   audio_list: Optional[Sequence[np.ndarray]] = None, silence_db: Optional[float] = -50.0) -> Codebook:
    """Concatenate grains from several files (any lengths). Optionally drop grains whose audio is near-silent."""
    t0 = time.perf_counter()
    raws, dropped = [], 0
    for k, lat in enumerate(latents_list):
        g = grains_from_latents(lat, unit, stride)
        if silence_db is not None and audio_list is not None and len(g):
            a = np.asarray(audio_list[k], dtype=np.float32)
            hop = len(a) / lat.shape[1]                                   # samples per latent (~4106)
            T = lat.shape[1]
            rms = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(T)])
            db = 20 * np.log10(rms + 1e-9)
            keep = np.array([np.max(db[i:i + unit]) > silence_db for i in range(0, T - unit + 1, stride)])
            dropped += int((~keep).sum())
            g = g[keep]
        raws.append(g)
    raw = np.concatenate(raws, axis=0) if raws else np.zeros((0, LATENT_DIM * unit), dtype=np.float32)
    centre = raw.mean(0) if len(raw) else np.zeros(LATENT_DIM * unit, dtype=np.float32)
    book = raw - centre
    book = book / (np.linalg.norm(book, axis=1, keepdims=True) + 1e-8)
    return Codebook(name=name, unit=unit, raw=raw.astype(np.float32), book=book.astype(np.float32), centre=centre.astype(np.float32),
                    sources=list(sources), seconds=seconds, build_s=time.perf_counter() - t0, dropped_silent=dropped)


# --------------------------------------------------------------------------- matching
@dataclass
class MatchInfo:
    grains: int = 0
    passthrough: int = 0
    coverage: Optional[float] = None       # mean best cosine per target grain (corpus↔input compatibility)
    picked: Optional[float] = None         # mean cosine of the grain actually chosen
    hubness: Optional[float] = None        # share of picks taken by the 5 most-used source grains (1.0 = one grain for everything)
    unique_frac: Optional[float] = None    # distinct source grains used / picks
    picks: Optional[list] = None           # per target grain: source index (single world) or (world_i, index) (blend)


STICK_BONUS = 0.6      # similarity bonus, at stick = 1, for the grain that FOLLOWS the last pick in its source


def _shape_grain(g_new: np.ndarray, centre: np.ndarray, scale: float, tg: np.ndarray, gain_match: bool,
                 depth: float) -> np.ndarray:
    """The per-grain post-processing every control shares, in order:
    intensity  — scale the grain's deviation from its world's centre (direction, not level: >1 exaggerates
                 what makes it unlike the world's average, <1 pulls it toward the average);
    gain-match — restore the target grain's level (so intensity cannot be a loudness control);
    depth      — interpolate in latent space between the grain you played (0) and the world's grain (1)."""
    if scale != 1.0:
        g_new = centre.reshape(g_new.shape) + scale * (g_new - centre.reshape(g_new.shape))
    if gain_match:
        n_t = np.linalg.norm(tg); n_s = np.linalg.norm(g_new) + 1e-8
        g_new = g_new * (n_t / n_s)
    if depth < 1.0:
        g_new = (1.0 - depth) * tg + depth * g_new
    return g_new


def smooth_latents(lat: np.ndarray, amount: float) -> np.ndarray:
    """Soften the joins: a [¼ ½ ¼] pass over the latent frames, applied one to three times as amount goes 0→1
    (fractional amounts blend between passes). That kernel is a true low-pass — a heavier 3-tap one is not, it
    lets alternating patterns through with a sign flip, which is exactly the roughness this control removes.
    Edges are left as they are so block boundaries do not drift."""
    if amount <= 0 or lat.shape[1] < 3:
        return lat
    x = min(1.0, float(amount)) * 3.0
    k = int(np.ceil(x - 1e-9)); f = x - (k - 1)
    def one(y):
        o = y.copy(); o[:, 1:-1] = 0.5 * y[:, 1:-1] + 0.25 * (y[:, :-2] + y[:, 2:]); return o
    prev = cur = lat
    for _ in range(k):
        prev, cur = cur, one(cur)
    return ((1.0 - f) * prev + f * cur).astype(lat.dtype)


def substitute(target_lat: np.ndarray, cb: Codebook, temperature: float = 0.05, gain_match: bool = True,
               threshold: Optional[float] = None, rng: Optional[np.random.Generator] = None,
               anti_hub: float = 0.0, stick: float = 0.0, depth: float = 1.0,
               scale: float = 1.0) -> tuple[np.ndarray, MatchInfo]:
    """Replace each target grain (non-overlapping, stride = unit) by a corpus grain. Returns new latents [dim, T].
    anti_hub > 0 penalises grains already picked in this block (log-count penalty on similarity).
    stick    > 0 favours the grain that follows the previous pick in its source recording (continuity).
    depth    < 1 interpolates toward the target grain (0 = your own sound re-decoded, 1 = fully the world).
    scale   != 1 exaggerates (>1) or flattens (<1) each grain's deviation from the world's centre."""
    rng = rng or np.random.default_rng(0)
    dim, T = target_lat.shape
    unit = cb.unit
    out = target_lat.copy()
    if cb.n == 0 or T < unit:
        return out, MatchInfo(grains=0)
    info = MatchInfo()
    best, picked, picks = [], [], []
    counts = np.zeros(cb.n, dtype=np.float32)
    prev = -1
    for i in range(0, T - unit + 1, unit):
        g = target_lat[:, i:i + unit].reshape(-1) - cb.centre
        g = g / (np.linalg.norm(g) + 1e-8)
        sims = cb.book @ g
        if anti_hub > 0:
            sims = sims - anti_hub * np.log1p(counts)
        if stick > 0 and 0 <= prev + 1 < cb.n:
            sims = sims.copy(); sims[prev + 1] += stick * STICK_BONUS
        dist = 1.0 - sims
        best.append(float(sims.max()))
        if threshold is not None and dist.min() > threshold:
            info.passthrough += 1
            continue
        if temperature <= 1e-6:
            j = int(np.argmax(sims))
        else:
            logits = -dist / temperature
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            j = int(rng.choice(cb.n, p=p))
        counts[j] += 1; prev = j
        picks.append(j); picked.append(float(sims[j]))
        g_new = cb.raw[j].reshape(dim, unit)
        out[:, i:i + unit] = _shape_grain(g_new, cb.centre, scale, target_lat[:, i:i + unit], gain_match, depth)
    info.grains = len(best)
    info.coverage = float(np.mean(best)) if best else None
    info.picked = float(np.mean(picked)) if picked else None
    if picks:
        c = np.bincount(np.array(picks), minlength=cb.n)
        info.hubness = float(np.sort(c)[::-1][:5].sum() / len(picks))
        info.unique_frac = float((c > 0).sum() / len(picks))
    info.picks = picks
    return out, info


# --------------------------------------------------------------------------- audio helpers
def load_mono_44k(path: str, offset: float = 0.0, seconds: Optional[float] = None, normalize: bool = True) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=CODEC_SR, mono=True, offset=offset, duration=seconds)
    y = y.astype(np.float32)
    if normalize and len(y):
        peak = float(np.max(np.abs(y)))
        if peak > 1e-6:
            y = y / peak * 0.9                     # per-file peak normalisation (loudness disparity → norm disparity)
    return y


def augment(y: np.ndarray, vol: Sequence[float] = (), pitch_st: Sequence[float] = ()) -> np.ndarray:
    parts = [y]
    for g in vol:
        parts.append(y * float(g))
    if pitch_st:
        import librosa
        for st in pitch_st:
            parts.append(librosa.effects.pitch_shift(y, sr=CODEC_SR, n_steps=float(st)).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(np.asarray(x, dtype=np.float32), sr_to // g, sr_from // g).astype(np.float32)


def substitute_blend(target_lat: np.ndarray, worlds: Sequence[tuple["Codebook", float]], mode: str = "mix",
                     temperature: float = 0.05, gain_match: bool = True,
                     rng: Optional[np.random.Generator] = None, anti_hub: float = 0.0, stick: float = 0.0,
                     depth: float = 1.0, scale: float = 1.0) -> tuple[np.ndarray, MatchInfo]:
    """Blend several codebooks (all with the same unit).
    mode 'mix'  : for each target grain draw ONE world with probability ∝ weight, then match inside it (mosaic mix);
    mode 'morph': match in EVERY world, then take the weight-averaged latent of the best grains (latent-space morph,
                  decoded as an in-between timbre).
    anti_hub / stick / depth / scale behave exactly as in `substitute`; hubness is measured over (world, grain)."""
    rng = rng or np.random.default_rng(0)
    worlds = [(cb, float(w)) for cb, w in worlds if cb is not None and cb.n > 0 and w > 0]
    if not worlds:
        return target_lat.copy(), MatchInfo(grains=0)
    if len(worlds) == 1:
        return substitute(target_lat, worlds[0][0], temperature, gain_match, rng=rng,
                          anti_hub=anti_hub, stick=stick, depth=depth, scale=scale)
    unit = worlds[0][0].unit
    dim, T = target_lat.shape
    out = target_lat.copy()
    wsum = sum(w for _, w in worlds); probs = np.array([w / wsum for _, w in worlds])
    best, picked, picks = [], [], []
    counts = [np.zeros(cb.n, dtype=np.float32) for cb, _ in worlds]
    prev = [-1] * len(worlds)
    for i in range(0, T - unit + 1, unit):
        tg = target_lat[:, i:i + unit]
        tgf = tg.reshape(-1)
        cands = []
        for wi, (cb, w) in enumerate(worlds):
            g = (tgf - cb.centre); g = g / (np.linalg.norm(g) + 1e-8)
            sims = cb.book @ g
            if anti_hub > 0:
                sims = sims - anti_hub * np.log1p(counts[wi])
            if stick > 0 and 0 <= prev[wi] + 1 < cb.n:
                sims = sims.copy(); sims[prev[wi] + 1] += stick * STICK_BONUS
            if temperature <= 1e-6:
                j = int(np.argmax(sims))
            else:
                logits = -(1.0 - sims) / temperature; logits -= logits.max(); pr = np.exp(logits); pr /= pr.sum()
                j = int(rng.choice(cb.n, p=pr))
            cands.append((float(sims.max()), float(sims[j]), cb.raw[j].reshape(dim, unit), w, j, cb))
        best.append(max(c[0] for c in cands))
        if mode == "morph":
            g_new = sum(c[2] * c[3] for c in cands) / wsum
            centre = sum(c[5].centre * c[3] for c in cands) / wsum
            picked.append(sum(c[1] * c[3] for c in cands) / wsum)
            for wi, c in enumerate(cands):
                counts[wi][c[4]] += 1; prev[wi] = c[4]; picks.append((wi, c[4]))
        else:
            k = int(rng.choice(len(cands), p=probs))
            g_new = cands[k][2]; centre = cands[k][5].centre; picked.append(cands[k][1])
            counts[k][cands[k][4]] += 1; prev[k] = cands[k][4]; picks.append((k, cands[k][4]))
        out[:, i:i + unit] = _shape_grain(g_new, centre, scale, tg, gain_match, depth)
    info = MatchInfo(grains=len(best), coverage=float(np.mean(best)) if best else None,
                     picked=float(np.mean(picked)) if picked else None)
    if picks:                                                    # hub over (world, grain) pairs
        from collections import Counter
        c = np.array(sorted(Counter(picks).values(), reverse=True))
        info.hubness = float(c[:5].sum() / len(picks))
        info.unique_frac = float(len(c) / len(picks))
    info.picks = picks
    return out, info
