# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/match.py (changed: path in docstring)
"""Grain selection over an .elkgrain pack — the instrument's matching math, standalone.

This mirrors `substitute()` in latent/granular.py (the desktop instrument)
(Tokui & Baker latent granular) EXACTLY, restricted to *selection*: which corpus grain plays
for each target grain. The audio-grain engine on the Stomp plays stored audio, so the latent
output-shaping half of substitute() (_shape_grain: intensity/gain-match/depth) is not needed
here — selection indices are unaffected by it.

Parity contract (enforced by tests/test_parity.py): given the same book/centre arrays, the
same target latents, and the same numpy Generator seed and parameters, `select()` returns
selection indices IDENTICAL to `substitute(...)[1].picks`. Every expression below is kept in
the original's order and dtype so the float32 arithmetic — and hence the softmax probabilities
fed to rng.choice — is bit-identical. Do not "clean up" the math.

This file is also the executable spec for the C++ matcher (T09). C++ cannot reproduce
numpy's rng.choice stream, so on-device sampling will differ by design; T09 validates the
similarity/penalty pipeline (everything up to `p`) against this code, not the draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

STICK_BONUS = 0.6   # == granular.STICK_BONUS: similarity bonus at stick=1 for the grain that
                    # FOLLOWS the previous pick in its source recording


@dataclass
class SelectionResult:
    picks: list                                  # source grain index per NON-passthrough target grain
    grains: int = 0                              # target grains visited
    passthrough: int = 0                         # grains skipped by the threshold gate
    coverage: Optional[float] = None             # mean best cosine per target grain
    picked: Optional[float] = None               # mean cosine of the chosen grains
    hubness: Optional[float] = None              # share of picks taken by the 5 busiest grains
    unique_frac: Optional[float] = None          # distinct grains used / picks
    target_frames: list = field(default_factory=list)  # target frame index i for each pick


class Matcher:
    """Selection over one codebook (one 'world'). Arrays may be mmap views from ElkGrainPack."""

    def __init__(self, book: np.ndarray, centre: np.ndarray, unit: int):
        self.book = book          # [N, dim*unit] mean-centred + L2-normalised, float32
        self.centre = centre      # [dim*unit] float32
        self.unit = int(unit)
        self.n = int(book.shape[0])

    @classmethod
    def from_pack(cls, pack) -> "Matcher":
        return cls(pack.emb, pack.centre, pack.unit)

    def select(self, target_lat: np.ndarray, temperature: float = 0.05,
               threshold: Optional[float] = None, rng: Optional[np.random.Generator] = None,
               anti_hub: float = 0.0, stick: float = 0.0) -> SelectionResult:
        """Pick one corpus grain per non-overlapping target grain (stride = unit).

        target_lat: float32 [dim, T] latents of the live/target signal.
        Mirrors granular.substitute() selection: mean-centred cosine, softmax(tau) sampling,
        anti_hub log-count penalty, stick continuity bonus, optional distance threshold gate.
        """
        rng = rng or np.random.default_rng(0)
        target_lat = np.asarray(target_lat)
        dim, T = target_lat.shape
        unit = self.unit
        res = SelectionResult(picks=[])
        if self.n == 0 or T < unit:
            return res
        best, picked, picks = [], [], []
        counts = np.zeros(self.n, dtype=np.float32)
        prev = -1
        for i in range(0, T - unit + 1, unit):
            g = target_lat[:, i:i + unit].reshape(-1) - self.centre
            g = g / (np.linalg.norm(g) + 1e-8)
            sims = self.book @ g
            if anti_hub > 0:
                sims = sims - anti_hub * np.log1p(counts)
            if stick > 0 and 0 <= prev + 1 < self.n:
                sims = sims.copy()
                sims[prev + 1] += stick * STICK_BONUS
            dist = 1.0 - sims
            best.append(float(sims.max()))
            if threshold is not None and dist.min() > threshold:
                res.passthrough += 1
                continue
            if temperature <= 1e-6:
                j = int(np.argmax(sims))
            else:
                logits = -dist / temperature
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
                j = int(rng.choice(self.n, p=p))
            counts[j] += 1
            prev = j
            picks.append(j)
            picked.append(float(sims[j]))
            res.target_frames.append(i)
        res.grains = len(best)
        res.coverage = float(np.mean(best)) if best else None
        res.picked = float(np.mean(picked)) if picked else None
        if picks:
            c = np.bincount(np.array(picks), minlength=self.n)
            res.hubness = float(np.sort(c)[::-1][:5].sum() / len(picks))
            res.unique_frac = float((c > 0).sum() / len(picks))
        res.picks = picks
        return res
