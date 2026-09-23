"""A generated demo signal for the notebooks. Added for the public repository.

The paper's corpora are the author's recordings and MUSDB18 previews, which are not redistributed. The
notebooks run on three short synthetic "recordings" instead, one per character of the paper's corpora
(percussive, tonal, textural), plus held-out queries of each character drawn with other seeds. Everything is
deterministic (fixed seeds) and written to $LBD_WORK/demo (default <repo>/work/demo, git-ignored), never
committed. The numbers the notebooks compute on it illustrate each step; the paper's numbers come from
results/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("LBD_WORK", REPO / "work"))
DEMO = WORK / "demo"
CORPUS_DIR = DEMO / "corpus"
QUERY_DIR = DEMO / "queries"
PACK = DEMO / "demo.elkgrain"


def scrub(text: str) -> str:
    """Absolute paths -> $LBD_WORK/..., <repo>/..., or the bare file name (notebook outputs are published)."""
    import re
    text = str(text).replace(str(WORK.resolve()), "$LBD_WORK").replace(str(WORK), "$LBD_WORK").replace(str(REPO), "<repo>")
    return re.sub(r"(?<![\w$<>.])(?:/[^\s/]+)+/([^\s/]+)", r"\1", text)


def log(msg="", *a, **k):
    print(scrub(msg))


class scrubbed:
    """Context manager: capture what the code inside prints and re-print it through scrub()."""
    def __enter__(self):
        import contextlib, io
        self._buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self._buf)
        self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        out = self._buf.getvalue()
        if out:
            print(scrub(out), end="")
        return False


def rel(path) -> str:
    return str(Path(path).resolve()).replace(str(WORK.resolve()), "$LBD_WORK").replace(str(REPO), "<repo>")


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def percussive(seconds: float, seed: int, bpm: float = 112.0) -> np.ndarray:
    """Kick (pitch-swept sine), snare (noise + tone), hats (high-passed noise) on a varied 16-step pattern."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    step = int(SR * 60.0 / bpm / 4)
    hp = butter(4, 6000, "highpass", fs=SR, output="sos")
    for i, s in enumerate(range(0, n - SR // 2, step)):
        k = i % 16
        if k in (0, 8) or (k == 10 and rng.random() < 0.5):          # kick
            m = int(0.35 * SR); t = np.arange(m) / SR
            f = 45 + 110 * np.exp(-t / 0.04)
            y[s:s + m] += 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(m, 0.002, 0.12)
        if k in (4, 12):                                              # snare
            m = int(0.25 * SR)
            y[s:s + m] += (0.5 * rng.standard_normal(m) + 0.3 * np.sin(2 * np.pi * 190 * np.arange(m) / SR)) * _env(m, 0.001, 0.07)
        if rng.random() < 0.8:                                        # hats
            m = int(0.06 * SR)
            y[s:s + m] += 0.25 * sosfilt(hp, rng.standard_normal(m)) * _env(m, 0.001, 0.02)
    return y


def tonal(seconds: float, seed: int) -> np.ndarray:
    """Additive notes (a few harmonics, vibrato, attack/decay) over a slow chord progression."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    roots = [57, 60, 53, 55, 62, 50]                                  # MIDI roots
    note_len = int(0.45 * SR)
    for s in range(0, n - note_len, note_len // 2):
        root = roots[(s // (4 * note_len)) % len(roots)]
        midi = root + int(rng.choice([0, 3, 4, 7, 10, 12]))
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        m = int(note_len * rng.uniform(0.8, 1.6)); m = min(m, n - s)
        t = np.arange(m) / SR
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
        tone = sum((0.6 / h) * np.sin(2 * np.pi * f0 * h * np.cumsum(vib) / SR) for h in range(1, 6))
        y[s:s + m] += 0.35 * tone * _env(m, 0.02, 0.35)
    return y


def textural(seconds: float, seed: int) -> np.ndarray:
    """Band-passed noise with a slowly sweeping centre, over a detuned drone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    blk = SR // 4
    centres = 300 * 2 ** (3 * (0.5 + 0.5 * np.sin(2 * np.pi * t[::blk] / rng.uniform(5, 9))))
    for i, s in enumerate(range(0, n, blk)):
        c = centres[i]
        sos = butter(2, [c / 1.4, min(c * 1.4, SR / 2 - 100)], "bandpass", fs=SR, output="sos")
        seg = sosfilt(sos, rng.standard_normal(min(blk + 512, n - s)))
        w = np.hanning(len(seg))
        out[s:s + len(seg)] += 0.6 * seg * w
    f = rng.uniform(55, 80)
    drone = sum(np.sin(2 * np.pi * f * k * (1 + 0.003 * j) * t) / k for k in (1, 2, 3) for j in (-1, 1))
    return out + 0.08 * drone * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))


MAKERS = {"perc": percussive, "tonal": tonal, "text": textural}
CORPUS = {"perc.wav": ("perc", 20.0, 1), "tonal.wav": ("tonal", 20.0, 2), "text.wav": ("text", 20.0, 3)}
QUERIES = {"qp1": ("perc", 8.0, 11), "qt1": ("tonal", 8.0, 12), "qx1": ("text", 8.0, 13)}


def _write(path: Path, y: np.ndarray):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32), SR)


def write_corpus() -> Path:
    for name, (kind, sec, seed) in CORPUS.items():
        if not (CORPUS_DIR / name).exists():
            _write(CORPUS_DIR / name, MAKERS[kind](sec, seed))
    return CORPUS_DIR


def write_queries() -> dict[str, tuple[Path, str]]:
    out = {}
    for qid, (kind, sec, seed) in QUERIES.items():
        p = QUERY_DIR / f"{qid}.wav"
        if not p.exists():
            _write(p, MAKERS[kind](sec, seed))
        out[qid] = (p, kind)
    return out


DEVICE = "cpu"   # the notebooks encode on the CPU so every reader gets the same latents (the paper used MPS)


def ensure_pack(log=lambda *a, **k: None) -> Path:
    """Build the demo pack once (music2latent encodes are cached in $LBD_WORK/cache/latents)."""
    from corpus_builder import build_corpus
    write_corpus()
    if not PACK.exists():
        build_corpus.build_pack(CORPUS_DIR, PACK, unit=1, stride=1, silence_db=-50.0, margin_ms=10.0,
                                device=DEVICE, log=log)
    return PACK


_codec = None


def codec():
    global _codec
    if _codec is None:
        from latent.granular import Codec
        _codec = Codec(device=DEVICE)
    return _codec


def encode_queries() -> dict[str, dict]:
    """Held-out demo queries: teacher latents [64, T] and the per-frame features (mel80, feat89, mfcc13)."""
    from distillation import features as F
    from latent import granular as g
    out = {}
    for qid, (path, kind) in write_queries().items():
        y44 = g.load_mono_44k(str(path))
        lat = np.asarray(codec().encode(y44), dtype=np.float32)
        out[qid] = {"character": kind, "lat": lat, **F.frame_features(y44, lat.shape[1])}
    return out
