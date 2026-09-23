# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/features.py (changed: docstring)
"""Handcrafted per-frame features on the music2latent frame grid (T06 rungs A/B/F).

Frame grid parity: music2latent's effective hop for a file is len(y44)/T samples (~4106
@44.1k, ~93 ms).  Frame i's analysis window starts at int(i*hop) — the SAME expression the
instrument's silence gate uses (granular.build_codebook) — and spans N_FFT=4096 samples
(zero-padded past EOF).  Everything below is one rFFT + O(bins) math per frame, i.e. cheap
enough for the Stomp's A7 at 10.7 Hz, and stateless except spectral flux (one
previous-spectrum vector).

Feature vector (dim 89), float32:
    [ 0:80]  log-mel, 80 bands, 20 Hz..16 kHz  (ln(mel_power + 1e-10))
    [80]     log RMS of the frame window       (ln(rms + 1e-9))
    [81]     spectral centroid (kHz)
    [82]     spectral bandwidth (kHz)           sqrt 2nd central moment
    [83]     spectral skewness                  3rd standardized moment
    [84]     spectral kurtosis                  4th standardized moment
    [85]     log spectral flatness              ln(geo mean / arith mean) of power
    [86]     rolloff 85% (kHz)
    [87]     spectral flux                      L2 of successive prob-normalised spectra
    [88]     zero-crossing rate                 fraction of sign changes in the window

MFCC13 (floor condition F): DCT-II(ortho) of the 80 log-mels, coefficients 0..12.
The student (rung B) uses the 80 log-mels alone (budget shape mel80->256->256->64).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct

SR = 44100
N_FFT = 4096
N_MELS = 80
FMIN, FMAX = 20.0, 16000.0
N_MFCC = 13

_mel_fb = None
_hann = None


def _mel():
    global _mel_fb, _hann
    if _mel_fb is None:
        import librosa
        _mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                      fmin=FMIN, fmax=FMAX).astype(np.float32)
        _hann = np.hanning(N_FFT).astype(np.float32)
    return _mel_fb, _hann


def frame_windows(y44: np.ndarray, T: int) -> np.ndarray:
    """[T, N_FFT] float32 windows, start-aligned to the codec frame grid, zero-padded."""
    y = np.asarray(y44, dtype=np.float32)
    hop = len(y) / T
    starts = (np.arange(T) * hop).astype(np.int64)          # == int(i*hop), hop > 0
    ypad = np.concatenate([y, np.zeros(N_FFT, dtype=np.float32)])
    idx = starts[:, None] + np.arange(N_FFT)[None, :]
    return ypad[idx]


def frame_features(y44: np.ndarray, T: int) -> dict[str, np.ndarray]:
    """All per-frame tables for one file: feat89 [T,89], mel80 [T,80], mfcc13 [T,13]."""
    fb, hann = _mel()
    W = frame_windows(y44, T)
    spec = np.fft.rfft(W * hann[None, :], axis=1)
    P = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)          # [T, 2049] power
    freqs_khz = (np.fft.rfftfreq(N_FFT, 1.0 / SR) / 1000.0).astype(np.float32)

    mel_power = P @ fb.T                                              # [T, 80]
    logmel = np.log(mel_power + 1e-10).astype(np.float32)

    rms = np.sqrt(np.mean(W ** 2, axis=1) + 1e-12)
    logrms = np.log(rms + 1e-9)

    Psum = P.sum(axis=1) + 1e-12
    Pn = P / Psum[:, None]                                            # prob-normalised
    centroid = Pn @ freqs_khz
    d = freqs_khz[None, :] - centroid[:, None]
    var = np.einsum("tk,tk->t", Pn, d * d)
    bw = np.sqrt(var + 1e-12)
    skew = np.einsum("tk,tk->t", Pn, d ** 3) / (bw ** 3 + 1e-12)
    kurt = np.einsum("tk,tk->t", Pn, d ** 4) / (var ** 2 + 1e-12)
    logflat = np.mean(np.log(P + 1e-12), axis=1) - np.log(np.mean(P, axis=1) + 1e-12)
    cum = np.cumsum(P, axis=1)
    roll = freqs_khz[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flux = np.zeros(len(P), dtype=np.float32)
    if len(P) > 1:
        flux[1:] = np.linalg.norm(Pn[1:] - Pn[:-1], axis=1)
    sgn = np.signbit(W)
    zcr = np.mean(sgn[:, 1:] != sgn[:, :-1], axis=1)

    feat89 = np.column_stack([logmel, logrms, centroid, bw, skew, kurt,
                              logflat, roll, flux, zcr]).astype(np.float32)
    mfcc13 = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC].astype(np.float32)
    return {"feat89": feat89, "mel80": logmel, "mfcc13": mfcc13}


FEAT89_NAMES = ([f"mel{i}" for i in range(80)] +
                ["logrms", "centroid", "bandwidth", "skew", "kurt",
                 "logflat", "rolloff85", "flux", "zcr"])
