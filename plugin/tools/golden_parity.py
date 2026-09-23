#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/tools/golden_parity.py (changed: paths from the repository and the environment, package imports)
"""Golden validation: C++ engine stages vs the python references.

Run from the repository root after building plugin/ on the desktop (it needs the ElkgrainGolden CLI):

    GOLDEN_PACK=<any .elkgrain pack, e.g. work/demo/demo.elkgrain from notebook 01> \
        python plugin/tools/golden_parity.py

The paper's run used the 12,018-grain field pack (200 of 200 identical selections); packs are not
redistributed, so any pack works here. GOLDEN_BIN overrides the CLI path, GOLDEN_WEIGHTS / GOLDEN_RIDGE
the student and ridge JSON (default: the repository's weights/).

Stages (each PASS/FAIL, exits nonzero on any FAIL):
  1. matcher vs corpus_builder/match.py over the pack (GOLDEN_PACK):
     tau=0 argmax picks must be IDENTICAL, pick/best sims <= 1e-5, full sims
     rows <= 1e-5, per-bar counts + prev exact — across penalty configs
     (anti_hub / stick on and off). Sampling streams (tau>0) are validated at
     the sims level only: numpy Generator.choice cannot be reproduced in C++
     (match.py docstring; the C++ sampler is a documented mt19937 inverse-CDF).
  2. student MLP: C++ RTNeural forward vs numpy forward from the same JSON.
  3. ridge pooled map: C++ vs ridgefit.apply_ridge.
  4. mel filterbank: C++ construction vs librosa's (the training filterbank).
  5. mel/feat89 frontend: C++ analyzeWindow44 vs features.frame_features on
     identical 4096-sample windows (feed y44 of length T*4096 so the python
     frame grid is exactly our windows).
  6. resampler: C++ streaming 147/160 vs scipy.signal.resample_poly (delay-
     aligned; the C++ stream is causal, scipy compensates group delay).
"""
from __future__ import annotations
import json

import re
import subprocess
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]            # the repository
BUILD = ROOT / "plugin" / "build"
GOLDEN = Path(os.environ.get("GOLDEN_BIN", BUILD / "ElkgrainGolden_artefacts" / "Release" / "ElkgrainGolden"))
PACK = Path(os.environ.get("GOLDEN_PACK", ROOT / "work" / "demo" / "demo.elkgrain"))
WEIGHTS = Path(os.environ.get("GOLDEN_WEIGHTS", ROOT / "weights" / "student_mlp_mel80.json"))
RIDGE = Path(os.environ.get("GOLDEN_RIDGE", ROOT / "weights" / "ridge_pooled.json"))
# Scratch dir for the intermediates (regenerated every run). Override with GOLDEN_TMP.
TMP = Path(os.environ.get("GOLDEN_TMP", BUILD / "golden-tmp"))
TMP.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from corpus_builder.elkgrain import ElkGrainPack   # noqa: E402
from corpus_builder.match import Matcher           # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def run(args: list[str]) -> str:
    r = subprocess.run([str(GOLDEN)] + [str(a) for a in args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise RuntimeError(f"golden {args[0]} failed rc={r.returncode}")
    return r.stdout


# ---------------------------------------------------------------- 1. matcher
def stage_matcher():
    pack = ElkGrainPack(PACK)
    n, dim = pack.n, pack.grain_dim
    rng = np.random.default_rng(1234)
    T = 200
    rows = rng.integers(0, n, size=T)
    q = np.asarray(pack.raw)[rows].astype(np.float32)
    q += (rng.standard_normal((T, dim)).astype(np.float32) * 0.05
          * np.abs(q).mean(axis=1, keepdims=True).astype(np.float32))
    qpath = TMP / "queries.f32"
    q.tofile(qpath)

    matcher = Matcher(np.asarray(pack.emb), np.asarray(pack.centre), pack.unit)

    configs = [
        ("tau0_plain", 0.0, 0.0, 0.0),
        ("tau0_antihub", 0.0, 0.25, 0.0),
        ("tau0_stick", 0.0, 0.0, 0.85),
        ("tau0_both", 0.0, 0.15, 0.6),
    ]
    for name, tau, ah, st in configs:
        simsN = 3
        simsPath = TMP / f"sims_{name}.f32"
        out = run(["match", PACK, qpath, T, tau, ah, st, 0, simsN, simsPath])
        picks_c, sim_c, best_c = [], [], []
        counts_c, prev_c = {}, None
        for line in out.splitlines():
            m = re.match(r"slot (\d+) pick (-?\d+) sim (\S+) best (\S+)", line)
            if m:
                picks_c.append(int(m.group(2)))
                sim_c.append(float(m.group(3)))
                best_c.append(float(m.group(4)))
                continue
            m = re.match(r"prev (-?\d+)", line)
            if m:
                prev_c = int(m.group(1))
                continue
            m = re.match(r"count (\d+) (\d+)", line)
            if m:
                counts_c[int(m.group(1))] = int(m.group(2))

        res = matcher.select(q.T.copy(), temperature=tau, anti_hub=ah, stick=st,
                             rng=np.random.default_rng(0))
        picks_p = res.picks
        same_picks = picks_c == picks_p
        check(f"matcher/{name}/picks", same_picks,
              f"{sum(int(a == b) for a, b in zip(picks_c, picks_p))}/{T} identical")

        # python reference sims per slot (recompute with penalties, like match.py)
        book, centre = np.asarray(pack.emb), np.asarray(pack.centre)
        counts = np.zeros(n, dtype=np.float32)
        prev = -1
        max_sim_err = max_best_err = 0.0
        sims_rows_err = 0.0
        for t in range(T):
            g = q[t] - centre
            g = g / (np.linalg.norm(g) + 1e-8)
            sims = book @ g
            if ah > 0:
                sims = sims - ah * np.log1p(counts)
            if st > 0 and 0 <= prev + 1 < n:
                sims = sims.copy()
                sims[prev + 1] += st * 0.6
            j = picks_p[t]
            max_best_err = max(max_best_err, abs(float(sims.max()) - best_c[t]))
            max_sim_err = max(max_sim_err, abs(float(sims[j]) - sim_c[t]))
            if t < simsN:
                cxx = np.fromfile(simsPath, dtype=np.float32).reshape(simsN, n)
                sims_rows_err = max(sims_rows_err,
                                    float(np.abs(cxx[t] - sims).max()))
            counts[j] += 1
            prev = j
        check(f"matcher/{name}/sims", max_sim_err <= 1e-5 and max_best_err <= 1e-5,
              f"max pick-sim err {max_sim_err:.2e}, best err {max_best_err:.2e}")
        check(f"matcher/{name}/sims_rows", sims_rows_err <= 1e-5,
              f"first-{simsN}-slot full sims max err {sims_rows_err:.2e}")

        bc = np.bincount(np.array(picks_p), minlength=n)
        counts_p = {int(i): int(c) for i, c in enumerate(bc) if c > 0}
        check(f"matcher/{name}/counters", counts_c == counts_p and prev_c == picks_p[-1],
              f"{len(counts_c)} nonzero counts, prev {prev_c}")
    pack.close()


# ------------------------------------------------------------ 2. student MLP
def stage_mlp():
    import json
    doc = json.loads(WEIGHTS.read_text())
    mu = np.array(doc["input"]["standardize"]["mean"], dtype=np.float32)
    sd = np.array(doc["input"]["standardize"]["std"], dtype=np.float32)
    Ws = [np.array(l["weights"][0], dtype=np.float32) for l in doc["layers"]]
    bs = [np.array(l["weights"][1], dtype=np.float32) for l in doc["layers"]]

    rng = np.random.default_rng(7)
    T = 64
    x = (mu[None, :] + sd[None, :] * rng.standard_normal((T, 80))).astype(np.float32)
    xpath = TMP / "mlp_in.f32"
    x.tofile(xpath)
    opath = TMP / "mlp_out.f32"
    run(["mlp", WEIGHTS, xpath, T, opath])
    ycxx = np.fromfile(opath, dtype=np.float32).reshape(T, 64)

    h = (x - mu) / sd
    h = np.maximum(h @ Ws[0] + bs[0], 0.0)
    h = np.maximum(h @ Ws[1] + bs[1], 0.0)
    yref = h @ Ws[2] + bs[2]
    err = float(np.abs(ycxx - yref).max())
    rel = err / (float(np.abs(yref).max()) + 1e-12)
    check("student_mlp", err <= 5e-4, f"max abs err {err:.2e} (rel {rel:.2e})")


# ------------------------------------------------------------------ 3. ridge
def stage_ridge():
    from distillation import ridgefit as R
    # Reference weights: a ridgefit.py ridge.npz if GOLDEN_RIDGE_NPZ names one, else the
    # plugin's own ridge_pooled.json — the same pooled mu/sd/W/b, so the check stays meaningful.
    npz = Path(os.environ.get("GOLDEN_RIDGE_NPZ", "/nonexistent"))
    if npz.exists():
        z = np.load(npz)
        m = {"mu": z["pooled_mu"], "sd": z["pooled_sd"], "W": z["pooled_W"], "b": z["pooled_b"]}
    else:
        j = json.loads(Path(RIDGE).read_text())
        g = lambda *names: next(np.asarray(j[n], dtype=np.float64) for n in names if n in j)
        m = {"mu": g("pooled_mu", "mu"), "sd": g("pooled_sd", "sd"), "W": g("pooled_W", "W"), "b": g("pooled_b", "b")}
    rng = np.random.default_rng(11)
    T = 64
    x = (m["mu"][None, :] + m["sd"][None, :] * rng.standard_normal((T, 89))).astype(np.float32)
    xpath = TMP / "ridge_in.f32"
    x.tofile(xpath)
    opath = TMP / "ridge_out.f32"
    run(["ridge", RIDGE, xpath, T, opath])
    ycxx = np.fromfile(opath, dtype=np.float32).reshape(T, 64)
    yref = R.apply_ridge(m, x)
    err = float(np.abs(ycxx - yref).max())
    check("ridge_pooled", err <= 5e-4, f"max abs err {err:.2e}")


# ----------------------------------------------------- 4. mel filterbank
def stage_fb():
    from distillation import features
    fb_ref, _ = features._mel()
    opath = TMP / "fb.f32"
    run(["fb", opath])
    fb_cxx = np.fromfile(opath, dtype=np.float32).reshape(80, 2049)
    err = float(np.abs(fb_cxx - fb_ref).max())
    check("mel_filterbank", err <= 1e-6, f"max abs err {err:.2e}")


# ----------------------------------------------------- 5. mel/feat frontend
def stage_frontend():
    from distillation import features
    rng = np.random.default_rng(3)
    T = 24
    sr = 44100
    n = T * 4096
    t = np.arange(n) / sr
    y = (0.4 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * (900 + 400 * np.sin(2 * np.pi * 0.5 * t)) * t)
         + 0.1 * rng.standard_normal(n)).astype(np.float32)
    y *= np.linspace(1.0, 0.3, n).astype(np.float32)

    ref = features.frame_features(y, T)["feat89"]      # hop == 4096 exactly
    W = y.reshape(T, 4096)
    wpath = TMP / "wins.f32"
    W.astype(np.float32).tofile(wpath)
    opath = TMP / "feat_out.f32"
    run(["mel", wpath, T, opath])
    cxx = np.fromfile(opath, dtype=np.float32).reshape(T, 89)

    mel_err = float(np.abs(cxx[:, :80] - ref[:, :80]).max())
    ex_err = float(np.abs(cxx[:, 80:] - ref[:, 80:]).max())
    ok = mel_err <= 2e-3 and ex_err <= 2e-3
    check("frontend_feat89", ok, f"logmel max err {mel_err:.2e}, extras max err {ex_err:.2e}")


# --------------------------------------------------------------- 6. resampler
def stage_resampler():
    from scipy.signal import resample_poly
    rng = np.random.default_rng(5)
    n = 48000
    t = np.arange(n) / 48000.0
    x = (0.5 * np.sin(2 * np.pi * 440 * t * (1 + 0.5 * t))
         + 0.1 * rng.standard_normal(n)).astype(np.float32)
    xpath = TMP / "rs_in.f32"
    x.tofile(xpath)
    opath = TMP / "rs_out.f32"
    run(["resamp", xpath, opath])
    ycxx = np.fromfile(opath, dtype=np.float32)
    yref = resample_poly(x.astype(np.float64), 147, 160)
    # C++ stream is causal: delayed by half_len/down = 1600/160 = 10 outputs
    d = 10
    m = min(len(ycxx) - d, len(yref)) - 20
    err = float(np.abs(ycxx[d:d + m] - yref[:m]).max())
    check("resampler_147_160", err <= 1e-4,
          f"max abs err {err:.2e} over {m} samples (delay-aligned {d})")


def main():
    if not GOLDEN.exists():
        print(f"build the golden tool first: {GOLDEN}")
        sys.exit(2)
    stage_fb()
    stage_frontend()
    stage_resampler()
    stage_mlp()
    stage_ridge()
    stage_matcher()
    print()
    if FAILURES:
        print(f"GOLDEN: {len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("GOLDEN: ALL STAGES PASS")


if __name__ == "__main__":
    main()
