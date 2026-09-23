#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/ridgefit.py (changed: docstring)
"""Rung A: ridge linear map, handcrafted feat89 -> 64-d music2latent frame.

Fit on each pack's own (features -> raw latent) pairs ("per-pack", the deploy story: the
laptop fits the map while building the pack) AND on all packs pooled ("pooled", one map for
every world).  Lambda chosen by leave-one-FILE-out CV (frames within a file are strongly
correlated; a frame-level split would flatter the map).  Score = mean cosine of mean-centred
prediction vs truth (the quantity matching actually uses).

Outputs $LBD_WORK/ab/ridge.npz:  {name}_{mu,sd,W,b,lam}  for name in packs + 'pooled'.
On-device cost of applying a map: 89*64 MACs ~ 5.7 k/frame — noise against the Cortex-A7 budget.
"""
from __future__ import annotations

import json

import numpy as np

from distillation import ab_common as C

LAMBDAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


def centred_cos(pred: np.ndarray, true: np.ndarray, centre: np.ndarray) -> float:
    a = pred - centre
    b = true - centre
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(num / den))


def fit_ridge(X, Y, lam):
    """Standardized-X ridge with intercept.  Returns mu, sd, W [d,64], b [64]."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    Yc = Y - ym
    d = Xs.shape[1]
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d, dtype=np.float64), Xs.T @ Yc)
    return mu.astype(np.float32), sd.astype(np.float32), W.astype(np.float32), ym.astype(np.float32)


def apply_ridge(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["W"] + m["b"]


def cv_lambda(X, Y, groups, centre, label):
    scores = {}
    for lam in LAMBDAS:
        ss = []
        for g in np.unique(groups):
            tr = groups != g
            if tr.all() or not tr.any():
                continue
            mu, sd, W, b = fit_ridge(X[tr], Y[tr], lam)
            pred = ((X[~tr] - mu) / sd) @ W + b
            ss.append(centred_cos(pred, Y[~tr], centre))
        scores[lam] = float(np.mean(ss))
    best = max(scores, key=scores.get)
    print(f"  [{label}] LOFO centred-cos by lambda: "
          + "  ".join(f"{l:g}:{scores[l]:.4f}" for l in LAMBDAS) + f"  -> lam={best:g}")
    return best, scores[best]


def main():
    packs = json.loads((C.AB / "packs.json").read_text())
    out, report = {}, {}
    pooled_X, pooled_Y, pooled_G, gbase = [], [], [], 0
    for name, path in packs.items():
        cf = np.load(C.AB / "corpusfeat" / f"{name}.npz")
        X = cf["feat89"].astype(np.float64)
        Y = cf["raw"].astype(np.float64)
        G = cf["file_idx"].astype(int)
        with C.load_pack(path) as p:
            centre = np.asarray(p.centre, np.float64)
        lam, cv = cv_lambda(X, Y, G, centre, name)
        mu, sd, W, b = fit_ridge(X, Y, lam)
        out |= {f"{name}_mu": mu, f"{name}_sd": sd, f"{name}_W": W, f"{name}_b": b,
                f"{name}_lam": np.float64(lam)}
        report[name] = {"lam": lam, "cv_centred_cos": round(cv, 4), "n": len(X)}
        pooled_X.append(X); pooled_Y.append(Y); pooled_G.append(G + gbase)
        gbase += G.max() + 1
    X = np.concatenate(pooled_X); Y = np.concatenate(pooled_Y); G = np.concatenate(pooled_G)
    centre = Y.mean(0)
    lam, cv = cv_lambda(X, Y, G, centre, "pooled")
    mu, sd, W, b = fit_ridge(X, Y, lam)
    out |= {"pooled_mu": mu, "pooled_sd": sd, "pooled_W": W, "pooled_b": b,
            "pooled_lam": np.float64(lam)}
    report["pooled"] = {"lam": lam, "cv_centred_cos": round(cv, 4), "n": len(X)}
    np.savez(C.AB / "ridge.npz", **out)
    C.jdump(report, C.AB / "ridge_report.json")
    print("ridge maps written:", C.AB / "ridge.npz")


def load_maps():
    z = np.load(C.AB / "ridge.npz")
    names = sorted({k.rsplit("_", 1)[0] for k in z.files})
    return {n: {"mu": z[f"{n}_mu"], "sd": z[f"{n}_sd"], "W": z[f"{n}_W"], "b": z[f"{n}_b"]}
            for n in names}


if __name__ == "__main__":
    main()
