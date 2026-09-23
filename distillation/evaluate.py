#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/evaluate.py (changed: package imports, usage lines)
"""T06 metrics: per (pack x query x condition) match-quality vs the music2latent oracle.

Conditions (query-side embedding only; the corpus side is always the pack's true tables):
  O      music2latent embedding of the query (the instrument today)  — reference
  O2     the same, matched against O with a different sampling seed  — seed-noise floor
  A      feat89 -> ridge fit on the SAME pack -> 64-d
  Apool  feat89 -> ridge fit on all three character packs pooled -> 64-d
  B      student MLP mel80->256->256->64 -> 64-d
  Bgru   student GRU-128 (only if trained)
  F      raw MFCC-13 cosine, no mapping (corpus side = MFCC of each grain's frame,
         mean-centred + L2 in MFCC space — same Matcher math, different space)

Ordering metrics (deterministic, from full sims matrices): top-1 agreement, top-5 overlap,
Spearman rank corr per frame.  Sampling metrics: Matcher.select picks with SAME seed at
tau in {0, .03, .05, .15} -> pick agreement with O at the same tau, plus hubness/coverage/
unique_frac per condition.  Embedding error (A/Apool/B): 1 - centred-cos vs oracle frame,
bucketed by onset (flux top-decile) and noisiness (logflat median split).

Writes $LBD_WORK/ab/out/results.json + agreement.csv; prints compact pivots.
Run: python -m distillation.evaluate
"""
from __future__ import annotations

import json

import numpy as np

from corpus_builder.match import Matcher
from distillation import ab_common as C
from distillation import config as CFG
from distillation import ridgefit as R


def rank_rows(S):
    """ranks along axis 1 (average-free: permutation ranks; ties ~ never in float sims)."""
    r = np.empty_like(S, dtype=np.float64)
    idx = np.argsort(S, axis=1)
    ar = np.arange(S.shape[1], dtype=np.float64)
    for t in range(S.shape[0]):
        r[t, idx[t]] = ar
    return r


def spearman_rows(SA, SB):
    ra, rb = rank_rows(SA), rank_rows(SB)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = np.einsum("ij,ij->i", ra, rb)
    den = np.sqrt(np.einsum("ij,ij->i", ra, ra) * np.einsum("ij,ij->i", rb, rb)) + 1e-12
    return num / den


def sims_matrix(book, centre, target):            # target [dim, T] -> S [T, N]
    Q = target.T.astype(np.float32) - centre[None, :]
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return Q @ book.T


def load_student():
    import torch
    ck = torch.load(C.AB / "student_mlp.pt", weights_only=False)
    from distillation.student import MLP
    m = MLP(); m.load_state_dict(ck["state"]); m.eval()
    mu, sd = ck["mu"], ck["sd"]

    def f(mel80):
        with torch.no_grad():
            return m(torch.tensor((mel80 - mu) / sd)).numpy().T.astype(np.float32)
    return f


def load_student_gru():
    import torch
    p = C.AB / "student_gru.pt"
    if not p.exists():
        return None
    ck = torch.load(p, weights_only=False)
    from distillation.student import GRUNet
    m = GRUNet(); m.load_state_dict(ck["state"]); m.eval()
    mu, sd = ck["mu"], ck["sd"]

    def f(mel80):
        with torch.no_grad():
            y, _ = m(torch.tensor((mel80 - mu) / sd)[None])
            return y[0].numpy().T.astype(np.float32)
    return f


def main():
    packs = json.loads((C.AB / "packs.json").read_text())
    queries = json.loads((C.AB / "queries.json").read_text())
    maps = R.load_maps()
    student = load_student()
    student_gru = load_student_gru()
    out_rows, results = [], {}
    pick_store = {}                                   # (pack, qid, cond) -> picks@tau_live

    for pname, ppath in packs.items():
        pack = C.load_pack(ppath)
        cf = np.load(C.AB / "corpusfeat" / f"{pname}.npz")
        book, centre = np.asarray(pack.emb), np.asarray(pack.centre)
        mf = cf["mfcc13"].astype(np.float32)
        mcentre = mf.mean(0)
        mbook = mf - mcentre
        mbook /= (np.linalg.norm(mbook, axis=1, keepdims=True) + 1e-8)
        matcher_lat = Matcher(book, centre, 1)
        matcher_mfcc = Matcher(mbook.astype(np.float32), mcentre.astype(np.float32), 1)

        for q in queries:
            qz = np.load(C.AB / "queries" / f"{q['id']}.npz")
            lat, feat89, mel80, mfcc13 = qz["lat"], qz["feat89"], qz["mel80"], qz["mfcc13"]
            targets = {
                "O":  lat,
                "A":  R.apply_ridge(maps[pname], feat89).T.astype(np.float32),
                "Apool": R.apply_ridge(maps["pooled"], feat89).T.astype(np.float32),
                "B":  student(mel80),
                "F":  mfcc13.T.astype(np.float32),
            }
            if student_gru is not None:
                targets["Bgru"] = student_gru(mel80)
            S = {c: sims_matrix(mbook if c == "F" else book,
                                mcentre if c == "F" else centre, t)
                 for c, t in targets.items()}
            SO = S["O"]
            top1_O = SO.argmax(1)
            top5_O = np.argpartition(-SO, 5, axis=1)[:, :5]
            row_base = {"pack": pname, "query": q["id"], "char": q["character"]}

            # sampled picks per tau, matched seed; O2 = oracle with seed+1 (noise floor)
            picks = {}
            for cond, t in targets.items():
                m = matcher_mfcc if cond == "F" else matcher_lat
                for tau in CFG.TAU_SWEEP:
                    r = m.select(t, temperature=tau, rng=np.random.default_rng(CFG.SEED))
                    picks[(cond, tau)] = r
            for tau in CFG.TAU_SWEEP:
                r = matcher_lat.select(lat, temperature=tau,
                                       rng=np.random.default_rng(CFG.SEED + 1))
                picks[("O2", tau)] = r
            pick_store.update({(pname, q["id"], c): picks[(c, CFG.MATCH_TAU_LIVE)].picks
                               for c in list(targets) + ["O2"]})

            for cond in list(targets.keys()) + ["O2"]:
                Sc = S.get(cond, SO)
                top1 = float(np.mean(Sc.argmax(1) == top1_O)) if cond != "O2" else 1.0
                t5c = np.argpartition(-Sc, 5, axis=1)[:, :5]
                top5 = float(np.mean([len(np.intersect1d(t5c[t], top5_O[t])) / 5.0
                                      for t in range(len(SO))])) if cond != "O2" else 1.0
                rho = float(np.mean(spearman_rows(Sc, SO))) if cond != "O2" else 1.0
                row = row_base | {"cond": cond, "top1": round(top1, 4),
                                  "top5": round(top5, 4), "spearman": round(rho, 4)}
                for tau in CFG.TAU_SWEEP:
                    a = np.asarray(picks[(cond, tau)].picks)
                    b = np.asarray(picks[("O", tau)].picks)
                    row[f"agree@{tau:g}"] = round(float(np.mean(a == b)), 4)
                rl = picks[(cond, CFG.MATCH_TAU_LIVE)]
                row |= {"hub@live": round(rl.hubness, 4), "uniq@live": round(rl.unique_frac, 4),
                        "coverage": round(rl.coverage, 4), "picked": round(rl.picked, 4)}
                out_rows.append(row)

            # embedding error buckets (mapped conditions only)
            for cond in ("A", "Apool", "B") + (("Bgru",) if student_gru else ()):
                a = targets[cond].T - centre
                b = lat.T - centre
                cos = np.einsum("ij,ij->i", a, b) / (
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                err = 1.0 - cos
                flux, flat = feat89[:, 87], feat89[:, 85]
                onset = flux >= np.quantile(flux, 0.9)
                noisy = flat >= np.median(flat)
                results.setdefault("emb_err", []).append(
                    {"pack": pname, "query": q["id"], "cond": cond,
                     "mean": round(float(err.mean()), 4),
                     "median": round(float(np.median(err)), 4),
                     "p90": round(float(np.quantile(err, 0.9)), 4),
                     "onset_mean": round(float(err[onset].mean()), 4),
                     "sustain_mean": round(float(err[~onset].mean()), 4),
                     "noisy_mean": round(float(err[noisy].mean()), 4),
                     "tonalish_mean": round(float(err[~noisy].mean()), 4)})
        pack.close()
        print(f"[eval] pack {pname} done")

    results["rows"] = out_rows
    C.jdump(results, C.AB / "out" / "results.json")
    np.save(C.AB / "out" / "picks_live.npy", np.array(list(pick_store.items()), dtype=object),
            allow_pickle=True)

    import csv
    with open(C.AB / "out" / "agreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # ---- compact pivots ----------------------------------------------------
    conds = ["O2", "A", "Apool", "B"] + (["Bgru"] if student_gru else []) + ["F"]
    print("\n=== mean over queries, by pack x condition "
          "(top1 / top5 / spearman / agree@0.05 / hub@0.05) ===")
    for pname in packs:
        print(f"-- {pname}")
        for cond in conds:
            rs = [r for r in out_rows if r["pack"] == pname and r["cond"] == cond]
            if not rs:
                continue
            m = {k: np.mean([r[k] for r in rs])
                 for k in ("top1", "top5", "spearman", "agree@0.05", "hub@live")}
            print(f"   {cond:6s} top1 {m['top1']:.3f}  top5 {m['top5']:.3f}  "
                  f"rho {m['spearman']:.3f}  agree@.05 {m['agree@0.05']:.3f}  "
                  f"hub {m['hub@live']:.3f}")
    print("\n=== tau sweep: mean pick agreement with O (all packs x queries) ===")
    for cond in conds:
        rs = [r for r in out_rows if r["cond"] == cond]
        vals = "  ".join(f"tau{t:g}:{np.mean([r[f'agree@{t:g}'] for r in rs]):.3f}"
                         for t in CFG.TAU_SWEEP)
        print(f"   {cond:6s} {vals}")


if __name__ == "__main__":
    main()
