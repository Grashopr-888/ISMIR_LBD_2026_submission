#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/supplement.py (changed: package imports, usage lines)
"""Supplemental T06 metric: ORACLE REGRET — when a cheap condition picks a grain, how good
is that grain according to music2latent?

Exact-pick agreement collapses at live tau (the oracle itself resamples differently seed to
seed), so the perceptually honest question is: do the grains a cheap condition selects sit
in the oracle's own top ranks?  For each (pack, query, cond):

  det_regret   mean_t [ S_O(t, argmax_cond) - S_O(t, argmax_O) ]   cosine points lost, tau=0
  det_pctile   mean_t oracle-percentile of argmax_cond             1.0 = oracle's own best
  live_pctile  mean_t oracle-percentile of the tau=0.05/seed-0 sampled pick
               (reference: O's own sampled picks -> how far below its argmax the live
                instrument ALREADY sits by design)

Writes $LBD_WORK/ab/out/supplement.json; prints pack x cond pivot + same/cross-character split.
Run: python -m distillation.supplement
"""
from __future__ import annotations

import json

import numpy as np

from corpus_builder.match import Matcher
from distillation import ab_common as C
from distillation import config as CFG
from distillation import ridgefit as R
from distillation.evaluate import load_student, load_student_gru, sims_matrix


def main():
    packs = json.loads((C.AB / "packs.json").read_text())
    queries = json.loads((C.AB / "queries.json").read_text())
    maps = R.load_maps()
    student = load_student()
    student_gru = load_student_gru()
    rows = []
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
        N = pack.n
        for q in queries:
            qz = np.load(C.AB / "queries" / f"{q['id']}.npz")
            lat = qz["lat"]
            targets = {"O": lat,
                       "A": R.apply_ridge(maps[pname], qz["feat89"]).T.astype(np.float32),
                       "Apool": R.apply_ridge(maps["pooled"], qz["feat89"]).T.astype(np.float32),
                       "B": student(qz["mel80"]),
                       "F": qz["mfcc13"].T.astype(np.float32)}
            if student_gru is not None:
                targets["Bgru"] = student_gru(qz["mel80"])
            SO = sims_matrix(book, centre, lat)                     # [T, N]
            order = SO.argsort(1)                                   # ascending
            pct = np.empty_like(SO)
            ar = (np.arange(N, dtype=np.float32) + 1.0) / N
            for t in range(len(SO)):
                pct[t, order[t]] = ar
            best_O = SO.max(1)
            aO = SO.argmax(1)
            Trng = np.arange(len(SO))
            for cond, tgt in targets.items():
                Sc = sims_matrix(mbook if cond == "F" else book,
                                 mcentre if cond == "F" else centre, tgt)
                ac = Sc.argmax(1)
                det_regret = float(np.mean(SO[Trng, ac] - best_O))
                det_pct = float(np.mean(pct[Trng, ac]))
                m = matcher_mfcc if cond == "F" else matcher_lat
                r = m.select(tgt, temperature=CFG.MATCH_TAU_LIVE,
                             rng=np.random.default_rng(CFG.SEED))
                lp = np.asarray(r.picks)
                live_pct = float(np.mean(pct[Trng[:len(lp)], lp]))
                live_regret = float(np.mean(SO[Trng[:len(lp)], lp] - best_O[:len(lp)]))
                rows.append({"pack": pname, "query": q["id"], "char": q["character"],
                             "cond": cond,
                             "det_regret": round(det_regret, 4),
                             "det_pctile": round(det_pct, 4),
                             "live_pctile": round(live_pct, 4),
                             "live_regret": round(live_regret, 4),
                             "same_char": (q["character"] == pname) or
                                          (pname == "field")})
        pack.close()
        print(f"[supp] {pname} done")
    C.jdump({"rows": rows}, C.AB / "out" / "supplement.json")

    conds = ["O", "A", "Apool", "B"] + (["Bgru"] if student_gru else []) + ["F"]
    print("\n=== oracle regret / percentile (mean over queries) ===")
    print("cond    | det_pctile det_regret | live_pctile live_regret   (O live = design ref)")
    for pname in packs:
        print(f"-- {pname}")
        for cond in conds:
            rs = [r for r in rows if r["pack"] == pname and r["cond"] == cond]
            m = {k: np.mean([r[k] for r in rs])
                 for k in ("det_pctile", "det_regret", "live_pctile", "live_regret")}
            print(f"   {cond:6s}|   {m['det_pctile']:.4f}   {m['det_regret']:+.4f}  |"
                  f"   {m['live_pctile']:.4f}   {m['live_regret']:+.4f}")
    print("\n=== same-character vs cross-character (live_pctile, char packs only) ===")
    for cond in conds:
        s = [r["live_pctile"] for r in rows
             if r["cond"] == cond and r["pack"] != "field" and r["char"] == r["pack"]]
        x = [r["live_pctile"] for r in rows
             if r["cond"] == cond and r["pack"] != "field" and r["char"] != r["pack"]]
        print(f"   {cond:6s} same {np.mean(s):.4f}   cross {np.mean(x):.4f}")


if __name__ == "__main__":
    main()
