#!/usr/bin/env python
"""Export Experiment 1's per-query metrics (numbers only) to results/exp1/. Added for the public repository.

Reads the outputs of evaluate.py, supplement.py and int8_table.py under $LBD_WORK and writes:

  metrics_per_query.csv  one row per (corpus, query, query encoder): ordering (top1, top5, Spearman rho),
                         selection agreement with the teacher at each tau, hub share / unique fraction at the
                         live tau, and the teacher percentile of each encoder's argmax and live selection
  embedding_error.csv    1 - mean-centred cosine to the teacher latent, by onset/sustained, noisy/tonal frames
  ridge_cv.json          ridge lambda chosen by leave-one-file-out CV and its score, per corpus and pooled
  corpora.json           grains, files and minutes per corpus; query ids, characters and frame counts
  int8_vs_f32.json       the 8-bit table study (int8_table.py eval), unchanged
  training_pairs.json    the student's training set: pairs, files and validation split by source kind
                         (counts only; from data_prep.py train, if it was run)

No latent, audio, file name or path is written. The second-seed teacher (O2) gets its own live percentile
here (the paper's Table 1 row "Teacher, second seed" printed the first seed's percentile and hub share).

    python -m distillation.export_results --int8 $LBD_WORK/int8/int8_vs_f32.json --out results/exp1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from corpus_builder.match import Matcher
from distillation import ab_common as C
from distillation import config as CFG
from distillation.evaluate import sims_matrix

RESULT_KEYS = ["top1", "top5", "spearman"] + [f"agree@{t:g}" for t in CFG.TAU_SWEEP] + \
              ["hub@live", "uniq@live", "coverage", "picked"]
SUPP_KEYS = ["det_pctile", "det_regret", "live_pctile", "live_regret"]


def o2_percentiles(packs: dict, queries: list) -> dict:
    """(pack, query) -> teacher percentile of the teacher's own selections drawn with seed + 1."""
    out = {}
    for pname, ppath in packs.items():
        pack = C.load_pack(ppath)
        book, centre = np.asarray(pack.emb), np.asarray(pack.centre)
        N = pack.n
        m = Matcher(book, centre, 1)
        for q in queries:
            lat = np.load(C.AB / "queries" / f"{q['id']}.npz")["lat"]
            SO = sims_matrix(book, centre, lat)
            order = SO.argsort(1)
            pct = np.empty_like(SO)
            ar = (np.arange(N, dtype=np.float32) + 1.0) / N
            for t in range(len(SO)):
                pct[t, order[t]] = ar
            r = m.select(lat, temperature=CFG.MATCH_TAU_LIVE, rng=np.random.default_rng(CFG.SEED + 1))
            lp = np.asarray(r.picks)
            best = SO.max(1)
            out[(pname, q["id"])] = (round(float(np.mean(pct[np.arange(len(lp)), lp])), 4),
                                     round(float(np.mean(SO[np.arange(len(lp)), lp] - best[:len(lp)])), 4))
        pack.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--int8", type=Path, help="int8_table.py eval output (optional)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "exp1")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    packs = json.loads((C.AB / "packs.json").read_text())
    queries = json.loads((C.AB / "queries.json").read_text())
    res = json.loads((C.AB / "out" / "results.json").read_text())
    sup = {(r["pack"], r["query"], r["cond"]): r
           for r in json.loads((C.AB / "out" / "supplement.json").read_text())["rows"]}
    o2 = o2_percentiles(packs, queries)

    rows = []
    for r in res["rows"]:
        k = (r["pack"], r["query"], r["cond"])
        row = {"corpus": r["pack"], "query": r["query"], "character": r["char"], "encoder": r["cond"]}
        row |= {key: r[key] for key in RESULT_KEYS}
        if r["cond"] == "O2":               # same similarities as O, so O's argmax; live pick drawn with seed + 1
            row |= {"det_pctile": 1.0, "det_regret": 0.0,
                    "live_pctile": o2[k[:2]][0], "live_regret": o2[k[:2]][1]}
        else:
            row |= {key: sup[k][key] for key in SUPP_KEYS}
        rows.append(row)
    with open(a.out / "metrics_per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ee = [{"corpus": r["pack"], "query": r["query"], "encoder": r["cond"]} |
          {k: v for k, v in r.items() if k not in ("pack", "query", "cond")} for r in res["emb_err"]]
    with open(a.out / "embedding_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ee[0]))
        w.writeheader()
        w.writerows(ee)

    (a.out / "ridge_cv.json").write_text(json.dumps(json.loads((C.AB / "ridge_report.json").read_text()),
                                                   indent=1, sort_keys=True) + "\n")

    corpora = {}
    for pname, ppath in packs.items():
        with C.load_pack(ppath) as p:
            st = p.meta["stats"]
            corpora[pname] = {"grains": int(p.n), "files": int(p.n_files),
                              "minutes": round(st["corpus_seconds44"] / 60.0, 2),
                              "dropped_silent": int(st["dropped_silent"])}
    doc = {"corpora": corpora,
           "queries": [{"id": q["id"], "character": q["character"], "frames": q["T"]} for q in queries],
           "tau_live": CFG.MATCH_TAU_LIVE, "tau_sweep": CFG.TAU_SWEEP, "seed": CFG.SEED}
    (a.out / "corpora.json").write_text(json.dumps(doc, indent=1) + "\n")

    gp = C.AB / "train" / "groups.json"
    if gp.exists():
        groups = json.loads(gp.read_text())
        kinds = sorted({g["kind"] for g in groups})
        tp = {"pairs": sum(g["frames"] for g in groups), "files": len(groups),
              "by_kind": {k: {"files": sum(g["kind"] == k for g in groups),
                              "pairs": sum(g["frames"] for g in groups if g["kind"] == k)} for k in kinds},
              "validation": {"files": sum(g["split"] == "val" for g in groups),
                             "pairs": sum(g["frames"] for g in groups if g["split"] == "val")},
              "note": "pack = the four evaluation corpora (always in training; the queries are held out), "
                      "extra = training-only recordings, musdb = MUSDB18 7 s previews (25 tracks x 3 stems)"}
        (a.out / "training_pairs.json").write_text(json.dumps(tp, indent=1) + "\n")

    if a.int8:
        d = json.loads(a.int8.read_text())
        (a.out / "int8_vs_f32.json").write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {len(rows)} metric rows, {len(ee)} embedding-error rows to {a.out}")


if __name__ == "__main__":
    main()
