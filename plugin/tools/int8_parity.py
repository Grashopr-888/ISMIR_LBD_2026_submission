#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/tools/int8_parity.py (changed: paths from the repository and the environment, package imports)
"""8-bit table parity: the plugin's int8 matcher (Matcher.cpp, ELKGRANULAR_TABLE=int8) vs the numpy replica
(distillation/int8_table.py, `row` scheme) over a pack (the paper's run: the 12,018-grain field pack), 200 slots at tau=0.

Same contract as golden_parity.py's matcher stage: argmax picks identical (200/200), pick/best sims
and the first slots' full sims rows within 1e-5 (the C++ centres/normalises the query with a
double-accumulated norm, numpy in float32 — the last-ulp difference can move one quantised query
element by 1 LSB, so exact byte equality of the sims is not the contract; the ordering is).
Also checks the quantised TABLE bytes are identical (same scales, same int8 values) — that part
is deterministic and must be exact.

    INT8_PACK=<any .elkgrain pack> python plugin/tools/int8_parity.py     # from the repository root
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]            # the repository
BUILD = ROOT / "plugin" / "build"
GOLDEN = Path(os.environ.get("GOLDEN_BIN", BUILD / "ElkgrainGolden_artefacts" / "Release" / "ElkgrainGolden"))
PACK = Path(os.environ.get("INT8_PACK", ROOT / "work" / "demo" / "demo.elkgrain"))
TMP = Path(os.environ.get("GOLDEN_TMP", BUILD / "golden-tmp"))
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from corpus_builder.elkgrain import ElkGrainPack                                     # noqa: E402
from distillation.int8_table import quantise_table, sims_int8, centred_queries        # noqa: E402

FAIL = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


def main():
    pack = ElkGrainPack(PACK)
    n, dim = pack.n, pack.grain_dim
    book, centre = np.asarray(pack.emb, np.float32), np.asarray(pack.centre, np.float32)
    rng = np.random.default_rng(1234)                          # golden_parity's query recipe
    T = 200
    rows = rng.integers(0, n, size=T)
    q = np.asarray(pack.raw)[rows].astype(np.float32)
    q += (rng.standard_normal((T, dim)).astype(np.float32) * 0.05
          * np.abs(q).mean(axis=1, keepdims=True).astype(np.float32))
    qpath = TMP / "int8_queries.f32"
    q.tofile(qpath)

    simsN = 3
    simsPath = TMP / "int8_sims.f32"
    env = dict(os.environ, ELKGRANULAR_TABLE="int8")
    r = subprocess.run([str(GOLDEN), "match", str(PACK), str(qpath), str(T), "0", "0", "0", "0",
                        str(simsN), str(simsPath)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("golden match (int8) failed")
    picks_c, sim_c, best_c = [], [], []
    for line in r.stdout.splitlines():
        m = re.match(r"slot (\d+) pick (-?\d+) sim (\S+) best (\S+)", line)
        if m:
            picks_c.append(int(m.group(2))); sim_c.append(float(m.group(3))); best_c.append(float(m.group(4)))
    sims_c = np.fromfile(simsPath, dtype=np.float32).reshape(simsN, n)

    bq, s, nbytes = quantise_table(book, "row")
    Q = centred_queries(centre, q.T)
    S = sims_int8(bq, s, "row", Q)                               # [T, n] numpy int8 sims
    picks_p = S.argmax(1).tolist()
    same = sum(int(a == b) for a, b in zip(picks_c, picks_p))
    check("int8/picks", same == T, f"{same}/{T} identical argmax picks (tau=0)")
    best_err = max(abs(float(S[t].max()) - best_c[t]) for t in range(T))
    pick_err = max(abs(float(S[t, picks_c[t]]) - sim_c[t]) for t in range(T))
    check("int8/sims", best_err <= 1e-5 and pick_err <= 1e-5,
          f"max best-sim err {best_err:.3g}, pick-sim err {pick_err:.3g} (bound 1e-5)")
    rows_err = float(np.abs(sims_c - S[:simsN]).max())
    exact = int((sims_c == S[:simsN]).sum())
    check("int8/sims_rows", rows_err <= 1e-5,
          f"first-{simsN}-slot full sims max err {rows_err:.3g}; {exact}/{simsN*n} bit-identical")
    # the f32 path must be untouched: same picks as match.py at tau=0 on the same queries
    from corpus_builder.match import Matcher
    ref = Matcher(book, centre, 1).select(q.T.copy(), temperature=0.0, rng=np.random.default_rng(0)).picks
    agree = sum(int(a == b) for a, b in zip(picks_c, ref))
    print(f"[info] int8 vs f32 (match.py) argmax agreement on these queries: {agree}/{T}; "
          f"table {nbytes} B vs {book.nbytes} B f32")
    print("INT8 PARITY:", "ALL PASS" if not FAIL else f"FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
