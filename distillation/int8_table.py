#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/int8_table.py (changed: package imports, usage lines)
"""D1 offline half — does an int8 matching table change the ordering? (ISMIR 2026 LBD, gap D1)

Quantises a pack's f32 matching table [N, 64] (mean-centred + L2 rows) to int8 with a symmetric
scale, quantises the centred+normalised query to int8 too, computes the similarities with an
EXACT integer dot product (int32; 64 * 127 * 127 < 2^31 — the same arithmetic a NEON
vmull_s8/vpadalq_s16 kernel would do), and recomputes T06's ordering metrics (evaluate.py /
match-quality-ab.md §2) for int8 vs f32 over the T06 packs and the 12 held-out queries:

  vs the f32 table, same query condition   rho (Spearman over all grains), top-1 agreement,
                                           top-5 overlap, oracle-percentile of the int8 argmax
                                           pick in the f32 ordering, pick agreement at tau (same
                                           seed, matched dice)
  vs the music2latent oracle (T06's table) top1 / top5 / rho, oracle-percentile argmax / live,
                                           hubness and unique fraction at tau (Matcher.select
                                           semantics, seed CFG.SEED)
  table bytes                              f32 vs int8 (+ scales)

Two scale schemes (both reported): `row` — one scale per grain, s_i = max|row_i| / 127 (the
runbook's P2e kernel: sims_i = s_i * s_q * dot_i); `dim` — one scale per latent dimension,
s_d = max_i |book[i, d]| / 127, folded into the query before its own quantisation
(sims_i = s_q * dot_i; the experiment design's "per-dim scale at pack build").

Conditions: O = the oracle (music2latent latent of the query, the instrument today) and B = the
deployed student MLP (mel80 -> 64, numpy forward from the RTNeural JSON, standardised input).

Runs from the repository root (it does not import ab_common):

    python -m distillation.int8_table queries --out <qdir>
    python -m distillation.int8_table eval --queries <qdir> \
        --packs perc=<perc.elkgrain> tonal=<tonal.elkgrain> text=<text.elkgrain> --out <results.json>

The query files written by data_prep.py ($LBD_WORK/ab/queries) work as <qdir> too.

`queries` re-cuts the 12 clips exactly as data_prep.make_queries (loudest 8 s window on a 1 s
grid, peak-normalised like live input), music2latent-encodes them (MPS, CPU fallback) and
computes features.frame_features -> <qdir>/<qid>.npz {lat [64,T], mel80 [T,80], feat89 [T,89]}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from corpus_builder import elkgrain
from corpus_builder.match import Matcher
from distillation import config as CFG
from distillation import features as F

HERE = Path(__file__).resolve().parent
STUDENT_JSON = HERE.parent / "weights" / "student_mlp_mel80.json"


# ------------------------------------------------------------------ queries (data_prep replica)
def pick_loudest_offset(g, path: Path, seconds: float) -> float:
    """data_prep.pick_loudest_offset verbatim: loudest `seconds` window on a 1 s grid."""
    y = g.load_mono_44k(str(path), normalize=False)
    sr = g.CODEC_SR
    win = int(seconds * sr)
    if len(y) <= win + 2 * sr:
        return 0.0
    e = np.cumsum(y.astype(np.float64) ** 2)
    starts = np.arange(sr, len(y) - win - sr, sr)
    energy = e[starts + win] - e[starts]
    return float(starts[np.argmax(energy)] / sr)


def make_queries(out: Path):
    import soundfile as sf
    from latent import granular as g
    out.mkdir(parents=True, exist_ok=True)
    codec = None
    manifest = []
    for qid, src, character in CFG.QUERIES:
        npz = out / f"{qid}.npz"
        wav = out / f"{qid}.wav"
        if npz.exists():
            print(f"[queries] {qid}: exists, skipping")
            continue
        off = pick_loudest_offset(g, src, CFG.QUERY_SECONDS)
        y = g.load_mono_44k(str(src), offset=off, seconds=CFG.QUERY_SECONDS)
        sf.write(wav, y, g.CODEC_SR, subtype="FLOAT")
        y44 = g.load_mono_44k(str(wav))                     # peak-normed like live input (ab_common)
        if codec is None:
            codec = g.Codec(device="auto")
            print(f"music2latent codec on '{codec.device}'")
        t0 = time.time()
        try:
            lat = codec.encode(y44)
        except Exception as e:                              # MPS misbehaving -> CPU
            print(f"  encode failed on '{codec.device}' ({e}); retrying on cpu")
            lat = g.Codec(device="cpu").encode(y44)
        lat = np.asarray(lat, dtype=np.float32)
        feats = F.frame_features(y44, lat.shape[1])
        np.savez(npz, lat=lat, mel80=feats["mel80"], feat89=feats["feat89"])
        manifest.append({"id": qid, "source": str(src), "character": character,
                         "offset_s": off, "T": int(lat.shape[1])})
        print(f"[queries] {qid}: {src.name} @ {off:.0f}s -> T={lat.shape[1]} ({time.time()-t0:.1f}s)")
    (out / "queries.json").write_text(json.dumps(manifest, indent=1))


# ------------------------------------------------------------------------------ student (numpy)
def load_student_np(path: Path = STUDENT_JSON):
    doc = json.loads(path.read_text())
    layers = [(np.asarray(L["weights"][0], np.float32), np.asarray(L["weights"][1], np.float32))
              for L in doc["layers"]]                       # kernel [in][out], bias [out]
    mean = np.asarray(doc["input"]["standardize"]["mean"], np.float32)
    std = np.asarray(doc["input"]["standardize"]["std"], np.float32)

    def f(mel80: np.ndarray) -> np.ndarray:                  # [T,80] -> [64,T]
        x = (mel80.astype(np.float32) - mean) / std
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x.T.astype(np.float32)
    return f


# -------------------------------------------------------------------------- similarity schemes
def centred_queries(centre: np.ndarray, target: np.ndarray) -> np.ndarray:  # [dim,T] -> [T,dim]
    Q = target.T.astype(np.float32) - centre[None, :]
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return Q


def sims_f32(book: np.ndarray, Q: np.ndarray) -> np.ndarray:                 # evaluate.sims_matrix
    return (Q @ book.T).astype(np.float32)


def quantise_table(book: np.ndarray, scheme: str):
    """Symmetric int8: returns (bq int8 [N,D], scale f32 [N] for 'row' or [D] for 'dim', bytes)."""
    # float32 arithmetic throughout (max|.| / 127.0f, rint = round-half-even) so the plugin's C++
    # int8 path (Matcher.cpp, ELKGRANULAR_TABLE=int8, the `row` scheme) reproduces these bytes exactly
    if scheme == "row":
        s = np.abs(book).max(axis=1) / np.float32(127)
        s = np.where(s > 0, s, np.float32(1)).astype(np.float32)
        bq = np.clip(np.rint(book / s[:, None]), -127, 127).astype(np.int8)
    elif scheme == "dim":
        s = np.abs(book).max(axis=0) / np.float32(127)
        s = np.where(s > 0, s, np.float32(1)).astype(np.float32)
        bq = np.clip(np.rint(book / s[None, :]), -127, 127).astype(np.int8)
    else:
        raise ValueError(scheme)
    return bq, s, int(bq.nbytes + s.nbytes)


def sims_int8(bq: np.ndarray, s: np.ndarray, scheme: str, Q: np.ndarray) -> np.ndarray:
    """Exact integer dot products, then the scales (float32 like the f32 path's output)."""
    G = Q if scheme == "row" else Q * s[None, :]            # 'dim': fold the per-dim scale in
    sq = np.abs(G).max(axis=1) / np.float32(127)            # one scale per query frame
    sq = np.where(sq > 0, sq, np.float32(1)).astype(np.float32)
    Gq = np.clip(np.rint(G / sq[:, None]), -127, 127).astype(np.int32)
    dots = Gq @ bq.astype(np.int32).T                       # [T,N] exact int32
    S = dots.astype(np.float32) * sq[:, None]
    if scheme == "row":
        S = S * s[None, :]
    return S.astype(np.float32)


# ------------------------------------------------------------------------------------ metrics
def rank_rows(S):                                            # evaluate.rank_rows verbatim
    r = np.empty_like(S, dtype=np.float64)
    idx = np.argsort(S, axis=1)
    ar = np.arange(S.shape[1], dtype=np.float64)
    for t in range(S.shape[0]):
        r[t, idx[t]] = ar
    return r


def spearman_rows(SA, SB):                                   # evaluate.spearman_rows verbatim
    ra, rb = rank_rows(SA), rank_rows(SB)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = np.einsum("ij,ij->i", ra, rb)
    den = np.sqrt(np.einsum("ij,ij->i", ra, ra) * np.einsum("ij,ij->i", rb, rb)) + 1e-12
    return num / den


def top5_overlap(SA, SB) -> float:
    a = np.argpartition(-SA, 5, axis=1)[:, :5]
    b = np.argpartition(-SB, 5, axis=1)[:, :5]
    return float(np.mean([len(np.intersect1d(a[t], b[t])) / 5.0 for t in range(len(SA))]))


def select_rows(S: np.ndarray, tau: float, seed: int) -> list[int]:
    """match.py's select() on precomputed sims rows (anti_hub 0, stick 0): identical draws."""
    rng = np.random.default_rng(seed)
    picks = []
    for sims in S:
        dist = 1.0 - sims
        if tau <= 1e-6:
            j = int(np.argmax(sims))
        else:
            logits = -dist / tau
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            j = int(rng.choice(S.shape[1], p=p))
        picks.append(j)
    return picks


def hubness(picks: list[int], n: int) -> tuple[float, float]:
    c = np.bincount(np.asarray(picks), minlength=n)
    return float(np.sort(c)[::-1][:5].sum() / len(picks)), float((c > 0).sum() / len(picks))


def oracle_percentile(S_ref: np.ndarray, picks) -> float:
    """Mean percentile of each pick within the reference ordering of its row (1.0 = the best)."""
    n = S_ref.shape[1]
    r = rank_rows(S_ref)                                    # 0 = lowest sim, n-1 = best
    return float(np.mean([r[t, j] / (n - 1) for t, j in enumerate(picks)]))


# --------------------------------------------------------------------------------------- eval
def evaluate(qdir: Path, packs: dict[str, Path], out: Path, tau: float, seed: int):
    student = load_student_np()
    qids = [q[0] for q in CFG.QUERIES]
    rows, tables = [], {}
    checked = False
    for pname, ppath in packs.items():
        pack = elkgrain.ElkGrainPack(ppath)
        book, centre = np.asarray(pack.emb, np.float32), np.asarray(pack.centre, np.float32)
        n = book.shape[0]
        tab = {"f32": (None, None, int(book.nbytes))}
        for sch in ("row", "dim"):
            tab[sch] = quantise_table(book, sch)
        tables[pname] = {f"bytes_{k}": v[2] for k, v in tab.items()} | {"grains": n, "latent_dim": int(book.shape[1])}
        print(f"[eval] {pname}: {n} grains; table bytes f32={tab['f32'][2]} "
              f"int8_row={tab['row'][2]} int8_dim={tab['dim'][2]}", flush=True)
        for qid in qids:
            z = np.load(qdir / f"{qid}.npz")
            targets = {"O": z["lat"], "B": student(z["mel80"])}
            S = {}
            for cond, t in targets.items():
                Q = centred_queries(centre, t)
                S[(cond, "f32")] = sims_f32(book, Q)
                for sch in ("row", "dim"):
                    bq, s, _ = tab[sch]
                    S[(cond, sch)] = sims_int8(bq, s, sch, Q)
            SO = S[("O", "f32")]                            # T06's oracle
            picks_O_live = select_rows(SO, tau, seed)
            if not checked:                                 # self-check: our sampler == match.py
                ref = Matcher(book, centre, 1).select(targets["O"], temperature=tau,
                                                      rng=np.random.default_rng(seed)).picks
                assert ref == picks_O_live, "select_rows does not reproduce match.py"
                print("[eval] sampler self-check vs match.Matcher.select: identical picks")
                checked = True
            for (cond, sch), Sc in S.items():
                Sf = S[(cond, "f32")]
                argmax = Sc.argmax(1)
                live = select_rows(Sc, tau, seed)
                hub, uniq = hubness(live, n)
                row = {"pack": pname, "query": qid, "cond": cond, "scheme": sch,
                       # vs the oracle (T06 §2 + oracle regret)
                       "top1_vs_O": float(np.mean(argmax == SO.argmax(1))),
                       "top5_vs_O": top5_overlap(Sc, SO),
                       "rho_vs_O": float(np.mean(spearman_rows(Sc, SO))),
                       "orc_pct_argmax": oracle_percentile(SO, argmax),
                       "orc_pct_live": oracle_percentile(SO, live),
                       f"agree_O@{tau:g}": float(np.mean(np.asarray(live) == np.asarray(picks_O_live))),
                       "hub": hub, "uniq": uniq,
                       # vs the f32 table of the same condition (quantisation alone)
                       "rho_vs_f32": float(np.mean(spearman_rows(Sc, Sf))),
                       "top1_vs_f32": float(np.mean(argmax == Sf.argmax(1))),
                       "top5_vs_f32": top5_overlap(Sc, Sf),
                       "pct_in_f32": oracle_percentile(Sf, argmax),
                       f"agree_f32@{tau:g}": float(np.mean(np.asarray(live)
                                                            == np.asarray(select_rows(Sf, tau, seed)))),
                       "max_abs_sim_err": float(np.abs(Sc - Sf).max()),
                       "T": int(Sc.shape[0])}
                rows.append(row)
        pack.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tau": tau, "seed": seed, "tables": tables, "rows": rows}, indent=1))
    print(f"wrote {out}")

    # ---- pivots (mean over the 12 queries) ------------------------------------------------
    def mean(keys, **sel):
        rs = [r for r in rows if all(r[k] == v for k, v in sel.items())]
        return {k: float(np.mean([r[k] for r in rs])) for k in keys}

    print(f"\n=== D1 offline: int8 table vs f32 (tau={tau:g}, seed={seed}; mean over 12 queries) ===")
    for pname in packs:
        t = tables[pname]
        print(f"-- {pname}: {t['grains']} grains; table MB f32 {t['bytes_f32']/1e6:.3f}  "
              f"int8/row {t['bytes_row']/1e6:.3f}  int8/dim {t['bytes_dim']/1e6:.3f}")
        print("   cond scheme | vs oracle O(f32): top1  top5   rho   orc%argmax orc%live "
              f"agree@{tau:g}  hub   uniq | vs own f32: rho    top1  top5  pct   agree@{tau:g} maxerr")
        for cond in ("O", "B"):
            for sch in ("f32", "row", "dim"):
                m = mean(["top1_vs_O", "top5_vs_O", "rho_vs_O", "orc_pct_argmax", "orc_pct_live",
                          f"agree_O@{tau:g}", "hub", "uniq", "rho_vs_f32", "top1_vs_f32",
                          "top5_vs_f32", "pct_in_f32", f"agree_f32@{tau:g}", "max_abs_sim_err"],
                         pack=pname, cond=cond, scheme=sch)
                print(f"   {cond:4s} {sch:6s} | {m['top1_vs_O']:.3f} {m['top5_vs_O']:.3f} "
                      f"{m['rho_vs_O']:.3f}   {m['orc_pct_argmax']:.3f}    {m['orc_pct_live']:.3f}   "
                      f"{m[f'agree_O@{tau:g}']:.3f}    {m['hub']:.3f} {m['uniq']:.3f} | "
                      f"{m['rho_vs_f32']:.4f} {m['top1_vs_f32']:.3f} {m['top5_vs_f32']:.3f} "
                      f"{m['pct_in_f32']:.4f} {m[f'agree_f32@{tau:g}']:.3f}  {m['max_abs_sim_err']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queries", help="cut + encode + featurise the 12 held-out queries")
    q.add_argument("--out", required=True, help="directory for <qid>.npz / .wav")
    e = sub.add_parser("eval", help="int8 vs f32 ordering metrics")
    e.add_argument("--queries", required=True)
    e.add_argument("--packs", nargs="+", required=True, help="name=path.elkgrain ...")
    e.add_argument("--out", required=True, help="results json")
    e.add_argument("--tau", type=float, default=CFG.MATCH_TAU_LIVE)
    e.add_argument("--seed", type=int, default=CFG.SEED)
    a = ap.parse_args(argv)
    if a.cmd == "queries":
        make_queries(Path(a.out))
    else:
        packs = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.packs}
        evaluate(Path(a.queries), packs, Path(a.out), a.tau, a.seed)


if __name__ == "__main__":
    main()
