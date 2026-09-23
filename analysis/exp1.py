"""Table 1 of the paper from results/exp1/metrics_per_query.csv (Experiment 1). Added for the public repository.

Aggregation, as the paper's caption states: Spearman rho and percentile rank (Pct.) are means over the
percussive, tonal and textural corpora (12 queries each, 36 query-corpus pairs); selection agreement
(Agr.) and hub share (Hub) are means over all 48 query-corpus pairs (the field corpus included), at
tau = 0.05. Encoder codes: O teacher (music2latent) with the reference seed, O2 teacher with a second seed,
F MFCC cosine (no map), Apool ridge map pooled over all corpora, Bgru GRU-128 student, B MLP student.
"""
from __future__ import annotations

import csv
from pathlib import Path

EXP1 = Path(__file__).resolve().parents[1] / "results" / "exp1"
CHAR_CORPORA = ("perc", "tonal", "text")

# multiply-accumulates per frame (student.py export; ridge 89 x 64)
MACS = {"Apool": 89 * 64, "Bgru": 88_064, "B": 102_400}

# Table 1 as printed in Draft 3 (paper/latex/lbd2026.tex), for comparison
PAPER = {
    "O2":    {"label": "Teacher, second seed", "rho": None, "pct": ".978", "agr": ".026", "hub": ".17"},
    "F":     {"label": "MFCC cosine, no map",  "rho": ".585", "pct": ".841", "agr": ".007", "hub": ".12"},
    "Apool": {"label": "Ridge map",            "rho": ".811", "pct": ".937", "agr": ".039", "hub": ".20"},
    "Bgru":  {"label": "GRU-128",              "rho": ".841", "pct": ".949", "agr": None,   "hub": None},
    "B":     {"label": "Student (MLP)",        "rho": ".843", "pct": ".950", "agr": ".047", "hub": ".22"},
}


def load(path: Path = EXP1 / "metrics_per_query.csv") -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("corpus", "query", "character", "encoder"):
                r[k] = float(v)
    return rows


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def cell(rows, encoder, key, corpora=None):
    sel = [r[key] for r in rows if r["encoder"] == encoder and (corpora is None or r["corpus"] in corpora)]
    return mean(sel), len(sel)


def table1(rows) -> list[dict]:
    """Unrounded Table 1 cells. The teacher row: Pct. and Hub for both seeds, Agr. for the second seed."""
    out = []
    for enc in ("O2", "F", "Apool", "Bgru", "B"):
        rho = None if enc == "O2" else cell(rows, enc, "spearman", CHAR_CORPORA)[0]
        pct = cell(rows, enc, "live_pctile", CHAR_CORPORA)[0]
        agr = cell(rows, enc, "agree@0.05")[0]
        hub = cell(rows, enc, "hub@live")[0]
        row = {"encoder": enc, "label": PAPER[enc]["label"], "macs": MACS.get(enc), "rho": rho,
               "pct": pct, "agr": agr, "hub": hub}
        if enc == "O2":   # what the paper's teacher row printed: the reference seed's percentile and hub share
            row["pct_seed0"] = cell(rows, "O", "live_pctile", CHAR_CORPORA)[0]
            row["hub_seed0"] = cell(rows, "O", "hub@live")[0]
        out.append(row)
    return out


def fmt3(x):
    return "---" if x is None else f"{x:.3f}".lstrip("0")


def fmt2(x):
    return "---" if x is None else f"{x:.2f}".lstrip("0")
