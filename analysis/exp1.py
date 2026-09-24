"""Table 1 of the paper from results/exp1/metrics_per_query.csv (Experiment 1). Added for the public repository.

Aggregation, as the paper's caption states: every column is a mean over all 48 query-corpus pairs (12 held-out
queries against each of the four corpora: percussive, tonal, textural, field recordings), at tau = 0.05. The
teacher row takes every column from the teacher's second seed. Encoder codes: O teacher (music2latent) with the
reference seed, O2 teacher with a second seed, F MFCC cosine (no map), Apool ridge map pooled over all corpora,
Bgru GRU-128 student, B MLP student.
"""
from __future__ import annotations

import csv
from pathlib import Path

EXP1 = Path(__file__).resolve().parents[1] / "results" / "exp1"
CORPORA = ("perc", "tonal", "text", "field")

# learned parameters (weights and biases) and multiply-accumulates per frame, from weights/*.json:
# ridge map 89 x 64 plus 64 biases, GRU-128 with its dense output layer, the MLP 80-256-256-64
PARAMS = {"Apool": 89 * 64 + 64, "Bgru": 88_896, "B": 102_976}
MACS = {"Apool": 89 * 64, "Bgru": 88_064, "B": 102_400}

# Table 1 as printed in the submitted paper (24 Sep 2026), for comparison
PAPER = {
    "O2":    {"label": "Teacher, second seed", "params": None,     "rho": None,   "pct": ".981", "agr": ".026", "hub": ".17"},
    "F":     {"label": "MFCC cosine, no map",  "params": None,     "rho": ".575", "pct": ".835", "agr": ".007", "hub": ".12"},
    "Apool": {"label": "Ridge map",            "params": "5.7 k",  "rho": ".797", "pct": ".933", "agr": ".039", "hub": ".20"},
    "Bgru":  {"label": "GRU-128",              "params": "0.09 M", "rho": ".826", "pct": ".945", "agr": ".045", "hub": ".21"},
    "B":     {"label": "Student (MLP)",        "params": "0.10 M", "rho": ".829", "pct": ".945", "agr": ".047", "hub": ".22"},
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
    """Unrounded Table 1 cells, every column over all 48 query-corpus pairs. The teacher row is the second seed;
    the reference seed's Pct. and Hub are added for comparison."""
    out = []
    for enc in ("O2", "F", "Apool", "Bgru", "B"):
        row = {"encoder": enc, "label": PAPER[enc]["label"], "params": PARAMS.get(enc), "macs": MACS.get(enc),
               "rho": None if enc == "O2" else cell(rows, enc, "spearman")[0],
               "pct": cell(rows, enc, "live_pctile")[0], "agr": cell(rows, enc, "agree@0.05")[0],
               "hub": cell(rows, enc, "hub@live")[0]}
        if enc == "O2":
            row["pct_seed0"] = cell(rows, "O", "live_pctile")[0]
            row["hub_seed0"] = cell(rows, "O", "hub@live")[0]
        out.append(row)
    return out


def per_query_lead(rows, a: str = "B", b: str = "Apool", key: str = "spearman") -> dict:
    """How consistently encoder a leads encoder b: per held-out query, the mean over the four corpora of a minus b.
    Returns the mean of those differences, the queries where a leads, and the query-corpus pairs where a leads."""
    by = {}
    for r in rows:
        if r["encoder"] in (a, b):
            by.setdefault((r["query"], r["corpus"]), {})[r["encoder"]] = r[key]
    diffs = {qc: v[a] - v[b] for qc, v in by.items()}
    per_q = {}
    for (q, _), d in diffs.items():
        per_q.setdefault(q, []).append(d)
    per_q = {q: mean(ds) for q, ds in per_q.items()}
    return {"mean": mean(per_q.values()), "queries_ahead": sum(d > 0 for d in per_q.values()), "queries": len(per_q),
            "pairs_ahead": sum(d > 0 for d in diffs.values()), "pairs": len(diffs)}


def fmt_count(n):
    """Table 1's style for parameter and MAC counts: 5.7 k, 0.09 M, 0.10 M."""
    return "---" if n is None else (f"{n / 1e3:.1f} k" if n < 1e4 else f"{n / 1e6:.2f} M")


def fmt3(x):
    return "---" if x is None else f"{x:.3f}".lstrip("0")


def fmt2(x):
    return "---" if x is None else f"{x:.2f}".lstrip("0")
