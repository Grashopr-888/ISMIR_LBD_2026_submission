"""Markdown tables for the notebooks. Added for the public repository."""
from __future__ import annotations

from IPython.display import Markdown, display


def _fmt(v, f):
    if v is None:
        return "—"
    if callable(f):
        return f(v)
    return format(v, f) if f else str(v)


def table(rows: list[dict], cols: list[tuple]):
    """cols: (key, header[, format spec or callable])."""
    cols = [(c[0], c[1], c[2] if len(c) > 2 else "") for c in cols]
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(_fmt(r.get(k), f) for k, _, f in cols) + " |" for r in rows]
    display(Markdown("\n".join(lines)))


def checks(items: list[tuple]) -> bool:
    """items: (what, paper, this notebook, equal?) -> a table; equal? is True, False, or None when the number
    cannot be computed from the data in this repository. Returns True if no item differs."""
    mark = {True: "✓", False: "✗ differs", None: "n/a"}
    rows = [{"what": w, "paper": p, "here": h, "ok": mark[ok]} for w, p, h, ok in items]
    table(rows, [("what", "Paper element"), ("paper", "Printed in the paper"), ("here", "Computed here"), ("ok", "Equal")])
    return all(ok is not False for *_, ok in items)
