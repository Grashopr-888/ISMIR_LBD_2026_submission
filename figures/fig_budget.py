# -*- coding: utf-8 -*-
# Provenance: Grashopr-888/ISMIR_LBD_2026@2bf21d1:paper/figures/fig_budget_i0.py (changed: data parsed from the board transcripts instead of the hand-copied budget.json; estimate branch removed)
"""Figure 2 of the paper: median time per analysis on the Elk Stomp, by stage and corpus size, with the corpus
latent table in 32-bit floats and in 8 bits.

Data: the STAGES lines (p50 per stage, ms) of the six resampler-fixed runs in results/board/raw/ (parsed by
analysis/board_logs.py). Each corpus is drawn as two stacked bars: front-end + student + search with the
32-bit table, and the same with the 8-bit table. Every bar is measured on the board.
Drawn at the printed column width (82 mm = 232.4 pt) so fonts print at their own size; greyscale-safe
(a dotted pattern marks the 8-bit search; Paul Tol's 'vibrant' colours).

    python figures/fig_budget.py            # writes figures/fig_budget.pdf and .png
"""
import os
import sys

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")          # file output only when run as a script; notebooks keep their inline backend
import matplotlib.pyplot as plt  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(H))
from analysis.board_logs import load_runs  # noqa: E402


def stage_rows(runs):
    """Per corpus (ascending size): search from both runs; front-end and student from the 32-bit run for
    both bars, as in the paper's figure (budget.json). The 8-bit run's own front-end and student medians
    differ by less than 0.06 ms (fe8, st8 below; notebook 05 prints both)."""
    by = {(r.corpus, r.table): r for r in runs}
    out = []
    for corpus in sorted({r.corpus for r in runs}, key=lambda c: by[(c, "f32")].grains):
        f32, i8 = by[(corpus, "f32")], by[(corpus, "int8")]
        out.append(dict(corpus=corpus, grains=f32.grains,
                        fe32=f32.stages["fe_p50"], st32=f32.stages["st_p50"], m32=f32.stages["se_p50"],
                        fe8=i8.stages["fe_p50"], st8=i8.stages["st_p50"], m8=i8.stages["se_p50"]))
    return out


def draw(rows, out_base=os.path.join(H, "fig_budget")):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})   # embed TrueType, not Type 3
    fig, ax = plt.subplots(figsize=(232.4 / 72, 1.08), dpi=200)          # one ISMIR column
    w = 0.36
    C_FRONT, C_STUD, C_M32, C_M8 = "#BBBBBB", "#0077BB", "#EE7733", "#009988"

    def bar(i, off, val, bottom, color, label=None, pattern=None):
        ax.bar(i + off, val, w, bottom=bottom, color=color, edgecolor="black", linewidth=0.5,
               hatch=pattern, label=label)

    for i, r in enumerate(rows):
        for off, fe, st, m, color, lab, pat in ((-w / 2, r["fe32"], r["st32"], r["m32"], C_M32, "search, 32-bit table", None),
                                                (w / 2, r["fe32"], r["st32"], r["m8"], C_M8, "search, 8-bit table", "....")):
            first = i == 0 and off < 0
            bar(i, off, fe, 0, C_FRONT, "front-end" if first else None)
            bar(i, off, st, fe, C_STUD, "student" if first else None)
            bar(i, off, m, fe + st, color, lab if i == 0 else None, pattern=pat)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f'{r["grains"]:,}' for r in rows], fontsize=7.5)
    ax.set_xlabel("corpus size (grains)", fontsize=7.5, labelpad=1)
    ax.set_ylabel("time per\nanalysis (ms)", fontsize=7.5)
    ax.set_ylim(0, 20.0)
    ax.set_yticks([0, 10, 20])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1,
              handlelength=1.3, labelspacing=0.35)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}")
    return fig


if __name__ == "__main__":
    rows = stage_rows(load_runs())
    draw(rows)
    print("fig_budget written |", [(r["grains"], round(r["fe32"] + r["st32"] + r["m32"], 1),
                                     round(r["fe32"] + r["st32"] + r["m8"], 1)) for r in rows])
