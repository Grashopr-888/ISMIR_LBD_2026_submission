"""Parse the Elk Stomp serial transcripts in results/board/raw/ (Experiment 2). Added for the public repository.

Each R1 transcript is one 210-330 s run of the full stack (SUSHI + the plugin, Elk's demo loop as input,
`sushi --timing-statistics`, EVL statistics) followed by commands that print the run's logs. The parser
matches line content, not line numbers:

  worker header  `# elkgranular T09 worker ... hop=4480 sr=48000 target_ms=93.333 ... grains=N ... table=f32|int8 table_bytes=B`
  SUMMARY        interval between analyses (p50/p90/p99/min/max, ms) and the loss counters
  STAGES         per-analysis stage times: fe = front-end (resample, 4096-point FFT, mel), st = student,
                 se = search (similarities over all grains, penalties, softmax), sc = scheduling, wk = whole
                 worker analysis; p50/p99/max in ms
  timings table  SUSHI's per-processor share of the 64-sample callback (100 % = 1333.33 us)
  EVL snapshot   in-band switches (ISW) of the real-time audio thread at 25 s and at the end of the run

S6b-bench.txt holds the isolated microbenchmarks (`elkgrain_bench student|match`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "board" / "raw"
RUNS = {  # transcript -> (corpus, table); R1-perc-* were invalid (no pack loaded) and re-run as R1b-perc-*
    "R1-gonza-f32.txt": ("gonza", "f32"), "R1-gonza-int8.txt": ("gonza", "int8"),
    "R1b-perc-f32.txt": ("perc", "f32"), "R1b-perc-int8.txt": ("perc", "int8"),
    "R1-field-f32.txt": ("field", "f32"), "R1-field-int8.txt": ("field", "int8"),
}

_KV = re.compile(r"(\w+)=([-\w.]+)")


def _kv(line: str) -> dict:
    out = {}
    for k, v in _KV.findall(line):
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


@dataclass
class Run:
    transcript: str
    corpus: str
    table: str
    stamp: str = ""
    header: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    callback: dict = field(default_factory=dict)      # processor -> (avg, min, max) in % of the callback
    isw: list = field(default_factory=list)           # EVL in-band switches of the audio thread, per snapshot
    run_seconds: int = 0

    @property
    def grains(self) -> int:
        return int(self.stages["grains"])

    @property
    def period_ms(self) -> float:                     # the analysis period: hop samples / sample rate
        return 1000.0 * self.header["hop"] / self.header["sr"]


def parse_run(path: Path, corpus: str = "", table: str = "") -> Run:
    r = Run(path.name, corpus, table)
    in_timings = False
    for line in path.read_text(errors="replace").splitlines():
        if m := re.match(r"===== RUN (\d+)s:", line):
            r.run_seconds = int(m.group(1))
        if m := re.match(r"stamp=(\S+)", line):
            r.stamp = m.group(1)
        if "# elkgranular T09 worker" in line:
            r.header = _kv(line.split("worker", 1)[1])
        elif ":SUMMARY " in line:
            r.summary = _kv(line.split("SUMMARY", 1)[1])
        elif ":STAGES " in line:
            r.stages = _kv(line.split("STAGES", 1)[1])
        elif line.startswith("Performance timings for all processors"):
            in_timings = True
        elif in_timings and (m := re.match(r"\s*(source|granular|Track total|Engine total)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)):
            r.callback[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
            if m.group(1) == "Engine total":
                in_timings = False
        elif m := re.match(r"\s*\d+\s+\d+\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\S*\s*raspa_pimpl_task", line):
            r.isw.append(int(m.group(1)))
    assert r.header and r.summary and r.stages and "granular" in r.callback, f"incomplete transcript {path.name}"
    assert r.header["table"] == table, (path.name, r.header["table"], table)
    return r


def load_runs(raw: Path = RAW) -> list[Run]:
    return [parse_run(raw / name, *ct) for name, ct in RUNS.items()]


def parse_bench(path: Path = RAW / "S6b-bench.txt") -> list[dict]:
    """One dict per `elkgrain_bench` result line, with the CPU the run was pinned to."""
    out, cpu, env = [], None, ""
    for line in path.read_text(errors="replace").splitlines():
        if m := re.match(r"===== RUN \d+s: (.*)", line):
            cmd = m.group(1)
            cpu = int(re.search(r"taskset -c (\d)", cmd).group(1))
            env = "int8" if "ELKGRANULAR_TABLE=int8" in cmd else ""
        elif line.startswith("BENCH student weights=") or line.startswith("BENCH match pack="):
            meta = _kv(line)
        elif line.startswith("BENCH student p50_us") or line.startswith("BENCH match p50_us"):
            kind = line.split()[1]
            out.append({"kind": kind, "cpu": cpu, **meta, **_kv(line)} | ({"table": meta.get("table", env)} if kind == "match" else {}))
    return out
