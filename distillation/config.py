# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/config.py (changed: file lists read from a TOML file instead of hard-coded paths)
"""Experiment 1 data designation: packs, held-out queries, student-training extras.

The original file hard-coded the author's recordings (not redistributable). Here the same lists are
read from a TOML file named by $LBD_DATA_CONFIG (default: distillation/data.toml, git-ignored);
distillation/data.example.toml shows the format. Without that file every list is empty, so the
modules still import (the notebooks use them on a generated demo signal) and data_prep.py refuses
to run.

Rules enforced by assertions below (as in the original):
  - QUERY files never appear in any pack and never in student training;
  - student trains on pack corpora + TRAIN_EXTRA (+ optional MUSDB18 previews) only.
An existing pack (the paper's field-recording corpus) can be evaluated as a fourth metrics-only
corpus via [field]; its queries are held out from it too (checked in data_prep).
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("LBD_DATA_CONFIG", HERE / "data.toml"))
DATA_CONFIGURED = CONFIG_PATH.exists()

_cfg: dict = tomllib.loads(CONFIG_PATH.read_text()) if DATA_CONFIGURED else {}
_root = Path(os.path.expandvars(_cfg.get("data_root", "."))).expanduser()
if not _root.is_absolute():
    _root = (CONFIG_PATH.parent / _root).resolve()


def _p(rel: str) -> Path:
    q = Path(os.path.expandvars(rel)).expanduser()
    return q if q.is_absolute() else _root / q


# --------------------------------------------------------------------- packs (5-15 min each)
PACKS: dict[str, list[Path]] = {name: [_p(f) for f in files]
                                for name, files in _cfg.get("packs", {}).items()}

# optional existing pack evaluated as a metrics-only corpus (the paper's field recordings)
_field = _cfg.get("field", {})
FIELD_PACK: Path | None = _p(_field["pack"]) if "pack" in _field else None
FIELD_CORPUS_DIR: Path | None = _p(_field["corpus_dir"]) if "corpus_dir" in _field else None

# ------------------------------------------------------- held-out queries (never in packs)
# (id, source file, character); offset auto-picked = loudest 8 s window (stems have long silences).
QUERY_SECONDS = float(_cfg.get("query_seconds", 8.0))
QUERIES: list[tuple[str, Path, str]] = [(q["id"], _p(q["file"]), q["character"])
                                        for q in _cfg.get("queries", [])]

# ------------------------------------------------- student-training extras (training ONLY)
TRAIN_EXTRA: list[Path] = [_p(f) for f in _cfg.get("train_extra", [])]

# MUSDB18-7 previews for training diversity: N tracks x stems {drums=1, other=3, vocals=4}
_musdb = _cfg.get("musdb", {})
MUSDB: Path | None = _p(_musdb["dir"]) if "dir" in _musdb else None
MUSDB_N_TRACKS = int(_musdb.get("n_tracks", 25))
MUSDB_STEMS = {"drums": 1, "other": 3, "vocals": 4}

# ------------------------------------------------------------------------------- sanity
_q = {p for _, p, _ in QUERIES}
_pk = {p for fs in PACKS.values() for p in fs}
assert not (_q & _pk), "query files must be held out from packs"
assert not (_q & set(TRAIN_EXTRA)), "query files must be held out from student training"
assert not (set(TRAIN_EXTRA) & _pk), "extras duplicate pack files (allowed but pointless)"
for _n, _fs in PACKS.items():
    assert len(_fs) == len(set(_fs))
for _src in list(_q | _pk | set(TRAIN_EXTRA)):
    assert _src.exists(), f"missing source: {_src}"

# field pack held-out check happens in data_prep (its file list comes from the pack JSON).

MATCH_TAU_LIVE = 0.05          # the instrument's live default
TAU_SWEEP = [0.0, 0.03, 0.05, 0.15]
SEED = 0
