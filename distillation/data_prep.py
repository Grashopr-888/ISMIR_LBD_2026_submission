#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/data_prep.py (changed: package imports, paths from config, work directory)
"""T06 data prep — stage corpora, build packs, cut queries, assemble student training pairs.

    python -m distillation.data_prep packs      # symlink-stage + build 3 packs
    python -m distillation.data_prep queries    # cut 12 held-out clips + encode + features
    python -m distillation.data_prep corpusfeat # per-pack (features -> latent) tables
    python -m distillation.data_prep train      # student pairs (packs + extras + MUSDB)
    python -m distillation.data_prep all

The file lists come from distillation/config.py ($LBD_DATA_CONFIG). Intermediates land in $LBD_WORK/ab/.  Sources are read-only; MUSDB18-7 .stem.mp4 previews
are decoded with ffmpeg (stream map: 0=mixture 1=drums 2=bass 3=other 4=vocals).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from distillation import ab_common as C
from distillation import config as CFG
from distillation import features as F


def stage_and_build_packs():
    from corpus_builder import build_corpus
    packs = {}
    for name, files in CFG.PACKS.items():
        stage = C.AB / "corpora" / name
        stage.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = stage / src.name
            if not dst.is_symlink():
                dst.symlink_to(src)
        out = C.AB / "packs" / f"{name}.elkgrain"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            print(f"[packs] {out.name} exists, skipping build")
        else:
            print(f"\n=== building pack '{name}' ({len(files)} files) ===")
            build_corpus.build_pack(stage, out, unit=1, stride=1, silence_db=-50.0,
                                    margin_ms=10.0, device="auto", cache_dir=C.CACHE)
        packs[name] = str(out)
    # an existing pack (the paper's field recordings) rides along as a metrics-only corpus
    fp = CFG.FIELD_PACK
    if fp is not None and fp.exists():
        packs["field"] = str(fp)
        with C.load_pack(fp) as p:
            names = {f["name"] for f in p.meta["files"]}
            for qid, qpath, _ in CFG.QUERIES:
                assert qpath.name not in names, f"query {qid} is inside field pack!"
    C.jdump(packs, C.AB / "packs.json")


def pick_loudest_offset(path: Path, seconds: float) -> float:
    """Deterministic query offset: loudest `seconds` window on a 1 s grid (stems are sparse)."""
    g = C.granular()
    y = g.load_mono_44k(str(path), normalize=False)
    sr = g.CODEC_SR
    win = int(seconds * sr)
    if len(y) <= win + 2 * sr:
        return 0.0
    e = np.cumsum(y.astype(np.float64) ** 2)
    starts = np.arange(sr, len(y) - win - sr, sr)          # avoid first/last second
    energy = e[starts + win] - e[starts]
    return float(starts[np.argmax(energy)] / sr)


def make_queries():
    import soundfile as sf
    qdir = C.AB / "queries"
    qdir.mkdir(exist_ok=True)
    manifest = []
    for qid, src, character in CFG.QUERIES:
        wav = qdir / f"{qid}.wav"
        if not wav.exists():
            off = pick_loudest_offset(src, CFG.QUERY_SECONDS)
            y = C.granular().load_mono_44k(str(src), offset=off, seconds=CFG.QUERY_SECONDS)
            sf.write(wav, y, 44100, subtype="FLOAT")
            print(f"[queries] {qid}: {src.name} @ {off:.0f}s -> {wav.name}")
        else:
            off = None
        y44, lat = C.encode_cached(wav)                     # peak-normed like live input
        feats = F.frame_features(y44, lat.shape[1])
        np.savez(qdir / f"{qid}.npz", y44=y44, lat=lat, **feats)
        manifest.append({"id": qid, "source": str(src), "character": character,
                         "offset_s": off, "T": int(lat.shape[1])})
        print(f"[queries] {qid}: T={lat.shape[1]} frames")
    C.jdump(manifest, C.AB / "queries.json")


def corpus_feature_tables():
    packs = json.loads((C.AB / "packs.json").read_text())
    for name, path in packs.items():
        out = C.AB / "corpusfeat" / f"{name}.npz"
        if out.exists():
            print(f"[corpusfeat] {out.name} exists, skipping")
            continue
        with C.load_pack(path) as p:
            cdir = Path(p.meta["build"]["corpus_dir"])
            if name == "field" and CFG.FIELD_CORPUS_DIR is not None:
                cdir = CFG.FIELD_CORPUS_DIR     # the folder the field pack was built from may have moved
            assert cdir.exists(), f"corpus dir gone: {cdir}"
            print(f"[corpusfeat] {name}: {p.n} grains from {p.n_files} files …")
            tab = C.pack_pair_table(p, cdir, F.frame_features)
            C.save_npz(out, raw=np.asarray(p.raw), **tab)


def decode_musdb() -> list[Path]:
    mdir = C.AB / "musdb"
    mdir.mkdir(exist_ok=True)
    tracks = sorted((CFG.MUSDB / "train").glob("*.stem.mp4"))[:CFG.MUSDB_N_TRACKS] if CFG.MUSDB else []
    if not tracks:
        print("[train] MUSDB18 not found/empty — skipping (documented in memo)")
        return []
    outs = []
    for t in tracks:
        base = t.stem.replace(".stem", "").replace(" ", "_")[:40]
        for stem, idx in CFG.MUSDB_STEMS.items():
            wav = mdir / f"{base}__{stem}.wav"
            if not wav.exists():
                r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(t),
                                    "-map", f"0:{idx}", "-ac", "1", "-ar", "44100",
                                    "-c:a", "pcm_s16le", str(wav)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[train] ffmpeg failed on {t.name} stem {stem}: {r.stderr.strip()[:200]}")
                    continue
            outs.append(wav)
    print(f"[train] MUSDB: {len(outs)} stem clips from {len(tracks)} tracks")
    return outs


def build_training_pairs():
    """(mel80 -> raw latent frame) pairs from pack corpora + extras + MUSDB, grouped by file."""
    packs = json.loads((C.AB / "packs.json").read_text())
    X, Y, G, groups = [], [], [], []       # groups: {gid, name, kind}

    def add(x, y, name, kind):
        gid = len(groups)
        groups.append({"gid": gid, "name": name, "kind": kind, "frames": int(len(x))})
        X.append(np.asarray(x, np.float32)); Y.append(np.asarray(y, np.float32))
        G.append(np.full(len(x), gid, np.int32))

    for name, path in packs.items():
        cf = np.load(C.AB / "corpusfeat" / f"{name}.npz")
        mel, raw, fidx = cf["mel80"], cf["raw"], cf["file_idx"]
        with C.load_pack(path) as p:
            fnames = [f["name"] for f in p.meta["files"]]
        for fi in np.unique(fidx):
            m = fidx == fi
            add(mel[m], raw[m], f"pack:{name}/{fnames[fi]}", "pack")

    extra = list(CFG.TRAIN_EXTRA) + decode_musdb()
    for k, path in enumerate(extra):
        y44, lat = C.encode_cached(path)
        if lat.shape[1] < 4:
            continue
        mel = F.frame_features(y44, lat.shape[1])["mel80"]
        kind = "musdb" if path.parent.name == "musdb" else "extra"
        add(mel, lat.T, f"{kind}:{path.name}", kind)
        if k % 15 == 0:
            print(f"[train] extras {k+1}/{len(extra)} …")

    X = np.concatenate(X); Y = np.concatenate(Y); G = np.concatenate(G)
    # validation split: 15% of NON-pack groups (packs are the deploy corpus -> always train)
    rng = np.random.default_rng(CFG.SEED)
    nonpack = [g["gid"] for g in groups if g["kind"] != "pack"]
    val = set(rng.choice(nonpack, size=max(3, int(0.15 * len(nonpack))), replace=False).tolist())
    is_val = np.isin(G, list(val))
    for g in groups:
        g["split"] = "val" if g["gid"] in val else "train"
    C.save_npz(C.AB / "train" / "pairs.npz", X=X, Y=Y, G=G, is_val=is_val)
    C.jdump(groups, C.AB / "train" / "groups.json")
    print(f"[train] {len(X)} frames total; {is_val.sum()} val "
          f"({len(val)}/{len(groups)} groups) from {len(groups)} files")


if __name__ == "__main__":
    if not CFG.DATA_CONFIGURED:
        sys.exit(f"no data configuration at {CFG.CONFIG_PATH}: copy distillation/data.example.toml "
                 "to distillation/data.toml (or set LBD_DATA_CONFIG) and list your own recordings")
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["packs", "queries", "corpusfeat", "train"]
    for s in stages:
        {"packs": stage_and_build_packs, "queries": make_queries,
         "corpusfeat": corpus_feature_tables, "train": build_training_pairs}[s]()
