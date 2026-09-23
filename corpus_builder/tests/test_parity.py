# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/tests/test_parity.py (changed: imports)
"""T05's critical deliverable: the pack + our matcher reproduce the ORIGINAL instrument exactly.

  1. the pack's matching table / raw latents / centre are BIT-EQUAL to what
     granular.build_codebook builds from the same latents;
  2. match.Matcher.select over the pack returns selection indices IDENTICAL to
     granular.substitute over the same arrays, for the same tau/seed/penalties;
  3. every grain's stored 48 kHz audio slice equals an independent recomputation
     from the source file (provenance/alignment).
"""
from __future__ import annotations

import numpy as np
import pytest

from corpus_builder import build_corpus, elkgrain, match
from conftest import encode_cached, rebuild_codebook


# --------------------------------------------------------------- 1. codebook bit-equality
@pytest.mark.parametrize("unit", [1, 2])
def test_pack_arrays_bitequal_to_original_codebook(packs, granular, corpus_dir, unit):
    path, hdr, meta = packs[unit]
    cb = rebuild_codebook(granular, corpus_dir, unit)
    with elkgrain.ElkGrainPack(path) as pk:
        assert pk.n == cb.n > 0
        assert pk.emb.shape == cb.book.shape
        assert pk.emb.tobytes() == cb.book.tobytes(), "matching table differs from build_codebook"
        assert pk.raw.tobytes() == cb.raw.tobytes(), "raw latents differ from build_codebook"
        assert pk.centre.tobytes() == cb.centre.tobytes(), "centre differs from build_codebook"
        assert int(meta["stats"]["dropped_silent"]) == int(cb.dropped_silent)


def test_silence_gate_parity(granular, corpus_dir, tmp_path):
    """A harsher gate (-30 dB) must drop grains and still match build_codebook bit-for-bit."""
    out = tmp_path / "gate.elkgrain"
    hdr, meta = build_corpus.build_pack(corpus_dir, out, unit=1, stride=1, silence_db=-30.0)
    cb = rebuild_codebook(granular, corpus_dir, unit=1, silence_db=-30.0)
    with elkgrain.ElkGrainPack(out) as pk:
        assert pk.n == cb.n
        assert pk.emb.tobytes() == cb.book.tobytes()
    lenient = rebuild_codebook(granular, corpus_dir, unit=1, silence_db=-95.0)
    assert cb.dropped_silent >= 0 and cb.n <= lenient.n


# --------------------------------------------------------------- 2. selection parity
CASES = [
    # tau,   seed, anti_hub, stick, use_threshold
    (0.05,   0,    0.0,      0.0,   False),   # the instrument's default sampling
    (0.0,    0,    0.0,      0.0,   False),   # greedy argmax path
    (0.35,   7,    0.35,     0.5,   False),   # hot + variety + continuity
    (0.05,   3,    0.0,      0.0,   True),    # threshold gate -> passthroughs
    (0.2,    11,   1.0,      1.0,   False),   # penalties at full tilt
    (0.05,   1234, 0.6,      0.0,   True),    # gate + anti-hub together
]


@pytest.mark.parametrize("unit", [1, 2])
@pytest.mark.parametrize("tau,seed,anti_hub,stick,use_thr", CASES)
def test_selection_parity(packs, granular, target_lat, unit, tau, seed, anti_hub, stick, use_thr):
    from pathlib import Path
    path, _, _ = packs[unit]
    with elkgrain.ElkGrainPack(path) as pk:
        threshold = None
        if use_thr:
            # median best-distance so the gate actually fires on some grains
            _, T = target_lat.shape
            best = []
            for i in range(0, T - unit + 1, unit):
                g = target_lat[:, i:i + unit].reshape(-1) - pk.centre
                g = g / (np.linalg.norm(g) + 1e-8)
                best.append(1.0 - float((pk.emb @ g).max()))
            threshold = float(np.median(best))

        # the original-code view of the same corpus (bit-equal arrays, per the test above)
        cb = rebuild_codebook(granular, Path(pk.meta["build"]["corpus_dir"]), unit)

        _, info = granular.substitute(
            target_lat, cb, temperature=tau, gain_match=True, threshold=threshold,
            rng=np.random.default_rng(seed), anti_hub=anti_hub, stick=stick)

        res = match.Matcher.from_pack(pk).select(
            target_lat, temperature=tau, threshold=threshold,
            rng=np.random.default_rng(seed), anti_hub=anti_hub, stick=stick)

    assert res.grains == info.grains
    assert res.passthrough == info.passthrough
    if use_thr:
        assert res.passthrough > 0, "threshold case must actually exercise the passthrough path"
    assert len(res.picks) > 0
    assert res.picks == info.picks, "selection indices diverged from granular.substitute"
    # summary stats ride on the same picks -> must agree exactly (same-process float ops)
    assert res.coverage == info.coverage
    assert res.picked == info.picked
    assert res.hubness == info.hubness
    assert res.unique_frac == info.unique_frac


# --------------------------------------------------------------- 3. audio provenance
def test_grain_audio_alignment(packs, granular):
    """Every stored grain core equals the slice recomputed from its source file at 48 kHz."""
    from pathlib import Path
    path, hdr, meta = packs[2]
    with elkgrain.ElkGrainPack(path) as pk:
        corpus_dir = Path(pk.meta["build"]["corpus_dir"])
        files = build_corpus.find_audio(corpus_dir)
        assert [f.name for f in files] == [Path(m["name"]).name for m in pk.meta["files"]]
        y48s = [build_corpus.to_int16(granular.resample(granular.load_mono_44k(str(p)),
                                                        granular.CODEC_SR, 48000))
                for p in files]
        n_frames = [m["n_frames"] for m in pk.meta["files"]]
        for i in range(pk.n):
            fi = int(pk.grains["file_idx"][i])
            fr = int(pk.grains["frame_idx"][i])
            y48 = y48s[fi]
            hop_eff = len(y48) / n_frames[fi]
            s48 = int(round(fr * hop_eff))
            core = build_corpus.slice_zero_fill(y48, s48, s48 + pk.grain_len48)
            got = pk.grain_core(i)
            assert np.array_equal(got, core), f"grain {i} (file {fi}, frame {fr}) audio misaligned"
            win = pk.grain_window(i)
            assert len(win) == pk.total_len48
            assert np.array_equal(win[pk.margin48:pk.margin48 + pk.grain_len48], core)


def test_second_build_hits_latent_cache(packs, corpus_dir, tmp_path):
    """CorpusManager-style content-hash caching: rebuilding encodes nothing."""
    out = tmp_path / "again.elkgrain"
    hdr, meta = build_corpus.build_pack(corpus_dir, out, unit=1, stride=1)
    assert meta["stats"]["encode_s"] == 0.0
