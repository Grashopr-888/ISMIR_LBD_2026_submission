# Provenance: Grashopr-888/elk-mutek@3baef53:tools/corpus_builder/tests/test_roundtrip.py (changed: imports)
"""Pure format tests: write -> mmap-load -> bit-equal arrays, header sanity, determinism.
No ML involved — synthetic arrays only. This is what the C++ loader (T09) must also pass.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from corpus_builder import elkgrain


def make_arrays(seed=42, n=37, unit=2, latent_dim=64, audio_samples=100_000,
                hop48=4468, margin48=480):
    rng = np.random.default_rng(seed)
    gd = latent_dim * unit
    emb = rng.standard_normal((n, gd)).astype(np.float32)
    raw = rng.standard_normal((n, gd)).astype(np.float32)
    centre = rng.standard_normal(gd).astype(np.float32)
    audio = rng.integers(-32768, 32768, size=audio_samples, dtype=np.int16)
    total = unit * hop48 + 2 * margin48
    recs = np.zeros(n, dtype=elkgrain.GRAIN_REC_DTYPE)
    recs["audio_pos"] = rng.integers(0, audio_samples - total, size=n, dtype=np.uint64)
    recs["file_idx"] = rng.integers(0, 3, size=n, dtype=np.uint32)
    recs["frame_idx"] = np.arange(n, dtype=np.uint32)
    meta = {"format": "elkgrain", "note": "synthetic round-trip test", "files": ["a", "b", "c"]}
    return dict(emb=emb, raw=raw, centre=centre, audio=audio, grain_records=recs,
                unit=unit, hop48=hop48, margin48=margin48, n_files=3, meta=meta)


def test_roundtrip_bitequal(tmp_path):
    kw = make_arrays()
    p = tmp_path / "t.elkgrain"
    hdr = elkgrain.write_pack(p, **kw)

    with elkgrain.ElkGrainPack(p) as pk:
        # header fields
        assert pk.version == 1
        assert pk.n == kw["emb"].shape[0]
        assert pk.latent_dim == 64 and pk.unit == kw["unit"]
        assert pk.grain_dim == 64 * kw["unit"]
        assert pk.hop48 == kw["hop48"] and pk.margin48 == kw["margin48"]
        assert pk.grain_len48 == kw["unit"] * kw["hop48"]
        assert pk.total_len48 == pk.grain_len48 + 2 * pk.margin48
        assert pk.sample_rate == 48000 and pk.n_files == 3
        assert pk.flags & elkgrain.FLAG_HAS_RAW

        # every block 128-byte aligned (BLAS bit-parity contract from pack-format.md)
        for off in (pk.grains_off, pk.centre_off, pk.emb_off, pk.raw_off, pk.audio_off):
            assert off % 128 == 0, off

        # bit-equality of every block
        assert pk.emb.tobytes() == kw["emb"].tobytes()
        assert pk.raw.tobytes() == kw["raw"].tobytes()
        assert pk.centre.tobytes() == kw["centre"].tobytes()
        assert pk.audio.tobytes() == kw["audio"].tobytes()
        assert pk.grains.tobytes() == kw["grain_records"].tobytes()
        assert pk.meta == kw["meta"]

        # window/core views
        i = 5
        pos = int(kw["grain_records"]["audio_pos"][i])
        assert np.array_equal(pk.grain_window(i), kw["audio"][pos:pos + pk.total_len48])
        assert np.array_equal(pk.grain_core(i),
                              kw["audio"][pos + pk.margin48:pos + pk.margin48 + pk.grain_len48])

        # arrays are read-only views (no accidental mutation of the mapping)
        with pytest.raises(ValueError):
            pk.emb[0, 0] = 0.0


def test_write_is_deterministic(tmp_path):
    kw = make_arrays()
    p1, p2 = tmp_path / "a.elkgrain", tmp_path / "b.elkgrain"
    elkgrain.write_pack(p1, **kw)
    elkgrain.write_pack(p2, **kw)
    assert p1.read_bytes() == p2.read_bytes()


def test_header_is_128_bytes_and_le(tmp_path):
    assert struct.calcsize(elkgrain.HEADER_FMT) == 128
    kw = make_arrays()
    p = tmp_path / "t.elkgrain"
    elkgrain.write_pack(p, **kw)
    b = p.read_bytes()
    assert b[:8] == b"ELKGRAIN"
    assert struct.unpack_from("<I", b, 8)[0] == 1          # version, little-endian on disk
    assert struct.unpack_from("<I", b, 20)[0] == kw["emb"].shape[0]   # n_grains


def test_audio_overrun_rejected(tmp_path):
    kw = make_arrays()
    kw["grain_records"]["audio_pos"][0] = len(kw["audio"])  # window would run past the block
    with pytest.raises(ValueError, match="overruns"):
        elkgrain.write_pack(tmp_path / "bad.elkgrain", **kw)


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "x.elkgrain"
    p.write_bytes(b"NOTAPACK" + b"\x00" * 200)
    with pytest.raises(ValueError, match="magic"):
        elkgrain.ElkGrainPack(p)
