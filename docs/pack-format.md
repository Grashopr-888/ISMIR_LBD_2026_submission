<!-- Provenance: Grashopr-888/elk-mutek@3baef53:research/raw/pack-format.md (changed: file paths; task labels T05/T09 kept, they name the pack builder and the board engine) -->
# `.elkgrain` pack format — v1 (byte-exact spec)

**Status:** implemented + validated 2026-08-25 (T05). Writer/reader of record:
`corpus_builder/elkgrain.py`; matching reference: `corpus_builder/match.py`.
**This document alone must suffice to write the C++ loader (T09, `plugin/src/ElkgrainPack.cpp`).** If code and doc ever
disagree, fix whichever is wrong *and* the tests in `corpus_builder/tests/`.

A pack freezes one corpus "world" for the on-device engine: for every grain — its 48 kHz
int16 audio slice (with crossfade margins), its music2latent embedding (both the
mean-centred/L2 **matching table** row and the **raw latent**), and provenance
(source file, latent frame). One file, mmap-friendly, no parsing beyond the header + one
JSON block. Multiple worlds = multiple packs.

---

## 1. Global conventions

- **Byte order: little-endian** for every integer and float (ARMv7 and Apple Silicon are
  both LE; no byteswap path exists).
- Types: `u32`/`u64` unsigned ints; `f32` IEEE-754 binary32; `i16` two's-complement int16.
- **Every binary block starts at a file offset that is a multiple of 128 bytes.**
  Padding bytes between blocks are `0x00`. Do not relax this: it is not cosmetic —
  see §7 (BLAS bit-parity). The JSON block is exempt (it starts right after the header at
  offset 128, which happens to be aligned anyway).
- All block positions are given **explicitly in the header** as absolute byte offsets.
  A loader must use the offsets, never recompute them from block sizes.
- One pack = one corpus world; `unit` (latent frames per grain) is fixed per pack.

## 2. Header — fixed 128 bytes at offset 0

Python struct format: `"<8s14I8Q"` (8-byte magic, fourteen u32, eight u64 = 128 bytes).

| off | size | type | field | value / meaning |
|----:|-----:|------|-------|-----------------|
| 0   | 8    | char | `magic` | ASCII `"ELKGRAIN"` (45 4C 4B 47 52 41 49 4E) |
| 8   | 4    | u32  | `version` | `1` — reject anything else |
| 12  | 4    | u32  | `header_bytes` | `128` — reject anything else |
| 16  | 4    | u32  | `flags` | bit 0: raw-latents block present (always 1 in v1); bits 1–31 reserved, 0 |
| 20  | 4    | u32  | `n_grains` | N, number of grains |
| 24  | 4    | u32  | `latent_dim` | `64` (music2latent frame dim) |
| 28  | 4    | u32  | `unit` | latent frames per grain: 1, 2 or 4 (≈ 93/186/372 ms) |
| 32  | 4    | u32  | `grain_dim` | `latent_dim * unit` — row length of both f32 tables |
| 36  | 4    | u32  | `hop48` | nominal samples per latent frame at 48 kHz (see §5.1) |
| 40  | 4    | u32  | `margin48` | crossfade margin samples stored on EACH side of every grain |
| 44  | 4    | u32  | `grain_len48` | `unit * hop48` — core samples per grain |
| 48  | 4    | u32  | `total_len48` | `grain_len48 + 2*margin48` — stored samples per grain |
| 52  | 4    | u32  | `sample_rate` | `48000` (the board's fixed rate; audio is pre-resampled) |
| 56  | 4    | u32  | `n_files` | source files described in the JSON `files` array |
| 60  | 4    | u32  | `_pad0` | `0` |
| 64  | 8    | u64  | `json_off` | byte offset of JSON metadata (always 128 in v1) |
| 72  | 8    | u64  | `json_len` | JSON length in bytes (UTF-8, no NUL terminator) |
| 80  | 8    | u64  | `emb_off` | → matching table, `f32[N][grain_dim]` row-major |
| 88  | 8    | u64  | `raw_off` | → raw latents, `f32[N][grain_dim]` row-major (only if flags bit 0) |
| 96  | 8    | u64  | `centre_off` | → corpus centre, `f32[grain_dim]` |
| 104 | 8    | u64  | `audio_off` | → audio block, `i16[audio_samples]` mono 48 kHz |
| 112 | 8    | u64  | `audio_samples` | count of i16 samples in the audio block |
| 120 | 8    | u64  | `grains_off` | → grain table, N records of 16 bytes (§3) |

Worked example (test pack `demo_u1.elkgrain`: N=255, unit=1, 3 files, 24.1 s corpus):

```
00000000  45 4c 4b 47 52 41 49 4e  01000000 80000000   magic, version=1, header_bytes=128
00000010  01000000 ff000000 40000000 01000000          flags=1, n_grains=255, latent_dim=64, unit=1
00000020  40000000 a6110000 e0010000 a6110000          grain_dim=64, hop48=4518, margin48=480, grain_len48=4518
00000030  66150000 80bb0000 03000000 00000000          total_len48=5478, sample_rate=48000, n_files=3, pad
00000040  8000..0000000000 5804..0000000000            json_off=128, json_len=1112
00000050  0016..0000000000 0015 0100 00000000          emb_off=0x1600=5632, raw_off=0x11500=70912
00000060  0015..0000000000 0014 0200 00000000          centre_off=0x1500=5376, audio_off=0x21400=136192
00000070  409f 1100 00000000 0005..0000000000          audio_samples=0x119f40=1154880, grains_off=0x500=1280
```

(Block order **in the file** happens to be JSON → grain table → centre → embeddings → raw
→ audio, but never assume it — read the offsets.)

## 3. Grain table — `grains_off`, N × 16 bytes

Packed struct, no padding (verified `sizeof == 16`):

| off | type | field | meaning |
|----:|------|-------|---------|
| 0   | u64  | `audio_pos` | index **in samples** (not bytes) into the audio block where this grain's stored window begins |
| 8   | u32  | `file_idx` | index into the JSON `files` array (provenance) |
| 12  | u32  | `frame_idx` | first latent frame of this grain within its source file (provenance; grain covers frames `frame_idx … frame_idx+unit-1`) |

```c
#pragma pack(push,1)
struct GrainRec { uint64_t audio_pos; uint32_t file_idx; uint32_t frame_idx; };
#pragma pack(pop)
```

## 4. Audio block — deduplicated regions

`i16[audio_samples]`, mono, 48 kHz. Grain `i`'s stored window is exactly

```
audio[audio_pos_i .. audio_pos_i + total_len48)      // margin | core | margin
core  = audio[audio_pos_i + margin48 .. + margin48 + grain_len48)
```

- **Windows of neighbouring grains overlap inside shared regions** — the builder merges
  overlapping windows per source file into contiguous regions, so each source sample is
  stored once even with `unit=4, stride=1` (measured dedup ×4.05) and margins are free.
  A loader never needs to know about regions: `audio_pos` is all there is.
- The core is aligned to the latent-frame grid of the *source file* (§5.1); the margins are
  the neighbouring source audio, for equal-power crossfades between consecutively
  triggered grains. A margin that would extend past the start/end of the source file is
  **zero samples** (written into the pack; no runtime edge case).
- int16 encode was `clip(rint(x * 32767), -32768, 32767)` from float in [-0.9, 0.9]
  (per-file peak-normalised source). Decode on device as `x = s / 32768.0f`
  (the 32767/32768 = −0.00026 dB gain error is irrelevant; /32768 is a shift).

## 5. Embedding blocks and the matching math

`emb` (matching table) and `raw` are `f32[N][grain_dim]`, row-major, row `i` ↔ grain `i`
of the grain table. `centre` is `f32[grain_dim]`.

**Flattening order** (must match to be compatible with the tables): a grain covering latent
frames `f..f+unit-1` of a `[64][T]` latent array flattens **dim-major**:
`vec[d*unit + t] = lat[d][f+t]` (this is numpy C-order reshape of the `[64, unit]` slice —
NOT frame-major interleaving).

Construction (already applied by the builder; loader uses them as-is):
- `raw[i]` = flattened raw latent grain (f32).
- `centre` = mean of all `raw` rows.
- `emb[i]` = `(raw[i] - centre) / (||raw[i] - centre||_2 + 1e-8)` — mean-centred, L2-normalised.

**Query/selection** (the on-device loop at ~10.7 Hz; reference: `match.py`, which is
verified index-identical to the original instrument's `granular.substitute`):

```
q  = flatten(query frames) - centre;  q /= (||q||_2 + 1e-8)
sims[i] = dot(emb[i], q)                                  // cosine
sims[i] -= anti_hub * log1p(counts[i])                    // variety penalty (counts = picks so far this block)
if (prev >= 0 && prev+1 < N) sims[prev+1] += stick * 0.6  // continuity bonus (STICK_BONUS = 0.6)
dist = 1 - sims
if (threshold set && min(dist) > threshold)  passthrough (no pick, counts/prev unchanged)
if (temperature <= 1e-6) j = argmax(sims)
else  p = softmax(-dist / temperature); j = sample(p)     // the instrument's "Random" knob
counts[j] += 1; prev = j
```

All f32. Gain-matching happens in the audio domain on-device (RMS), not here.

### 5.1 The 48 kHz frame grid, `hop48`, and why `audio_pos` exists

music2latent's effective hop is `len(audio)/T` samples per frame — **it varies slightly per
file** (edge padding; e.g. 4517.6 @48k for an 8 s file, →≈4459 asymptotically for long
files). The builder therefore computes every grain's audio position **per file** as
`round(frame_idx * len48_file / T_file) - margin48` — stored in `audio_pos`, so there is
**zero cumulative drift** and the loader never does this math. `hop48` in the header is the
corpus-wide rounded average, used only to size `grain_len48`; the per-grain core may be
misaligned vs. the file's true grid by ≤ `unit/2 + 1` samples, absorbed by the margins.
Per-file effective hops are in JSON (`files[k].hop48_eff`) for tooling.

## 6. JSON metadata block

UTF-8 JSON object at `json_off`, `json_len` bytes, minified, no NUL terminator. The C++
loader may ignore everything except (optionally) `files[*].name`. Schema (v1):

```jsonc
{
  "format": "elkgrain", "format_version": 1,
  "tool": {"name": "corpus_builder", "version": "0.1.0", "music2latent": "0.1.8",
            "numpy": "2.5.2", "codec_device": "mps"},
  "created_utc": "2026-08-25T...",
  "build": {"corpus_dir": "...", "unit": 1, "stride": 1, "silence_db": -50.0,
             "margin_ms": 10.0, "normalize": true, "codec_sr": 44100,
             "resampler": "scipy.signal.resample_poly 160/147"},
  "files": [ {"i": 0, "name": "rel/path.wav", "sha1": "16-hex content hash",
              "seconds44": 8.0, "n_frames": 85, "len48": 384000, "hop48_eff": 4517.6471,
              "n_grains_kept": 85, "n_grains_dropped": 0}, ... ],
  "stats": {"n_grains": 255, "dropped_silent": 0, "corpus_seconds44": 24.0,
             "audio_samples": 1154880, "audio_seconds48": 24.06, "n_regions": 3,
             "dedup_ratio": 0.8267, "encode_s": 1.1}
}
```

Unknown keys must be ignored (forward compatibility inside v1).

## 7. Alignment is load-bearing (measured, 2026-08-25)

Apple **Accelerate's `sgemv` output depends on the matrix base-pointer alignment**: for the
same bits, pointers `≡ 0 (mod 128)` give one result; 4/16/32/64 mod 128 give results
differing by 1 ULP on ~40 % of rows (different summation order). Fresh large numpy/malloc
allocations are page-aligned, mmap bases are page-aligned (4 KB Linux / 16 KB macOS — both
multiples of 128), so **128-byte-aligned block offsets make mmap'd matching bit-identical
to in-memory matching** — that is what lets the parity suite assert *identical* softmax
draws. Consequences:
- writers MUST align all binary blocks to 128;
- laptop tools may mmap freely;
- the board (NEON/no Accelerate) is unaffected numerically, but keep the alignment anyway —
  it also guarantees natural alignment for every element type and cache-line-clean rows.
Cross-machine float32 dot products are still not guaranteed bit-identical between BLAS
implementations (Accelerate vs NEON C++); T09 validates the C++ matcher against `match.py`
at a tolerance for `sims`, exact for everything after (see T05 report).

## 8. Silence gate (what is *not* in the pack)

Grains whose per-frame audio (44.1 k domain) never exceeds `silence_db` (default −50 dBFS,
per-frame max over the grain's `unit` frames) were dropped at build time — identical logic
to the instrument's `build_codebook`, so pack contents bit-match the live engine's
codebooks. `frame_idx` therefore may skip values; the audio block contains no
grain-referenced silence.

## 9. Size model (v1, mono)

| component | bytes | 30 min corpus (unit=1, stride=1) |
|---|---|---|
| audio | ≈ `48000·2·seconds` (dense corpus) | 172.8 MB |
| matching table + raw | `2 · N · grain_dim · 4` (N ≈ 10.75/s · seconds/stride) | ≈ 9.9 MB |
| grain table | `16·N` | 0.3 MB |
| header+JSON | ~2 kB + ~0.2 kB/file | — |
| **total** | | **≈ 183 MB dense; the −50 dB gate must drop ≥ 13 % (typical for real material) to meet the <150 MB target** |

Measured on the 24 s test corpus: 2.4 MB (u1) / 2.8 MB (u4·s1, audio identical thanks to
region dedup) → 6.1 MB per corpus-minute. Levers if a pack is too big: silence gate,
`--stride`, shorter corpus; (v2 candidates: optional raw-latents omission via flags bit 0,
8-bit audio, per-region codec).

## 10. Minimal C++ loader sketch

```c
mmap file; verify magic "ELKGRAIN", version==1, header_bytes==128;
read u32/u64 fields at the offsets in §2 (memcpy from the mapping — all aligned);
emb    = (const float*)   (base + emb_off);      // N*grain_dim
raw    = (flags&1) ? (const float*)(base + raw_off) : NULL;
centre = (const float*)   (base + centre_off);
audio  = (const int16_t*) (base + audio_off);    // audio_samples
grains = (const GrainRec*)(base + grains_off);   // N
sanity: for all i: grains[i].audio_pos + total_len48 <= audio_samples;
        emb_off etc. % 128 == 0; file size covers every block.
```

## 11. Versioning

`version` is a hard gate: readers reject unknown versions. Any layout change (field, block,
alignment, flattening order, int16 scale) bumps it. New *optional* blocks within v1 are not
allowed (v1 is frozen by the T05 validation suite); use v2.
