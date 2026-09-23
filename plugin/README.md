# plugin/ — the instrument as a VST3 plugin (JUCE 8, RTNeural)

| Source | Role (Section 3 of the paper) |
|---|---|
| `PluginProcessor.*` | the audio callback: writes input into a ring buffer, mixes scheduled grains, never waits for the analysis |
| `WorkerCore.*`, `ElkAsync.*` | the worker thread (a plain thread, or SUSHI's asynchronous work through Elk's extension): one analysis every 4,480 samples (93.3 ms), per-stage timers (the STAGES line), selection for the following bar |
| `MelFrontend.*` | the front-end: 48 to 44.1 kHz polyphase resampling, 4096-point FFT, 80 mel bands, spectral descriptors |
| `Embedder.*` | the student (RTNeural, STL backend) and the ridge map |
| `Matcher.*` | the search: mean-centred cosine to every grain (32-bit or 8-bit table), penalties, softmax sampling at τ |
| `GrainEngine.h` | grain playback with crossfades from the pack's 16-bit audio |
| `ElkgrainPack.*` | the memory-mapped `.elkgrain` reader (`docs/pack-format.md`) |
| `SpscQueue.h`, `SpscRing.h`, `RtGuard.h` | lock-free queues, allocation guard for the callback in debug builds |
| `GoldenMain.cpp` | `ElkgrainGolden`: a command-line tool that runs each C++ stage on given inputs, for the parity checks |
| `BenchMain.cpp` | `elkgrain_bench`: the microbenchmark of `results/board/raw/S6b-bench.txt` |

## Desktop build

```bash
cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin/build -j 8
```

CMake fetches JUCE 8.0.8, RTNeural (commit `95c3c0f`) and Elk's `elk_vst3_extensions.h` (commit `23edab4`).
`-DRTNEURAL_SOURCE_DIR=` and `-DELK_PLUGIN_EXTENSIONS_DIR=` take local checkouts instead, and `-DJUCE_SOURCE_DIR=` or
`-DELKGRANULAR_SYSTEM_JUCE=ON` an existing JUCE. The board build is in `board/crossbuild/`.

At run time the plugin reads `ELKGRANULAR_PACK`, `ELKGRANULAR_STUDENT` and `ELKGRANULAR_RIDGE` (defaults
`packs/field.elkgrain`, `packs/student_mlp_mel80.json`, `packs/ridge_pooled.json` relative to the host's working
directory), `ELKGRANULAR_TABLE=int8` for the 8-bit table, and `ELKGRANULAR_LOG` for the worker log.

## C++ = Python checks

```bash
GOLDEN_PACK=work/demo/demo.elkgrain uv run python plugin/tools/golden_parity.py   # matcher, student, ridge, filterbank, front-end, resampler
INT8_PACK=work/demo/demo.elkgrain uv run python plugin/tools/int8_parity.py       # the 8-bit table and its selections
```

On the paper's 12,018-grain field pack the matcher selected the same grain as `corpus_builder/match.py` on 200 of 200
frames (τ = 0), and the same holds on the demo pack of notebook 01 (checked 23 Sep 2026 on this tree). Sampling at
τ > 0 is compared at the level of the similarities, since C++ cannot reproduce NumPy's random stream.
