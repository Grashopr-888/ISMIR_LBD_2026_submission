# Neural Codec Latent Distillation for Corpus Retrieval Synthesis on the Elk Stomp

Code, measurements and notebooks for the ISMIR 2026 late-breaking demo by Trent Eriksen (Leiden University, Media
Technology).

Latent granular synthesis replaces each frame of a live input with a recorded grain selected by similarity in the
latent space of a neural audio codec. Here the codec is music2latent, which the Elk Stomp Dev Kit, a board without a
neural processing unit, cannot run. The paper asks whether a small query encoder distilled from the music2latent
encoder selects grains as that encoder would, and which part of the instrument then limits real-time operation on the
board. Experiment 1 compares five query encoders offline on held-out input (Table 1). Experiment 2 times each analysis
stage on the board (Figure 2).

This repository lets you read the code behind each result, run notebooks that show each step and reproduce the
paper's numbers from the data that can be shared, and rebuild the board plugin.

## Contents

| Path | What |
|---|---|
| `latent/granular.py` | the desktop instrument's core: music2latent encoding, grains, the matching table, selection by a softmax over mean-centred cosine similarities with temperature τ (with anti-repetition and continuity penalties), decoding |
| `corpus_builder/` | the offline pack builder (`build_corpus.py`), the `.elkgrain` format (`elkgrain.py`, specified in `docs/pack-format.md`), selection over a pack (`match.py`, identical selections to `granular.substitute`), tests |
| `distillation/` | Experiment 1: corpora, queries, mel–latent pairs, the student (MLP, GRU), the ridge maps, the metrics, the 8-bit table study (see its README) |
| `plugin/` | the instrument as a VST3 plugin for SUSHI (JUCE 8, RTNeural): audio callback, worker thread, front-end, student, search, grain scheduling, parity and benchmark tools |
| `board/` | cross-build image and toolchain, deployment over the serial console, the run script and SUSHI configurations for the Elk Stomp |
| `weights/` | the paper's trained student MLP (the network the board runs), the GRU and the pooled ridge map (CC BY-NC 4.0) |
| `results/` | numbers only: Experiment 1's per-query metrics, the board transcripts of Experiment 2 |
| `analysis/` | transcript parser, Table 1 builder, the generated demo signal, notebook helpers |
| `notebooks/` | five notebooks, committed with their outputs, each ending with a table of the paper's numbers it reproduces |
| `figures/` | Figure 1 (TikZ) and Figure 2 (matplotlib, drawn from the transcripts) |

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it). The first run downloads music2latent's
weights (CC BY-NC 4.0) from Hugging Face.

```bash
uv sync                                                                        # the environment, pinned by uv.lock
uv run jupyter lab notebooks/                                                  # read and run the notebooks
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb    # re-execute them all (about 3 min on a laptop CPU)
uv run pytest                                                                  # pack format and selection-parity tests
```

The notebooks run on a generated demo signal (`analysis/demo.py`) and on `results/`. Nothing they write
(`$LBD_WORK`, default `work/`) is part of the repository.

| Notebook | Shows | Reproduces |
|---|---|---|
| `01_corpus_packing` | encoding a corpus offline, grains, the matching table, writing and reading a pack, selection over the pack equal to the desktop instrument's | music2latent's 58 M parameters, 64-d latents per ≈93 ms frame, 16-bit pack audio, table sizes |
| `02_student_distillation` | mel–latent pairs, training the MLP under the paper's loss, parameters and MACs, RTNeural export checked against PyTorch, the paper's students round-tripping exactly | 102,976 parameters (103 k), 0.10 M and 0.09 M MACs, 45,983 training pairs |
| `03_retrieval_fidelity` | the four metrics on the demo pack, why the teacher agrees with itself on few selections at τ = 0.05 (Σ p²), Table 1 from `results/exp1` | Table 1, the abstract's ρ .84/.81/.58 and "top 4–7 %", onset error .295 against .262 |
| `04_int8_table` | quantising the latent table to 8 bits, orderings before and after | ρ = 1.000 between the student's 32- and 8-bit orderings |
| `05_board_measurements` | parsing the board transcripts, every run, stage medians, callback shares, microbenchmarks, Figure 2 | every on-board number in Results and Figure 2 |

## Paper to code

Every number the paper prints, where it comes from, and where it is reproduced. "Data" names the shipped file the
number is computed from.

| Paper | Value | Code | Data | Reproduced in |
|---|---|---|---|---|
| Abstract, Table 1: student cost | 0.10 M MACs per frame | `distillation/student.py` (`MLP`, `export_mlp_json`) | `weights/student_mlp_mel80.json` | 02 |
| Abstract, Table 1, Results: ρ student / ridge map / MFCC | .84 / .81 / .58 (.843 / .811 / .585) | `distillation/evaluate.py` (`spearman_rows`), `analysis/exp1.py` | `results/exp1/metrics_per_query.csv` | 03 |
| Abstract: student's selections in the top 4–7 % of the teacher's ordering | 4–7 % | `distillation/supplement.py` (`live_pctile`) | same | 03 |
| Abstract, Results: student about 1 % of the analysis period | 0.97–1.03 % | `plugin/src/WorkerCore.cpp` (STAGES timers), `analysis/board_logs.py` | `results/board/raw/R1*.txt` | 05 |
| Abstract, Figure 2: only the search grows with the corpus | front-end 3.93–3.98 ms, student 0.91–0.96 ms, search 1.29 to 12.73 ms | same | same | 05 |
| Introduction: 64-sample callback at 48 kHz | 1.333 ms | SUSHI `--timing-statistics` ("100 % = 1333.33 µs") | same | 05 |
| Introduction, Discussion: music2latent parameters | 58 M | `music2latent` (counted) | — | 01 |
| Section 3: 64-d latents, one per ≈93 ms frame | 64, ≈93 ms | `latent/granular.py` (`Codec`) | — | 01 |
| Section 3: a pack holds 16-bit audio and the corpus latent table | 16-bit | `corpus_builder/build_corpus.py`, `elkgrain.py` | — | 01 |
| Section 3: 80-band log-mel from one 4096-point FFT | 80, 4096 | `distillation/features.py`, `plugin/src/MelFrontend.cpp` | — | 02 |
| Section 3: two hidden layers of 256 units | 256 | `distillation/student.py` | `weights/student_mlp_mel80.json` | 02 |
| Section 3: 45,983 pairs from the author's recordings and MUSDB18 previews | 45,983 | `distillation/data_prep.py` (`train`) | `results/exp1/training_pairs.json` | 02 |
| Section 3: loss | MSE + 0.3 (1 − mean-centred cosine) | `distillation/student.py` (`composite_loss`) | — | 02 |
| Section 3: analysis every 4,480 samples | 93.3 ms | `plugin/src/WorkerCore.cpp` | `results/board/raw/R1*.txt` (worker header) | 05 |
| Section 4: 12 queries held out at file level, 4 corpora, same τ, seed and selection code | 12, 4, τ = 0.05 | `distillation/config.py` (assertions), `corpus_builder/match.py` | `results/exp1/corpora.json` | 03 |
| Section 4: board corpora | 1,107 (a 103 s field recording), 5,115, 12,018 grains | `corpus_builder/build_corpus.py` | `results/board/raw/R1*.txt` | 05 (the 103 s is in the pack's metadata, not shipped) |
| Table 1: MACs of the ridge map and the GRU | 5.7 k, 0.09 M | `distillation/ridgefit.py` (89 × 64), `student.py` (`export_gru_json`) | `weights/` | 02, 03 |
| Table 1: ρ, Pct., Agr., Hub of the five encoders | see Table 1 | `evaluate.py`, `supplement.py`, `export_results.py`, `analysis/exp1.py` | `results/exp1/metrics_per_query.csv` | 03 |
| Results: student and GRU ρ | .843 and .841 | same | same | 03 |
| Results: student leads the ridge map and MFCCs on every corpus | ρ per corpus | same | same | 03 |
| Results: student's percentile rank against the teacher's | .95 against .978 (logs: .979, see below) | same | same | 03 |
| Results: no run lost input or dropped an analysis | all loss counters 0 | `plugin/src/WorkerCore.cpp` (SUMMARY) | `results/board/raw/R1*.txt` | 05 |
| Results: callback share, average and largest | 1.95–2.39 %, up to 10.9 % | SUSHI `--timing-statistics` | same | 05 |
| Results: the same plugin on an M1 Max laptop | 0.093 % | SUSHI `--timing-statistics` | not shipped (the file was not archived, the value is from the author's notes) | — |
| Results: search the largest cost from 5,115 grains on with the 32-bit table | stage medians | `plugin/src/Matcher.cpp` | `results/board/raw/R1*.txt` | 05 |
| Results: 8-bit table cut the largest search to | 4.79 ms | `plugin/src/Matcher.cpp` (`ELKGRANULAR_TABLE=int8`) | same | 05 |
| Results: ρ between the student's 32- and 8-bit orderings | 1.000 | `distillation/int8_table.py` | `results/exp1/int8_vs_f32.json` | 04 |
| Figure 2: stage medians, 8-bit search 2.6 to 3.1 times faster | 2.59–3.08 | `figures/fig_budget.py` | `results/board/raw/R1*.txt` | 05 |
| Discussion: 103 k parameters on the board | 102,976 | `distillation/student.py` | `weights/student_mlp_mel80.json` | 02 |
| Discussion: onset frames the student's weakest | cosine distance .295 against .262 | `distillation/evaluate.py` (embedding error) | `results/exp1/embedding_error.csv` | 03 |
| Figure 1 | — | `figures/fig_system.tex` | — | — |

Checks behind the text that the paper does not print: C++ and Python select identical grains (200 of 200) and agree on
the student, ridge map, filterbank, front-end and resampler (`plugin/tools/golden_parity.py`), the C++ 8-bit table is
byte-identical to the NumPy one (`plugin/tools/int8_parity.py`), and pack selection equals the desktop instrument's
(`corpus_builder/tests/test_parity.py`).

### Where the logs and the printed paper differ

Table 1's teacher row prints a percentile rank of .978. The per-query logs give .97853 with the teacher's reference
seed, which rounds to **.979** (the .978 came from averaging per-corpus values already rounded to three decimals), and
.982 with the second seed that labels the row. Its hub share is .17 with either seed. Notebook 03 prints both values.
Every other number above equals the paper at its printed precision.

## Rebuilding the plugin

```bash
cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release && cmake --build plugin/build -j 8   # desktop: VST3, ElkgrainGolden, elkgrain_bench
board/crossbuild/build.sh                                                                       # Elk Stomp (Docker), see board/README.md
```

CMake fetches JUCE 8.0.8, RTNeural at `95c3c0f` and Elk's plugin-extension header at `23edab4`. Details in
`plugin/README.md` and `board/README.md`.

## What is not here, and why

- **Audio of any kind** (corpora, queries, MUSDB18 stems, rendered clips): MUSDB18's terms, and the rights of some of
  the author's field recordings are unconfirmed. The notebooks generate their own signal.
- **Packs and corpus latent tables**: packs contain audio, and music2latent latents decode back to audio.
- **music2latent's weights**: CC BY-NC 4.0, downloaded by the `music2latent` package on your machine.
- **JUCE, RTNeural, Elk's extension header**: fetched by CMake at pinned commits. Elk's header has no licence file upstream.
- **Elk Audio's image of the board** (Figure 1, panel C): third-party. `figures/fig_system.tex` draws a labelled frame.
- **The board's controls glue** (pots and switches over gRPC): adapted from an Elk example and not needed for the
  measurements.

## Licences

The author's code and documentation: MIT (`LICENSE`). `weights/`: CC BY-NC 4.0 (`weights/README.md`). Third-party
components and their licences: `THIRD_PARTY.md`. music2latent's non-commercial licence applies to any use of the
pipeline that runs it, whatever the licence of this code.

## Provenance

Every file copied from the author's working repositories carries a first-line comment
`Provenance: <repository>@<commit>:<path>` naming what was changed. Files that must stay byte-identical (transcripts,
JSON, weights) list their provenance in their folder's README. The work was done at the Music Hackspace × MUTEK
Montréal hackathon (22–23 Aug 2026) and in the four weeks after it.

## Citation

See `CITATION.cff`.

## AI usage

Anthropic Claude helped prepare this repository (porting the code, the notebooks, the documentation) and wrote parts
of the analysis scripts, under the author's direction and review.
