# results/ — the numbers behind the paper (no audio, no latents)

## exp1/ — Experiment 1, retrieval fidelity (Table 1)

Regenerated on 23 Sep 2026 with this repository's code (`distillation/`) on the paper's data, which is not
redistributed. The 25 Aug 2026 run's intermediate outputs had not been kept, so the run was repeated:

- The three character corpora were rebuilt as packs from the author's recordings. Their music2latent latents came from
  the latent cache of the original run, so they are the 25 Aug latents (the 25 Aug packs themselves were not kept). The field-recordings pack
  (12,018 grains) is the original file, its source recordings checked by the SHA-1 stored in the pack.
- The students were restored from `weights/` (`python -m distillation.student restore`, exact, see notebook 02) and the
  ridge maps refitted (the pooled map is bit-identical to `weights/ridge_pooled.json`).
- `evaluate`, `supplement`, `int8_table eval` and `export_results` wrote the files below.

Every per-corpus number of the 25 Aug results memo was reproduced, and the 8-bit study reproduced the 16 Sep
`int8_vs_f32.json` exactly (all 288 rows equal). The same pipeline run with the original scripts (elk-mutek@3baef53)
gives identical files.

| File | Content |
|---|---|
| `metrics_per_query.csv` | 336 rows: 4 corpora × 12 queries × 7 query encoders. Columns: `top1`, `top5` (agreement of the best grain and overlap of the five best with the teacher's), `spearman` (ρ between orderings of all grains, mean over frames), `agree@τ` (selection agreement with the teacher at τ = 0, 0.03, 0.05, 0.15, same seed), `hub@live`, `uniq@live` (hub share and distinct grains per selection at τ = 0.05), `coverage`, `picked` (mean best cosine and mean cosine of the selected grain), `det_pctile`, `live_pctile` (percentile rank in the teacher's ordering of the encoder's best grain and of its selection at τ = 0.05), `det_regret`, `live_regret` (cosine similarity to the teacher's best grain lost by the same grains) |
| `embedding_error.csv` | 192 rows: 1 − mean-centred cosine between an encoder's output and the teacher's latent, all frames and split by onset (top-decile spectral flux) and by noisiness |
| `ridge_cv.json` | λ and leave-one-file-out score of each ridge fit |
| `corpora.json` | grains, files, minutes per corpus, query ids and frame counts (no file names) |
| `training_pairs.json` | the students' training set: 45,983 pairs from 114 files, by kind, and the 13-file validation split |
| `int8_vs_f32.json` | the 8-bit table study: per corpus × query × encoder (teacher `O`, student `B`) × scheme (`f32`, `row`, `dim`) |

Encoder codes: `O` teacher (music2latent) with the reference seed, `O2` the teacher with a second seed (its live
percentile computed by `export_results.py`, the other columns by `evaluate.py`), `A` ridge map fitted on the same
corpus (not in Table 1), `Apool` ridge map fitted on all four corpora (Table 1's "Ridge map"), `B` student MLP, `Bgru`
student GRU-128, `F` MFCC cosine. Query ids: `qp*` percussive, `qt*` tonal, `qx*` textural (4 each).

Table 1 is built from `metrics_per_query.csv` by `analysis/exp1.py` (notebook 03).

## board/ — Experiment 2, the Elk Stomp measurements (Figure 2, Results)

| File | Content |
|---|---|
| `raw/R1-gonza-f32.txt`, `raw/R1-gonza-int8.txt` | the 1,107-grain corpus (a 103 s field recording), 32-bit and 8-bit table |
| `raw/R1b-perc-f32.txt`, `raw/R1b-perc-int8.txt` | the percussive corpus, 5,115 grains |
| `raw/R1-field-f32.txt`, `raw/R1-field-int8.txt` | the field-recordings corpus, 12,018 grains |
| `raw/S6b-bench.txt` | microbenchmarks of the student and the search in isolation (`elkgrain_bench`) |
| `cmds/*.txt` | the command files each transcript ran (`board/serial/serial_console.py --cmds-file`) |
| `budget.json` | the per-stage medians copied by hand into the paper's figure script (notebook 05 checks them against the transcripts) |

The transcripts are byte-identical copies of the serial-console captures of 16–17 Sep 2026 (Grashopr-888/ISMIR_LBD_2026@2bf21d1:exp/board-session-2026-09-16/),
with the resampler-fixed build of the plugin. The board has no real-time clock, so its timestamps read 2025-05-29.
Not included: the console mirrors of the same runs and the session's setup transcripts (they contain local paths and
a login prompt), two invalid runs (`R1-perc-*`, no pack loaded because a transfer had failed, re-run as `R1b-perc-*`),
and the runs of the build before the resampler fix (superseded). `analysis/board_logs.py` parses the transcripts
(notebook 05).

Not reproducible from these files: the plugin's 0.093 % share of the audio callback on an M1 Max laptop (Results).
That `sushi --timing-statistics` file was not archived, and the value comes from the author's notes of 25 Aug 2026.
