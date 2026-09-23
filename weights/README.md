# weights/ — the paper's trained networks (CC BY-NC 4.0)

| File | What | Size | Provenance |
|---|---|---|---|
| `student_mlp_mel80.json` | the student: MLP, 80 log-mel bands, two hidden layers of 256 units, 64 outputs. 102,976 parameters, 102,400 multiply–accumulates per frame. **The network the board runs** | 2.2 MB | Grashopr-888/elk-mutek@3baef53:tools/match_ab/weights/student_mlp_mel80.json (byte-identical, trained 25 Aug 2026) |
| `student_gru128_mel80.json` | the GRU of Table 1: a 128-unit GRU and a dense output layer. 88,896 parameters, 88,064 multiply–accumulates per frame | 1.9 MB | Grashopr-888/elk-mutek@3baef53:tools/match_ab/weights/student_gru128_mel80.json (byte-identical) |
| `ridge_pooled.json` | the ridge map of Table 1: 80 log-mel bands and 9 spectral descriptors to 64 dimensions, fitted on all four corpora (λ = 100 by leave-one-file-out cross-validation). The board's fallback query encoder | 0.1 MB | Grashopr-888/elk-mutek@3baef53:work/elk-granular/packs/ridge_pooled.json (byte-identical) |

Format: RTNeural's JSON (Keras layout, kernel `[n_in][n_out]` then bias), with an input standardisation block
(`input.standardize.mean`, `.std`) that can be folded into the first layer. The GRU uses Keras's gate order (z, r, h,
`reset_after`). `distillation/student.py` writes them (`export_mlp_json`, `export_gru_json`) and reads them back
(`mlp_from_json`, `gru_from_json`). Re-exporting the restored networks reproduces these files byte for byte (notebook 02),
so they are exactly the networks behind the paper's numbers. The ridge map's `mu`, `sd`, `W`, `b` are bit-identical to a
refit with `distillation/ridgefit.py` on the paper's corpora (checked 23 Sep 2026).

## Training data

45,983 mel–latent pairs (`results/exp1/training_pairs.json`): the four evaluation corpora (25 files of the author's
recordings, always in training, the held-out queries never), 14 training-only recordings of the author's, and 75
MUSDB18 7 s preview stems (25 tracks × drums, other, vocals). The targets are music2latent latents. None of the audio is
distributed. Retrain on your own audio with `distillation/` (see its README).

## Licence

These weight files are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International** licence
(CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/): you may share and adapt them for non-commercial
purposes with attribution. Non-commercial because the networks were distilled from music2latent (CC BY-NC 4.0) and
trained partly on MUSDB18 previews, whose terms allow research use only.

Attribution: Trent Eriksen, "Neural Codec Latent Distillation for Corpus Retrieval Synthesis on the Elk Stomp",
ISMIR 2026 Late-Breaking Demo (see `CITATION.cff`).
