# distillation/ — Experiment 1 on your own recordings

The code that produced Table 1: pack building for the corpora, held-out queries, the student and ridge baselines, and
the metrics. It runs on audio you supply. The paper's recordings are not redistributed (`results/exp1/` holds the
numbers they gave).

## Setup

Copy `data.example.toml` to `data.toml` (git-ignored) or point `LBD_DATA_CONFIG` at your own file, and list three
character corpora (5–15 min each), held-out query files (never in a corpus or in training), optional training-only
extras, an optional existing pack as a fourth metrics-only corpus, and optionally MUSDB18 7 s previews (decoded with
`ffmpeg`). Intermediates (packs, latents, features) go to `$LBD_WORK/ab` (default `work/ab`, git-ignored).

## Pipeline (from the repository root)

| Step | Command | Writes |
|---|---|---|
| packs | `uv run python -m distillation.data_prep packs` | `$LBD_WORK/ab/packs/*.elkgrain`, `packs.json` |
| queries | `uv run python -m distillation.data_prep queries` | 8 s clips (loudest window), latents, features |
| corpus features | `uv run python -m distillation.data_prep corpusfeat` | per-grain features aligned with each pack |
| training pairs | `uv run python -m distillation.data_prep train` | `train/pairs.npz`, `groups.json` (file-level validation split) |
| ridge maps | `uv run python -m distillation.ridgefit` | `ridge.npz` (per corpus and pooled, λ by leave-one-file-out CV) |
| student | `uv run python -m distillation.student` (MLP), `... student gru` | checkpoints and RTNeural JSON in `$LBD_WORK/weights` |
| or the paper's students | `uv run python -m distillation.student restore` | checkpoints rebuilt exactly from `weights/` |
| metrics | `uv run python -m distillation.evaluate`, `uv run python -m distillation.supplement` | `out/results.json`, `out/supplement.json` |
| 8-bit table | `uv run python -m distillation.int8_table eval --queries $LBD_WORK/ab/queries --packs perc=... tonal=... text=... --out ...` | `int8_vs_f32.json` |
| export | `uv run python -m distillation.export_results --int8 <int8 json> --out <dir>` | per-query CSV/JSON, numbers only |

| Module | Role |
|---|---|
| `config.py` | the data designation (from TOML) and the hold-out assertions, τ = 0.05 and the seed |
| `ab_common.py` | cached music2latent encoding, pack loading, per-grain feature tables |
| `features.py` | per-frame features on the codec's frame grid: 80 log-mel bands (20 Hz–16 kHz, 4096-point FFT), 9 spectral descriptors, 13 MFCCs |
| `data_prep.py` | corpora, queries, training pairs |
| `ridgefit.py` | ridge maps (features to 64-d latent) |
| `student.py` | MLP and GRU students, loss MSE + 0.3 (1 − mean-centred cosine), RTNeural export and its inverse |
| `evaluate.py` | ρ, top-1/top-5, selection agreement over a τ sweep with matched seeds, hub share, embedding error |
| `supplement.py` | percentile rank of each encoder's best and selected grain in the teacher's ordering |
| `int8_table.py` | 8-bit latent tables (per-grain or per-dimension scale) against the 32-bit table |
| `export_results.py` | the per-query numbers in `results/exp1/` |

All selection goes through `corpus_builder/match.py`, which is `latent/granular.py`'s selection (tested for identical
selections in `corpus_builder/tests/`). MPS encoding is not bit-reproducible across runs. The latent cache freezes each
file's latents at first encode, and every later step is deterministic.
