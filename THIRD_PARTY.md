# Third-party software and data

The author's code in this repository is MIT-licensed (`LICENSE`). The trained weights in `weights/` are CC BY-NC 4.0
(`weights/README.md`). Nothing below is committed to this repository: each item is downloaded by a package manager or
by CMake, runs on the board, or was used to produce the results and is not redistributed. Licences were read from each
project's licence file or package metadata on 23 Sep 2026.

## Downloaded when you build the environment or the plugin

| Component | Version / pin | Licence | How it arrives | Used for |
|---|---|---|---|---|
| [music2latent](https://github.com/SonyCSLParis/music2latent) (code and weights, Sony CSL Paris) | 0.1.8 | **CC BY-NC 4.0** | `uv sync` (PyPI). The weights (`music2latent.pt`) download from Hugging Face (`SonyCSLParis/music2latent`) on first use | the teacher: encoding corpora and queries, decoding in the desktop instrument |
| [JUCE](https://github.com/juce-framework/JUCE) | 8.0.8 (desktop, CMake FetchContent), 8.0.12 (board: Elk's SDK and `board/crossbuild`) | AGPLv3 or the commercial JUCE licence | CMake / the crossbuild image | the VST3 plugin, the parity CLI |
| Steinberg VST3 SDK (bundled in JUCE) | as bundled in JUCE | Steinberg VST3 licence or GPLv3 | with JUCE | the VST3 plugin format |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | commit `95c3c0f` (STL backend) | BSD-3-Clause | CMake FetchContent | the student's forward pass in C++ |
| nlohmann/json (vendored in RTNeural) | 3.11.1 | MIT | with RTNeural | reading the weights JSON in C++ |
| [elk-plugin-extensions](https://github.com/elk-audio/elk-plugin-extensions) (`elk_vst3_extensions.h`, © Elk Audio AB) | commit `23edab4` | **no licence file** in the upstream repository | CMake FetchContent (not redistributed here for that reason) | SUSHI's asynchronous-work extension for the worker thread |
| PyTorch | 2.13.0 | BSD-3-Clause with bundled third-party licences (package metadata) | `uv sync` | training, music2latent |
| NumPy | 2.5.2 | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | `uv sync` | everything |
| SciPy | 1.18.0 | BSD-3-Clause | `uv sync` | resampling, filters |
| librosa | 1.0.0 | ISC | `uv sync` | loading audio, the mel filterbank |
| soundfile | 0.14.0 | BSD-3-Clause | `uv sync` | reading and writing audio |
| huggingface_hub (music2latent's dependency) | 1.32.0 | Apache-2.0 | `uv sync` | downloading the music2latent weights |
| matplotlib | 3.11.2 | matplotlib licence (PSF-based) | `uv sync` | figures |
| JupyterLab, nbconvert, nbformat, ipykernel | 4.6.4, 7.17.1, 5.11.1, 7.3.0 | BSD-3-Clause | `uv sync` | the notebooks |
| pytest | 9.1.1 | MIT | `uv sync` (dev group) | tests |

`uv.lock` pins every other transitive Python dependency. Their licences are in each package's metadata in the
environment (`importlib.metadata.metadata(name)`).

**Consequences.** The music2latent licence (non-commercial) restricts any use of the pipeline that runs the teacher,
whatever the licence of this repository's code. A plugin binary built from `plugin/` links JUCE and the VST3 SDK, so
distributing such a binary requires AGPLv3/GPLv3 compliance (or commercial licences). SUSHI, which loads the plugin on
the board, is AGPLv3 with an exception for plugins whose source is available on request under terms that allow
commercial, royalty-free use, which the MIT licence meets. This repository distributes source only.

## On the board (not distributed)

| Component | Licence | Note |
|---|---|---|
| Elk Audio OS, SUSHI (1.3.0), SENSEI | SUSHI: AGPLv3 with a plugin exception (see its `LICENSE.md`) | shipped on the Elk Stomp Dev Kit image. `board/` only contains configuration and scripts for them |
| Elk's demo loop (`/home/mind/wavfiles/elG_08.wav`) | Elk's | the input of Experiment 2, read on the board, never copied here |

## Used to produce the results, not redistributed

| Data | Why it is not here |
|---|---|
| The author's recordings (the corpora, the 12 queries, the training extras) | audio. The rights of some field recordings are unconfirmed |
| [MUSDB18](https://sigsep.github.io/datasets/musdb.html) 7 s preview stems (75 stems, training only) | MUSDB18's terms (research, non-commercial). They are part of the students' training data, which is one reason `weights/` is CC BY-NC 4.0 |
| `.elkgrain` packs, corpus latent tables, query latents | packs contain audio, and music2latent latents can be decoded back to audio (an audio derivative) |
| Elk Audio's render of the board (Figure 1, panel C) | third-party image. `figures/fig_system.tex` draws a labelled frame instead unless you supply the file |
