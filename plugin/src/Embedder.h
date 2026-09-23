// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/Embedder.h (changed: paths in comments)
// T09 pluggable analysis->embedding stage (worker thread only).
//
// Modes (plugin parameter `embed_mode`):
//   0 student : T06's winning rung B — distilled MLP mel80 -> 256 -> 256 -> 64
//               via RTNeural (compile-time ModelT, STL backend), weights from
//               weights/student_mlp_mel80.json (RTNeural/keras
//               layout: kernel [n_in][n_out], bias [n_out], plus an input
//               standardization block: x = (mel80 - mean)/std).
//   1 ridge   : T06 rung A pooled ridge map feat89 -> 64
//               (y = ((x - mu)/sd) @ W[89][64] + b), from a JSON export of
//               ab/ridge.npz's pooled_{mu,sd,W,b}. Per-pack ridge overfits —
//               pooled only (T06 decision).
//   2 rawmel  : plumbing debug ONLY — first 64 log-mels passed through as the
//               "embedding". Not in latent space; matching runs but the picks
//               are meaningless. (Raw-feature matching was a FAIL in T06.)
//
// The JSON loader is deliberately tolerant: it reads only the keys it needs,
// ignores everything else (forward compatibility with T06 re-exports).
#pragma once

#include <array>
#include <memory>
#include <string>

class Embedder
{
public:
    enum class Mode { Student = 0, Ridge = 1, RawMel = 2 };
    static constexpr int kLatentDim = 64;
    static constexpr int kMelDim = 80;
    static constexpr int kFeatDim = 89;

    Embedder();
    ~Embedder();

    // Non-RT (prepareToPlay / worker prepare). Both may fail independently;
    // availability is queried before use and the worker falls back to RawMel.
    bool loadStudent (const std::string& jsonPath, std::string& errOut);
    bool loadRidge (const std::string& jsonPath, std::string& errOut);

    bool studentReady() const noexcept;
    bool ridgeReady() const noexcept;

    // feat89 in (frontend order: [mel80, extras9]), 64-d latent-domain
    // embedding out. Worker thread; no allocation.
    void embed (Mode m, const float* feat89, float* latent64) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
