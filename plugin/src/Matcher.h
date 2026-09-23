// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/Matcher.h (changed: paths in comments)
// T09 grain selection — exact C++ port of corpus_builder/match.py
// (which is verified index-identical to the laptop instrument's
// granular.substitute selection). Worker thread only.
//
// Per slot (one target grain), in match.py's order and dtype discipline:
//   g = q_raw - centre;  g /= (||g||_2 + 1e-8)          float32
//   sims[i] = dot(emb[i], g)                            float32 (double acc)
//   if anti_hub > 0: sims[i] -= anti_hub * log1p(counts[i])
//   if stick > 0 and 0 <= prev+1 < N: sims[prev+1] += float(stick * 0.6)
//   dist = 1 - sims;  best = max(sims)
//   optional threshold gate: min(dist) > threshold -> passthrough
//                            (no pick, counts/prev unchanged)
//   tau <= 1e-6 -> argmax (FIRST max, like np.argmax)
//   else p = softmax(-dist/tau) sampled with this class's RNG
//   counts[j] += 1;  prev = j
//
// counts/prev reset per select "call" — in the plugin that is per BAR
// (matching the instrument, where substitute() runs once per block/bar).
//
// RNG: std::mt19937 (documented, seedable via the `seed` parameter),
// uniform double u in [0,1), inverse-CDF walk over p (numpy searchsorted
// side='right' semantics). numpy's Generator.choice stream CANNOT be
// reproduced in C++ — sampling draws differ BY DESIGN; T09's golden parity
// validates argmax selections + sims + penalty/counter math instead
// (match.py docstring says exactly this).
//
// int8 table (D1, ISMIR 2026; OFF unless ELKGRANULAR_TABLE=int8 at prepare()):
// the f32 table is quantised per row (symmetric, scale_i = max|row_i| / 127)
// into int8 + one float scale per grain; each query g is quantised the same way
// (scale_q); sims_i = (float(dot_i) * scale_q) * scale_i with dot_i an EXACT
// int32 sum (NEON vmull_s8 + vpadalq_s16 on ARM; scalar elsewhere). Everything
// after sims is unchanged. distillation/int8_table.py is the numpy replica
// (its `row` scheme, same float32 arithmetic) — the ordering study lives there,
// plugin/tools/int8_parity.py checks the two agree.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

class Matcher
{
public:
    // book: [n][dim] mean-centred+L2 rows (pack matching table, mmap view ok)
    // centre: [dim]. Both must outlive the Matcher. Non-RT.
    void prepare (const float* book, const float* centre, int n, int dim);

    void setSeed (uint32_t seed);       // reseeds the mt19937
    void resetBar() noexcept;           // counts -> 0, prev -> -1

    struct Result
    {
        int pick = -1;          // -1 = passthrough (threshold gate)
        float bestSim = 0.0f;   // max(sims) after penalties
        float pickSim = 0.0f;   // sims[pick]
    };

    // One slot. threshold < 0 disables the gate (match.py threshold=None).
    Result selectOne (const float* qRaw, float tau, float antiHub, float stick,
                      float threshold = -1.0f) noexcept;

    // Golden-harness access: sims AFTER penalties for the last selectOne call.
    const std::vector<float>& lastSims() const noexcept { return _sims; }
    const std::vector<float>& counts() const noexcept { return _counts; }
    int prev() const noexcept { return _prev; }
    int size() const noexcept { return _n; }

    // D1: which table the search streams, and how many bytes of it per query.
    bool int8Active() const noexcept { return _int8; }
    const char* tableName() const noexcept { return _int8 ? "int8" : "f32"; }
    size_t tableBytes() const noexcept;

private:
    const float* _book = nullptr;
    const float* _centre = nullptr;
    int _n = 0, _dim = 0;

    // int8 table (D1); empty unless ELKGRANULAR_TABLE=int8
    bool _int8 = false;
    std::vector<int8_t> _bq;   // [n][dim]
    std::vector<float> _bs;    // [n] per-row scales
    std::vector<int8_t> _gq;   // [dim] quantised query scratch

    std::vector<float> _counts;
    int _prev = -1;

    std::vector<float> _g, _sims, _p; // preallocated scratch
    std::mt19937 _rng { 0 };
};
