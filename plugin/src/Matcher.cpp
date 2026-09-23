// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/Matcher.cpp (verbatim)
#include "Matcher.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace
{
    // D1: exact int32 dot product of two int8 vectors. |a*b| <= 127*127 fits int16, and
    // vpadalq_s16 widens each pair before accumulating, so nothing overflows for dim <= 64k.
    int32_t dot8 (const int8_t* a, const int8_t* b, int dim) noexcept
    {
        int32_t sum = 0;
        int d = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int32x4_t acc = vdupq_n_s32 (0);
        for (; d + 16 <= dim; d += 16)
        {
            const int8x16_t va = vld1q_s8 (a + d);
            const int8x16_t vb = vld1q_s8 (b + d);
            acc = vpadalq_s16 (acc, vmull_s8 (vget_low_s8 (va), vget_low_s8 (vb)));
            acc = vpadalq_s16 (acc, vmull_s8 (vget_high_s8 (va), vget_high_s8 (vb)));
        }
        sum = vgetq_lane_s32 (acc, 0) + vgetq_lane_s32 (acc, 1)
              + vgetq_lane_s32 (acc, 2) + vgetq_lane_s32 (acc, 3);
#endif
        for (; d < dim; ++d)
            sum += static_cast<int32_t> (a[d]) * static_cast<int32_t> (b[d]);
        return sum;
    }

    // symmetric quantisation of one vector: scale = max|v| / 127 (1 when all-zero), q = rint(v / scale)
    float quantise8 (const float* v, int dim, int8_t* out) noexcept
    {
        float maxAbs = 0.0f;
        for (int d = 0; d < dim; ++d)
            maxAbs = std::max (maxAbs, std::fabs (v[d]));
        const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
        for (int d = 0; d < dim; ++d)
        {
            const float q = std::nearbyint (v[d] / scale); // round-half-even, like numpy rint
            out[d] = static_cast<int8_t> (std::clamp (q, -127.0f, 127.0f));
        }
        return scale;
    }
}

void Matcher::prepare (const float* book, const float* centre, int n, int dim)
{
    _book = book;
    _centre = centre;
    _n = n;
    _dim = dim;
    _counts.assign (static_cast<size_t> (n), 0.0f);
    _g.assign (static_cast<size_t> (dim), 0.0f);
    _sims.assign (static_cast<size_t> (n), 0.0f);
    _p.assign (static_cast<size_t> (n), 0.0f);
    _prev = -1;

    // D1: int8 table, opt-in by environment (non-RT: prepare runs on the worker/prepare side)
    _int8 = false;
    _bq.clear();
    _bs.clear();
    _gq.clear();
    const char* tbl = std::getenv ("ELKGRANULAR_TABLE");
    if (tbl != nullptr && std::strcmp (tbl, "int8") == 0 && n > 0 && dim > 0)
    {
        _bq.resize (static_cast<size_t> (n) * static_cast<size_t> (dim));
        _bs.resize (static_cast<size_t> (n));
        _gq.assign (static_cast<size_t> (dim), 0);
        for (int i = 0; i < n; ++i)
            _bs[static_cast<size_t> (i)] = quantise8 (book + static_cast<size_t> (i) * static_cast<size_t> (dim), dim,
                                                      _bq.data() + static_cast<size_t> (i) * static_cast<size_t> (dim));
        _int8 = true;
    }
}

size_t Matcher::tableBytes() const noexcept
{
    return _int8 ? _bq.size() * sizeof (int8_t) + _bs.size() * sizeof (float)
                 : static_cast<size_t> (_n) * static_cast<size_t> (_dim) * sizeof (float);
}

void Matcher::setSeed (uint32_t seed)
{
    _rng.seed (seed);
}

void Matcher::resetBar() noexcept
{
    std::fill (_counts.begin(), _counts.end(), 0.0f);
    _prev = -1;
}

Matcher::Result Matcher::selectOne (const float* qRaw, float tau, float antiHub,
                                    float stick, float threshold) noexcept
{
    Result r;
    if (_n == 0)
        return r;

    // g = q - centre, L2-normalised (norm accumulated in double, applied f32
    // — numpy's float32 norm differs by <1 ulp-ish; golden tolerance covers)
    double nrmAcc = 0.0;
    for (int d = 0; d < _dim; ++d)
    {
        const float v = qRaw[d] - _centre[d];
        _g[static_cast<size_t> (d)] = v;
        nrmAcc += static_cast<double> (v) * static_cast<double> (v);
    }
    const float invn = 1.0f / (static_cast<float> (std::sqrt (nrmAcc)) + 1e-8f);
    for (int d = 0; d < _dim; ++d)
        _g[static_cast<size_t> (d)] *= invn;

    if (_int8)
    {
        // D1: quantised query, exact int32 dots, two float scales (same order as int8_table.py)
        const float sq = quantise8 (_g.data(), _dim, _gq.data());
        for (int i = 0; i < _n; ++i)
        {
            const int32_t dot = dot8 (_bq.data() + static_cast<size_t> (i) * static_cast<size_t> (_dim), _gq.data(), _dim);
            _sims[static_cast<size_t> (i)] = (static_cast<float> (dot) * sq) * _bs[static_cast<size_t> (i)];
        }
    }
    else
    {
        // sims = book @ g  (double accumulation per row, cast to float32 — see
        // pack-format.md §7: cross-BLAS f32 dots are not bit-identical; T09
        // validates at 1e-5 with exact argmax)
        for (int i = 0; i < _n; ++i)
        {
            const float* row = _book + static_cast<size_t> (i) * _dim;
            double acc = 0.0;
            for (int d = 0; d < _dim; ++d)
                acc += static_cast<double> (row[d]) * static_cast<double> (_g[static_cast<size_t> (d)]);
            _sims[static_cast<size_t> (i)] = static_cast<float> (acc);
        }
    }

    // variety penalty (anti_hub * log1p of per-bar pick counts)
    if (antiHub > 0.0f)
        for (int i = 0; i < _n; ++i)
            _sims[static_cast<size_t> (i)] -= antiHub * std::log1p (_counts[static_cast<size_t> (i)]);

    // stick continuity bonus on the grain FOLLOWING the previous pick.
    // NOTE: match.py's exact condition is `0 <= prev + 1 < N`, which is TRUE
    // for prev == -1 — the first slot of a bar gives grain 0 the bonus. A
    // quirk, but the parity contract says port it verbatim.
    if (stick > 0.0f && _prev + 1 >= 0 && _prev + 1 < _n)
        _sims[static_cast<size_t> (_prev + 1)] +=
            static_cast<float> (static_cast<double> (stick) * 0.6); // STICK_BONUS

    // best (after penalties, like match.py)
    int argmax = 0;
    float best = _sims[0];
    for (int i = 1; i < _n; ++i)
        if (_sims[static_cast<size_t> (i)] > best) // strict > keeps FIRST max (np.argmax)
        {
            best = _sims[static_cast<size_t> (i)];
            argmax = i;
        }
    r.bestSim = best;

    // optional distance-threshold passthrough (min(dist) > thr <=> best < 1-thr)
    if (threshold >= 0.0f && (1.0f - best) > threshold)
        return r; // pick stays -1; counts/prev untouched

    int j;
    if (tau <= 1e-6f)
    {
        j = argmax;
    }
    else
    {
        // p = softmax(-(1-sims)/tau), float32 like numpy, then inverse-CDF
        float maxLogit = -1e30f;
        for (int i = 0; i < _n; ++i)
        {
            const float logit = -(1.0f - _sims[static_cast<size_t> (i)]) / tau;
            _p[static_cast<size_t> (i)] = logit;
            if (logit > maxLogit)
                maxLogit = logit;
        }
        double sum = 0.0;
        for (int i = 0; i < _n; ++i)
        {
            const float e = std::exp (_p[static_cast<size_t> (i)] - maxLogit);
            _p[static_cast<size_t> (i)] = e;
            sum += static_cast<double> (e);
        }
        // draw: uniform in [0,1), first index with cumulative > u (numpy
        // rng.choice == searchsorted(cdf, u, side='right'))
        const double u = static_cast<double> (_rng()) * (1.0 / 4294967296.0);
        const double target = u * sum;
        double cum = 0.0;
        j = _n - 1;
        for (int i = 0; i < _n; ++i)
        {
            cum += static_cast<double> (_p[static_cast<size_t> (i)]);
            if (cum > target)
            {
                j = i;
                break;
            }
        }
    }

    _counts[static_cast<size_t> (j)] += 1.0f;
    _prev = j;
    r.pick = j;
    r.pickSim = _sims[static_cast<size_t> (j)];
    return r;
}
