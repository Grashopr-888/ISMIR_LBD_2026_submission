// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/BenchMain.cpp (changed: paths in comments)
// elkgrain_bench — standalone microbench for the ISMIR 2026 board session (E3; E2b cross-check).
//
// JUCE-free on purpose: Embedder (RTNeural student), Matcher and ElkgrainPack use no JUCE, so
// unlike elkgrain_golden (which links juce_dsp for the mel front-end) this target is part of the
// cross build (board/crossbuild/build.sh) and runs on the Elk Stomp.
//
//   elkgrain_bench student <student_mlp_mel80.json> [N=10000]
//       Embedder::loadStudent; ONE fixed mel-80 frame (mean + std * u, u in [-1.5, 1.5) from a
//       fixed LCG — a realistic standardised input; the cost does not depend on the values);
//       100 warm-up calls; then N calls of Embedder::embed(Student), each timed with
//       clock_gettime(CLOCK_MONOTONIC). Prints p50/p99/mean/max/min µs per call, the bulk mean
//       (whole loop / N — bounds the timer overhead) and M MAC/s from 102 400 MACs per call
//       (80*256 + 256*256 + 256*64; the net has 102 976 parameters incl. 576 biases).
//   elkgrain_bench match <pack.elkgrain> [N=200] [tau=0.05]
//       ElkgrainPack::open (mmap + page pre-touch + core RMS) + Matcher::prepare, then N timed
//       Matcher::selectOne calls (anti_hub 0, stick 0, per-bar reset every 21 calls) with
//       queries taken from the pack's own raw latents. Same stats plus ms per 1000 grains and
//       the table size — the standalone counterpart of the plugin's se_* STAGES numbers.
//       ELKGRANULAR_TABLE=int8 in the environment switches Matcher to the int8 table (D1),
//       exactly as it does in the plugin; the output line says table=int8 table_mb=…
//
// Every output line starts with "BENCH " so a console transcript can be grepped.
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <json.hpp> // nlohmann, vendored inside RTNeural (modules/json)

#include "ElkgrainPack.h"
#include "Embedder.h"
#include "Matcher.h"

namespace
{
    double nowUs()
    {
        timespec ts {};
        clock_gettime (CLOCK_MONOTONIC, &ts);
        return static_cast<double> (ts.tv_sec) * 1e6 + static_cast<double> (ts.tv_nsec) / 1e3;
    }

    struct Stats { double p50, p99, mean, mx, mn; };

    Stats stats (std::vector<double> v) // by value: sorts a copy
    {
        std::sort (v.begin(), v.end());
        auto at = [&] (double p) { return v[static_cast<size_t> (p * static_cast<double> (v.size() - 1))]; };
        double sum = 0.0;
        for (double x : v)
            sum += x;
        return { at (0.50), at (0.99), sum / static_cast<double> (v.size()), v.back(), v.front() };
    }

    void printBuildInfo()
    {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const int neon = 1;
#else
        const int neon = 0;
#endif
#ifdef __OPTIMIZE__
        const int opt = 1;
#else
        const int opt = 0;
#endif
        std::printf ("BENCH build compiler=\"%s\" neon=%d optimized=%d pointer_bytes=%zu clock=CLOCK_MONOTONIC\n",
                     __VERSION__, neon, opt, sizeof (void*));
    }

    int usage()
    {
        std::fprintf (stderr,
                      "usage: elkgrain_bench student <student_mlp_mel80.json> [N=10000]\n"
                      "       elkgrain_bench match <pack.elkgrain> [N=200] [tau=0.05]\n");
        return 2;
    }
}

int main (int argc, char** argv)
{
    if (argc < 3)
        return usage();
    const std::string mode = argv[1];
    printBuildInfo();

    if (mode == "student")
    {
        const int n = argc >= 4 ? std::atoi (argv[3]) : 10000;
        if (n <= 0)
            return usage();

        Embedder emb;
        std::string err;
        if (! emb.loadStudent (argv[2], err))
        {
            std::fprintf (stderr, "LOAD FAILED: %s\n", err.c_str());
            return 1;
        }

        // the fixed frame: mean + std * u from the JSON's own standardisation block, so the net
        // sees an input of unit-normal scale (like a real log-mel frame after standardising)
        std::vector<float> frame (static_cast<size_t> (Embedder::kFeatDim), 0.0f);
        {
            std::ifstream f (argv[2]);
            nlohmann::json doc;
            f >> doc;
            const auto& st = doc.at ("input").at ("standardize");
            uint32_t s = 0x9E3779B9u; // fixed LCG: identical frame on every machine
            for (int i = 0; i < Embedder::kMelDim; ++i)
            {
                s = s * 1664525u + 1013904223u;
                const float u = (static_cast<float> (s >> 8) / 16777216.0f) * 3.0f - 1.5f;
                frame[static_cast<size_t> (i)] = st.at ("mean")[static_cast<size_t> (i)].get<float>()
                                                 + st.at ("std")[static_cast<size_t> (i)].get<float>() * u;
            }
        }

        float out[Embedder::kLatentDim];
        double checksum = 0.0; // keeps the forward passes observable (no dead-code elimination)
        for (int i = 0; i < 100; ++i)
        {
            emb.embed (Embedder::Mode::Student, frame.data(), out);
            checksum += out[i & 63];
        }
        std::vector<double> us;
        us.reserve (static_cast<size_t> (n));
        const double tBulk0 = nowUs();
        for (int i = 0; i < n; ++i)
        {
            const double t0 = nowUs();
            emb.embed (Embedder::Mode::Student, frame.data(), out);
            const double t1 = nowUs();
            us.push_back (t1 - t0);
            checksum += out[i & 63];
        }
        const double bulkUs = (nowUs() - tBulk0) / static_cast<double> (n);
        const Stats s = stats (us);
        constexpr double kMacs = 80.0 * 256.0 + 256.0 * 256.0 + 256.0 * 64.0; // 102 400
        std::printf ("BENCH student weights=%s n=%d warmup=100 macs_per_call=%.0f params=102976 checksum=%.6g\n",
                     argv[2], n, kMacs, checksum);
        std::printf ("BENCH student p50_us=%.2f p99_us=%.2f mean_us=%.2f max_us=%.2f min_us=%.2f bulk_us=%.2f mmac_s=%.1f\n",
                     s.p50, s.p99, s.mean, s.mx, s.mn, bulkUs, kMacs / s.p50);
        return 0;
    }

    if (mode == "match")
    {
        const int n = argc >= 4 ? std::atoi (argv[3]) : 200;
        const float tau = argc >= 5 ? static_cast<float> (std::atof (argv[4])) : 0.05f;
        if (n <= 0)
            return usage();

        ElkgrainPack pack;
        const double tOpen0 = nowUs();
        if (! pack.open (argv[2]))
        {
            std::fprintf (stderr, "OPEN FAILED: %s\n", pack.error().c_str());
            return 1;
        }
        const double openMs = (nowUs() - tOpen0) / 1e3;
        const int N = static_cast<int> (pack.nGrains());
        const int dim = static_cast<int> (pack.grainDim());
        const float* src = pack.raw() != nullptr ? pack.raw() : pack.emb();

        Matcher m;
        m.prepare (pack.emb(), pack.centre(), N, dim);
        m.setSeed (0);
        m.resetBar();
        // queries: pack rows (i * 7919) mod N — raw latents; centring/normalising happens in selectOne
        auto q = [&] (int i) {
            const uint64_t row = (static_cast<uint64_t> (i) * 7919u) % static_cast<uint64_t> (N);
            return src + static_cast<size_t> (row) * static_cast<size_t> (dim);
        };

        double checksum = 0.0;
        for (int i = 0; i < 20; ++i)
        {
            const auto r = m.selectOne (q (i), tau, 0.0f, 0.0f);
            checksum += r.pick;
        }
        std::vector<double> us;
        us.reserve (static_cast<size_t> (n));
        const double tBulk0 = nowUs();
        for (int i = 0; i < n; ++i)
        {
            if (i % 21 == 0)
                m.resetBar(); // the plugin resets per bar (a 2 s bar = 21 hops)
            const double t0 = nowUs();
            const auto r = m.selectOne (q (i), tau, 0.0f, 0.0f);
            const double t1 = nowUs();
            us.push_back (t1 - t0);
            checksum += static_cast<double> (r.pick) + static_cast<double> (r.bestSim);
        }
        const double bulkUs = (nowUs() - tBulk0) / static_cast<double> (n);
        const Stats s = stats (us);
        const double tableMb = static_cast<double> (m.tableBytes()) / 1e6;
        std::printf ("BENCH match pack=%s grains=%d dim=%d table=%s table_mb=%.3f open_ms=%.1f n=%d tau=%.3f warmup=20 checksum=%.6g\n",
                     argv[2], N, dim, m.tableName(), tableMb, openMs, n, static_cast<double> (tau), checksum);
        std::printf ("BENCH match p50_us=%.1f p99_us=%.1f mean_us=%.1f max_us=%.1f min_us=%.1f bulk_us=%.1f ms_per_1000_grains=%.4f\n",
                     s.p50, s.p99, s.mean, s.mx, s.mn, bulkUs, N > 0 ? s.p50 / 1e3 / (static_cast<double> (N) / 1000.0) : 0.0);
        return 0;
    }

    return usage();
}
