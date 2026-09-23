// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/GoldenMain.cpp (changed: paths in comments)
// elkgrain_golden — offline validation CLI for T09 golden parity.
// Driven by plugin/tools/golden_parity.py,
// which compares every stage against the python references:
//   match.py (matcher), features.py/librosa (mel frontend), the T06 student
//   JSON (numpy forward), ridgefit.py (pooled map), scipy resample_poly.
//
// Modes:
//   header <pack>
//   fb <out.f32>                          mel filterbank [80][2049]
//   mel <win44.f32> <T> <out.f32>         T x 4096 raw 44.1k windows -> T x 89
//   resamp <in48.f32> <out.f32>           streaming 147/160 resampler
//   mlp <weights.json> <in.f32> <T> <out.f32>      T x 80 mel -> T x 64
//   ridge <ridge.json> <in.f32> <T> <out.f32>      T x 89 feat -> T x 64
//   match <pack> <q.f32> <T> <tau> <anti_hub> <stick> <seed> [simsN <sims.f32>]
//       q = T x grain_dim float32 raw-latent-domain queries; text results on
//       stdout (slot/pick/sim/best, then counts + prev), optionally the full
//       post-penalty sims rows for the first simsN slots.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ElkgrainPack.h"
#include "Matcher.h"
#include "Embedder.h"
#include "MelFrontend.h"

namespace
{
    std::vector<float> readF32 (const char* path)
    {
        FILE* f = std::fopen (path, "rb");
        if (f == nullptr)
        {
            std::fprintf (stderr, "cannot open %s\n", path);
            std::exit (2);
        }
        std::fseek (f, 0, SEEK_END);
        const long len = std::ftell (f);
        std::fseek (f, 0, SEEK_SET);
        std::vector<float> v (static_cast<size_t> (len) / 4);
        if (std::fread (v.data(), 4, v.size(), f) != v.size())
        {
            std::fprintf (stderr, "short read %s\n", path);
            std::exit (2);
        }
        std::fclose (f);
        return v;
    }

    void writeF32 (const char* path, const float* d, size_t n)
    {
        FILE* f = std::fopen (path, "wb");
        if (f == nullptr)
        {
            std::fprintf (stderr, "cannot write %s\n", path);
            std::exit (2);
        }
        std::fwrite (d, 4, n, f);
        std::fclose (f);
    }
}

int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf (stderr, "usage: elkgrain_golden <header|fb|mel|resamp|mlp|ridge|match> ...\n");
        return 2;
    }
    const std::string mode = argv[1];

    if (mode == "header" && argc >= 3)
    {
        ElkgrainPack pack;
        if (! pack.open (argv[2]))
        {
            std::fprintf (stderr, "OPEN FAILED: %s\n", pack.error().c_str());
            return 1;
        }
        std::printf ("ok n_grains=%u latent_dim=%u unit=%u grain_dim=%u hop48=%u margin48=%u "
                     "grain_len48=%u total_len48=%u sample_rate=%u audio_samples=%llu\n",
                     pack.nGrains(), pack.latentDim(), pack.unit(), pack.grainDim(),
                     pack.hop48(), pack.margin48(), pack.grainLen48(), pack.totalLen48(),
                     pack.sampleRate(), static_cast<unsigned long long> (pack.audioSamples()));
        // spot values for the python side to cross-check the views
        std::printf ("emb00=%.9g centre0=%.9g audio0=%d grain0_pos=%llu core_rms0=%.9g\n",
                     static_cast<double> (pack.emb()[0]), static_cast<double> (pack.centre()[0]),
                     static_cast<int> (pack.audio()[0]),
                     static_cast<unsigned long long> (pack.grains()[0].audio_pos),
                     static_cast<double> (pack.grainCoreRms()[0]));
        return 0;
    }

    if (mode == "fb" && argc >= 3)
    {
        std::vector<float> fb;
        MelFrontend::buildMelFilterbank (fb);
        writeF32 (argv[2], fb.data(), fb.size());
        return 0;
    }

    if (mode == "mel" && argc >= 5)
    {
        const auto in = readF32 (argv[2]);
        const int T = std::atoi (argv[3]);
        if (in.size() != static_cast<size_t> (T) * MelFrontend::kNFft)
        {
            std::fprintf (stderr, "mel: input size mismatch\n");
            return 2;
        }
        MelFrontend fe;
        fe.prepare (4480);
        std::vector<float> out (static_cast<size_t> (T) * MelFrontend::kFeatDim);
        for (int t = 0; t < T; ++t)
            fe.analyzeWindow44 (in.data() + static_cast<size_t> (t) * MelFrontend::kNFft,
                                out.data() + static_cast<size_t> (t) * MelFrontend::kFeatDim);
        writeF32 (argv[4], out.data(), out.size());
        return 0;
    }

    if (mode == "resamp" && argc >= 4)
    {
        const auto in = readF32 (argv[2]);
        MelFrontend fe;
        fe.prepare (4480);
        std::vector<float> out (in.size() + 16);
        const int n = fe.resampleOnly (in.data(), static_cast<int> (in.size()), out.data());
        writeF32 (argv[3], out.data(), static_cast<size_t> (n));
        return 0;
    }

    if ((mode == "mlp" || mode == "ridge") && argc >= 6)
    {
        const auto in = readF32 (argv[3]);
        const int T = std::atoi (argv[4]);
        const int inDim = mode == "mlp" ? Embedder::kMelDim : Embedder::kFeatDim;
        if (in.size() != static_cast<size_t> (T) * static_cast<size_t> (inDim))
        {
            std::fprintf (stderr, "%s: input size mismatch\n", mode.c_str());
            return 2;
        }
        Embedder emb;
        std::string err;
        const bool ok = mode == "mlp" ? emb.loadStudent (argv[2], err)
                                      : emb.loadRidge (argv[2], err);
        if (! ok)
        {
            std::fprintf (stderr, "LOAD FAILED: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> out (static_cast<size_t> (T) * Embedder::kLatentDim);
        std::vector<float> feat (Embedder::kFeatDim, 0.0f);
        for (int t = 0; t < T; ++t)
        {
            std::memcpy (feat.data(), in.data() + static_cast<size_t> (t) * inDim,
                         static_cast<size_t> (inDim) * sizeof (float));
            emb.embed (mode == "mlp" ? Embedder::Mode::Student : Embedder::Mode::Ridge,
                       feat.data(), out.data() + static_cast<size_t> (t) * Embedder::kLatentDim);
        }
        writeF32 (argv[5], out.data(), out.size());
        return 0;
    }

    if (mode == "match" && argc >= 9)
    {
        ElkgrainPack pack;
        if (! pack.open (argv[2]))
        {
            std::fprintf (stderr, "OPEN FAILED: %s\n", pack.error().c_str());
            return 1;
        }
        const auto q = readF32 (argv[3]);
        const int T = std::atoi (argv[4]);
        const float tau = static_cast<float> (std::atof (argv[5]));
        const float antiHub = static_cast<float> (std::atof (argv[6]));
        const float stick = static_cast<float> (std::atof (argv[7]));
        const auto seed = static_cast<uint32_t> (std::atol (argv[8]));
        const int dim = static_cast<int> (pack.grainDim());
        if (q.size() != static_cast<size_t> (T) * static_cast<size_t> (dim))
        {
            std::fprintf (stderr, "match: query size mismatch\n");
            return 2;
        }
        int simsN = 0;
        const char* simsPath = nullptr;
        if (argc >= 11)
        {
            simsN = std::atoi (argv[9]);
            simsPath = argv[10];
        }

        Matcher matcher;
        matcher.prepare (pack.emb(), pack.centre(), static_cast<int> (pack.nGrains()), dim);
        matcher.setSeed (seed);
        matcher.resetBar();

        std::vector<float> simsDump;
        for (int t = 0; t < T; ++t)
        {
            const auto r = matcher.selectOne (q.data() + static_cast<size_t> (t) * dim,
                                              tau, antiHub, stick);
            std::printf ("slot %d pick %d sim %.9g best %.9g\n", t, r.pick,
                         static_cast<double> (r.pickSim), static_cast<double> (r.bestSim));
            if (t < simsN)
                simsDump.insert (simsDump.end(), matcher.lastSims().begin(),
                                 matcher.lastSims().end());
        }
        // final counters (exactness check): nonzero counts + prev
        std::printf ("prev %d\n", matcher.prev());
        for (int i = 0; i < matcher.size(); ++i)
            if (matcher.counts()[static_cast<size_t> (i)] > 0.0f)
                std::printf ("count %d %.0f\n", i,
                             static_cast<double> (matcher.counts()[static_cast<size_t> (i)]));
        if (simsPath != nullptr && ! simsDump.empty())
            writeF32 (simsPath, simsDump.data(), simsDump.size());
        return 0;
    }

    std::fprintf (stderr, "bad arguments for mode '%s'\n", mode.c_str());
    return 2;
}
