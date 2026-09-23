// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/ElkgrainPack.cpp (verbatim)
#include "ElkgrainPack.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
    // Little-endian field reads. Both targets (Apple Silicon, ARMv7) are LE
    // and every header offset is naturally aligned, but memcpy keeps it UB-free.
    uint32_t rdU32 (const uint8_t* p) { uint32_t v; std::memcpy (&v, p, 4); return v; }
    uint64_t rdU64 (const uint8_t* p) { uint64_t v; std::memcpy (&v, p, 8); return v; }
}

ElkgrainPack::~ElkgrainPack() { close(); }

void ElkgrainPack::close()
{
    if (_base != nullptr)
        ::munmap (const_cast<uint8_t*> (_base), _mapLen);
    _base = nullptr;
    _mapLen = 0;
    _emb = _raw = _centre = nullptr;
    _audio = nullptr;
    _grains = nullptr;
    _coreRms.clear();
    _nGrains = 0;
}

bool ElkgrainPack::open (const std::string& path)
{
    close();
    _path = path;
    _error.clear();

    auto fail = [this] (const std::string& why) {
        _error = _path + ": " + why;
        close();
        return false;
    };

    const int fd = ::open (path.c_str(), O_RDONLY);
    if (fd < 0)
        return fail ("cannot open");

    struct stat st {};
    if (::fstat (fd, &st) != 0 || st.st_size < 128)
    {
        ::close (fd);
        return fail ("stat failed or file < 128 bytes");
    }
    _mapLen = static_cast<size_t> (st.st_size);

    void* m = ::mmap (nullptr, _mapLen, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close (fd); // mapping keeps its own reference
    if (m == MAP_FAILED)
    {
        _mapLen = 0;
        return fail ("mmap failed");
    }
    _base = static_cast<const uint8_t*> (m);

    // ---- header (spec §2: "<8s14I8Q", 128 bytes) --------------------------
    if (std::memcmp (_base, "ELKGRAIN", 8) != 0)
        return fail ("bad magic");
    const uint32_t version = rdU32 (_base + 8);
    if (version != 1)
        return fail ("unsupported version " + std::to_string (version));
    if (rdU32 (_base + 12) != 128)
        return fail ("unexpected header_bytes");

    _flags       = rdU32 (_base + 16);
    _nGrains     = rdU32 (_base + 20);
    _latentDim   = rdU32 (_base + 24);
    _unit        = rdU32 (_base + 28);
    _grainDim    = rdU32 (_base + 32);
    _hop48       = rdU32 (_base + 36);
    _margin48    = rdU32 (_base + 40);
    _grainLen48  = rdU32 (_base + 44);
    _totalLen48  = rdU32 (_base + 48);
    _sampleRate  = rdU32 (_base + 52);
    _nFiles      = rdU32 (_base + 56);

    const uint64_t jsonOff   = rdU64 (_base + 64);
    const uint64_t jsonLen   = rdU64 (_base + 72);
    const uint64_t embOff    = rdU64 (_base + 80);
    const uint64_t rawOff    = rdU64 (_base + 88);
    const uint64_t centreOff = rdU64 (_base + 96);
    const uint64_t audioOff  = rdU64 (_base + 104);
    _audioSamples            = rdU64 (_base + 112);
    const uint64_t grainsOff = rdU64 (_base + 120);

    // ---- sanity per §10 + §1 ---------------------------------------------
    if (_grainDim != _latentDim * _unit)
        return fail ("grain_dim != latent_dim*unit");
    if (_grainLen48 != _unit * _hop48 || _totalLen48 != _grainLen48 + 2 * _margin48)
        return fail ("grain_len48/total_len48 inconsistent");
    if (_sampleRate != 48000)
        return fail ("sample_rate != 48000");
    if (jsonOff + jsonLen > _mapLen)
        return fail ("json block overruns file");

    const uint64_t embBytes    = uint64_t (_nGrains) * _grainDim * 4u;
    const uint64_t centreBytes = uint64_t (_grainDim) * 4u;
    const uint64_t audioBytes  = _audioSamples * 2u;
    const uint64_t grainsBytes = uint64_t (_nGrains) * 16u;

    struct Blk { const char* name; uint64_t off, len; bool present; };
    const Blk blocks[] = {
        { "emb",    embOff,    embBytes,    true },
        { "raw",    rawOff,    embBytes,    (_flags & 1u) != 0 },
        { "centre", centreOff, centreBytes, true },
        { "audio",  audioOff,  audioBytes,  true },
        { "grains", grainsOff, grainsBytes, true },
    };
    for (const auto& b : blocks)
    {
        if (! b.present)
            continue;
        if (b.off % 128u != 0)
            return fail (std::string (b.name) + " offset not 128-byte aligned (load-bearing, spec §7)");
        if (b.off + b.len > _mapLen)
            return fail (std::string (b.name) + " block overruns file");
    }

    // GrainCommand carries sample positions as u32 — enforce (board packs are
    // <=150 MB, so this can never fire on a valid target pack).
    if (_audioSamples > 0xFFFFFFFFull)
        return fail ("audio_samples exceeds u32 (unsupported by the grain engine)");

    _emb    = reinterpret_cast<const float*> (_base + embOff);
    _raw    = (_flags & 1u) != 0 ? reinterpret_cast<const float*> (_base + rawOff) : nullptr;
    _centre = reinterpret_cast<const float*> (_base + centreOff);
    _audio  = reinterpret_cast<const int16_t*> (_base + audioOff);
    _grains = reinterpret_cast<const ElkgrainGrainRec*> (_base + grainsOff);

    // grain windows must fit the audio block (spec §10 sanity)
    for (uint32_t i = 0; i < _nGrains; ++i)
        if (_grains[i].audio_pos + _totalLen48 > _audioSamples)
            return fail ("grain " + std::to_string (i) + " window overruns audio block");

    // ---- pre-touch every mapped page (off-RT; makes later RT reads of the
    // audio + grain table fault-free in practice) ---------------------------
    {
        volatile uint64_t sink = 0;
        const size_t page = 4096;
        for (size_t off = 0; off < _mapLen; off += page)
            sink += _base[off];
        sink += _base[_mapLen - 1];
        (void) sink;
    }

    // ---- per-grain core RMS (mirrors render_clips.py: float64 accumulate
    // over the core, /32768 decode, sqrt(mean + 1e-12)) ---------------------
    _coreRms.assign (_nGrains, 0.0f);
    for (uint32_t i = 0; i < _nGrains; ++i)
    {
        const int16_t* core = _audio + _grains[i].audio_pos + _margin48;
        double acc = 0.0;
        for (uint32_t s = 0; s < _grainLen48; ++s)
        {
            const double v = static_cast<double> (core[s]) / 32768.0;
            acc += v * v;
        }
        _coreRms[i] = static_cast<float> (
            std::sqrt (acc / static_cast<double> (_grainLen48) + 1e-12));
    }

    return true;
}
