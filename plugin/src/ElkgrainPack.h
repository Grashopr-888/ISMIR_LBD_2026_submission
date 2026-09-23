// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/ElkgrainPack.h (changed: spec path, paths in comments)
// .elkgrain v1 pack loader — mmap, validate, expose tables (T09).
//
// Byte-exact spec: docs/pack-format.md (§2 header, §3 grain table,
// §10 loader sketch). Writer of record: corpus_builder/elkgrain.py.
//
// Threading contract:
//   * open()/close() run OFF the RT thread (prepareToPlay / worker context).
//   * After open(), everything is read-only and lock-free. The RT thread may
//     read audio() and grains() — open() pre-touches every mapped page (and
//     precomputes per-grain core RMS, which touches all grain audio) so the
//     RT thread never takes a first-touch page fault.
//   * emb()/raw()/centre() are consumed by the worker (matching), never RT.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct ElkgrainGrainRec
{
    uint64_t audio_pos;  // SAMPLES (not bytes) into the audio block
    uint32_t file_idx;   // provenance: JSON files[] index
    uint32_t frame_idx;  // provenance: first latent frame in its source file
};
#pragma pack(pop)
static_assert (sizeof (ElkgrainGrainRec) == 16, "spec §3: 16-byte packed records");

class ElkgrainPack
{
public:
    ElkgrainPack() = default;
    ~ElkgrainPack();

    ElkgrainPack (const ElkgrainPack&) = delete;
    ElkgrainPack& operator= (const ElkgrainPack&) = delete;

    // Non-RT. mmap + full validation per pack-format.md §10. On failure the
    // pack stays closed and error() explains why. Pre-touches all pages and
    // computes per-grain core RMS (float, audio domain, /32768 decode).
    bool open (const std::string& path);
    void close();

    bool isOpen() const noexcept { return _base != nullptr; }
    const std::string& error() const noexcept { return _error; }
    const std::string& path() const noexcept { return _path; }

    // header fields (valid after open)
    uint32_t nGrains() const noexcept { return _nGrains; }
    uint32_t latentDim() const noexcept { return _latentDim; }
    uint32_t unit() const noexcept { return _unit; }
    uint32_t grainDim() const noexcept { return _grainDim; }
    uint32_t hop48() const noexcept { return _hop48; }
    uint32_t margin48() const noexcept { return _margin48; }
    uint32_t grainLen48() const noexcept { return _grainLen48; }
    uint32_t totalLen48() const noexcept { return _totalLen48; }
    uint32_t sampleRate() const noexcept { return _sampleRate; }
    uint64_t audioSamples() const noexcept { return _audioSamples; }

    // tables (zero-copy views into the mapping; 128-byte aligned per spec §7)
    const float* emb() const noexcept { return _emb; }        // [N][grain_dim]
    const float* raw() const noexcept { return _raw; }        // [N][grain_dim] or null
    const float* centre() const noexcept { return _centre; }  // [grain_dim]
    const int16_t* audio() const noexcept { return _audio; }  // [audio_samples]
    const ElkgrainGrainRec* grains() const noexcept { return _grains; } // [N]

    // Precomputed at open(): RMS of each grain's CORE samples (int16/32768
    // decode, double accumulation, sqrt(mean + 1e-12)) — mirrors the T06
    // renderer's rms_g so audio-domain gain-match matches render_clips.py.
    const float* grainCoreRms() const noexcept { return _coreRms.data(); }

    // Convenience: pointer to grain i's stored window (margin|core|margin,
    // totalLen48 samples). Bounds were validated at open(); RT-safe.
    const int16_t* grainWindow (uint32_t i) const noexcept
    {
        return _audio + _grains[i].audio_pos;
    }

private:
    const uint8_t* _base = nullptr;
    size_t _mapLen = 0;
    std::string _path, _error;

    uint32_t _nGrains = 0, _latentDim = 0, _unit = 0, _grainDim = 0;
    uint32_t _hop48 = 0, _margin48 = 0, _grainLen48 = 0, _totalLen48 = 0;
    uint32_t _sampleRate = 0, _flags = 0, _nFiles = 0;
    uint64_t _audioSamples = 0;

    const float* _emb = nullptr;
    const float* _raw = nullptr;
    const float* _centre = nullptr;
    const int16_t* _audio = nullptr;
    const ElkgrainGrainRec* _grains = nullptr;

    std::vector<float> _coreRms;
};
