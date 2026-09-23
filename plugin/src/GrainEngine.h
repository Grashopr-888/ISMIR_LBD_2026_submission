// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/GrainEngine.h (changed: paths in comments)
// Grain command schema + fixed grain-voice pool (T09: real pack grains).
//
// The RT thread activates commands popped from the SPSC queue and renders
// int16 pack-audio windows with the T06 renderer's envelope (prototype:
// the offline listening-test renderer, not included):
//     seg      = stored window [margin-half .. margin-half + core + margin)
//              = half a margin before the core, half after (len L + M)
//     envelope = equal-power sin quarter-wave fade-in over the first M
//                samples, 1 in the middle, cos fade-out over the last M
//     decode   = int16 / 32768
// Consecutive bar slots overlap in their fade regions -> equal-power
// crossfade drawn from real neighbouring source audio (the pack margins).
//
// No allocation, no locks, no syscalls anywhere in here. The pack audio and
// grain table are mmap'd read-only and fully pre-touched at load
// (ElkgrainPack::open), so RT reads take no first-touch page faults.
#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <type_traits>

#include "ElkgrainPack.h"

// ---------------------------------------------------------------------------
// THE SPSC COMMAND SCHEMA (worker -> RT). 32 bytes, POD, trivially copyable.
// T09 evolution of the T08 contract: grains are pack indices and onsets are
// ABSOLUTE sample times on the plugin's own sample clock (published by the RT
// thread), so scheduling is immune to worker/drain timing jitter. Wraparound
// is handled with signed 32-bit differences (fine for < ~12 h sessions).
// ---------------------------------------------------------------------------
struct GrainCommand
{
    uint32_t seq;          // monotonic command counter (debug/tracing)
    uint32_t grainIndex;   // index into the pack's grain table
    uint32_t startSample;  // absolute sample-clock target for the seg start
    uint32_t lengthFrames; // seg length = grain_len48 + margin48 (info/debug)
    float    gain;         // linear gain (incl. audio-domain RMS gain-match)
    float    pan;          // -1 (L) .. +1 (R), equal-power
    float    pitchRatio;   // reserved; 1.0 (native rate) in T09
    uint32_t flags;        // reserved
};
static_assert (std::is_trivially_copyable_v<GrainCommand>);
static_assert (sizeof (GrainCommand) == 32);

class GrainVoicePool
{
public:
    // A bar can be up to 8 s = ~86 hop slots, all activated (waiting on their
    // onset) as soon as the worker schedules the answer bar, plus the tail of
    // the previous bar. Waiting voices cost ~nothing to render.
    static constexpr int kNumVoices = 128;
    static constexpr uint32_t kMaxMargin = 4096;

    // Non-RT (pack load): build the equal-power fade tables for this pack's
    // margin and cache geometry. Mirrors render_clips.py:
    //   fade_in  = sin(0.5*pi*n/M), fade_out = cos(0.5*pi*n/M), n = 0..M-1
    void preparePack (const ElkgrainPack& pack) noexcept
    {
        _margin = pack.margin48() < kMaxMargin ? pack.margin48() : kMaxMargin;
        _core = pack.grainLen48();
        _segLen = _core + _margin;         // L + M (render_clips seg length)
        _half = _margin / 2;
        for (uint32_t i = 0; i < _margin; ++i)
        {
            const float ph = 0.5f * 3.14159265358979f * static_cast<float> (i)
                             / static_cast<float> (_margin);
            _fadeIn[i] = std::sin (ph);
            _fadeOut[i] = std::cos (ph);
        }
        for (auto& v : _voices)
            v.active = false;
        _dropped = 0;
        _stale = 0;
    }

    // World switch (T10): the RT side calls this on a flagged command. Same geometry
    // (our packs: margin 480, core 4480) -> only kill the voices; different -> full
    // re-prepare (2*margin trig calls; ~20 us for margin 480 on the A7, acceptable).
    void preparePackIfChanged (const ElkgrainPack& pack) noexcept
    {
        const uint32_t m = pack.margin48() < kMaxMargin ? pack.margin48() : kMaxMargin;
        if (m == _margin && pack.grainLen48() == _core && _segLen == _core + _margin)
        {
            killAll();
            return;
        }
        preparePack (pack);
    }

    uint32_t segLen() const noexcept { return _segLen; }
    uint32_t halfMargin() const noexcept { return _half; }

    // RT: activate a command. rtPos = absolute sample clock at the START of
    // this block. Late commands begin immediately (whole window, slightly
    // late — never mid-fade); commands entirely in the past are dropped.
    void activate (const GrainCommand& c, const ElkgrainPack* pack, uint64_t rtPos) noexcept
    {
        if (pack == nullptr || ! pack->isOpen() || c.grainIndex >= pack->nGrains()
            || _segLen < 2 * _margin || _margin < 2)
            return;

        const int32_t delta = static_cast<int32_t> (c.startSample - static_cast<uint32_t> (rtPos));
        if (delta < -static_cast<int32_t> (_segLen))
        {
            ++_stale;
            return;
        }

        for (auto& v : _voices)
        {
            if (! v.active)
            {
                v.active = true;
                // seg starts half a margin into the stored leading margin
                v.src = pack->grainWindow (c.grainIndex) + (_margin - _half);
                v.phase = 0;
                v.onset = delta > 0 ? static_cast<uint32_t> (delta) : 0;
                const float p = 0.25f * 3.14159265f * (c.pan + 1.0f); // 0..pi/2
                v.gainL = c.gain * std::cos (p);
                v.gainR = c.gain * std::sin (p);
                return;
            }
        }
        ++_dropped; // pool exhausted
    }

    // RT: additively render n frames of all active voices into outL/outR.
    void render (float* outL, float* outR, int n) noexcept
    {
        const uint32_t fadeOutStart = _segLen - _margin;
        for (auto& v : _voices)
        {
            if (! v.active)
                continue;

            int i = 0;
            if (v.onset > 0)
            {
                const uint32_t skip = v.onset < static_cast<uint32_t> (n)
                                          ? v.onset : static_cast<uint32_t> (n);
                v.onset -= skip;
                i = static_cast<int> (skip);
            }
            for (; i < n && v.phase < _segLen; ++i)
            {
                float w = 1.0f;
                if (v.phase < _margin)
                    w = _fadeIn[v.phase];
                else if (v.phase >= fadeOutStart)
                    w = _fadeOut[v.phase - fadeOutStart];
                const float s = static_cast<float> (v.src[v.phase]) * (1.0f / 32768.0f);
                outL[i] += s * w * v.gainL;
                outR[i] += s * w * v.gainR;
                ++v.phase;
            }
            if (v.phase >= _segLen)
                v.active = false;
        }
    }

    void killAll() noexcept
    {
        for (auto& v : _voices)
            v.active = false;
    }

    int activeCount() const noexcept
    {
        int c = 0;
        for (const auto& v : _voices)
            c += v.active ? 1 : 0;
        return c;
    }

    uint32_t dropped() const noexcept { return _dropped; }
    uint32_t stale() const noexcept { return _stale; }

private:
    struct Voice
    {
        bool active = false;
        const int16_t* src = nullptr; // seg base pointer inside the pack audio
        uint32_t phase = 0;           // frames rendered so far (0.._segLen)
        uint32_t onset = 0;           // frames until the seg starts
        float gainL = 0.0f, gainR = 0.0f;
    };

    std::array<Voice, kNumVoices> _voices {};
    std::array<float, kMaxMargin> _fadeIn {}, _fadeOut {};
    uint32_t _margin = 0, _core = 0, _segLen = 0, _half = 0;
    uint32_t _dropped = 0, _stale = 0;
};
