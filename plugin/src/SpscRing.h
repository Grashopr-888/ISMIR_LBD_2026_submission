// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/SpscRing.h (verbatim)
// Single-producer / single-consumer mono sample ring.
//   producer = RT audio thread (processBlock writes a mono mixdown)
//   consumer = analysis worker (reads whole hops)
// Fixed capacity, preallocated, no locks, no allocation after construction.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>

template <size_t CapacityPow2>
class SpscMonoRing
{
    static_assert ((CapacityPow2 & (CapacityPow2 - 1)) == 0, "capacity must be a power of two");

public:
    // RT producer. Mixes ch0..(numCh-1) to mono and appends n frames.
    // If the ring is full the NEW samples are dropped (overrun counter bumps);
    // the read side is never touched by the producer.
    void writeMonoFrom (const float* const* channels, int numCh, int n) noexcept
    {
        const uint64_t w = _write.load (std::memory_order_relaxed);
        const uint64_t r = _read.load (std::memory_order_acquire);
        const uint64_t free = CapacityPow2 - (w - r);

        int toWrite = n;
        if (static_cast<uint64_t> (toWrite) > free)
        {
            toWrite = static_cast<int> (free);
            _overruns.fetch_add (1, std::memory_order_relaxed);
        }

        const float norm = numCh > 0 ? 1.0f / static_cast<float> (numCh) : 1.0f;
        for (int i = 0; i < toWrite; ++i)
        {
            float acc = 0.0f;
            for (int c = 0; c < numCh; ++c)
                acc += channels[c][i];
            _buf[static_cast<size_t> ((w + static_cast<uint64_t> (i)) & _mask)] = acc * norm;
        }
        _write.store (w + static_cast<uint64_t> (toWrite), std::memory_order_release);
    }

    // Worker consumer. Copies exactly n frames into dst if available.
    bool readBlock (float* dst, int n) noexcept
    {
        const uint64_t r = _read.load (std::memory_order_relaxed);
        const uint64_t w = _write.load (std::memory_order_acquire);
        if (w - r < static_cast<uint64_t> (n))
            return false;

        for (int i = 0; i < n; ++i)
            dst[i] = _buf[static_cast<size_t> ((r + static_cast<uint64_t> (i)) & _mask)];
        _read.store (r + static_cast<uint64_t> (n), std::memory_order_release);
        return true;
    }

    uint64_t framesAvailable() const noexcept
    {
        return _write.load (std::memory_order_acquire) - _read.load (std::memory_order_acquire);
    }

    uint64_t overruns() const noexcept { return _overruns.load (std::memory_order_relaxed); }

    void reset() noexcept
    {
        _read.store (0);
        _write.store (0);
        _overruns.store (0);
    }

private:
    static constexpr uint64_t _mask = CapacityPow2 - 1;
    std::array<float, CapacityPow2> _buf {};
    alignas (64) std::atomic<uint64_t> _write { 0 };
    alignas (64) std::atomic<uint64_t> _read { 0 };
    std::atomic<uint64_t> _overruns { 0 };
};
