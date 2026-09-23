// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/SpscQueue.h (verbatim)
// Single-producer / single-consumer POD command queue.
//   producer = analysis worker (grain-play commands)
//   consumer = RT audio thread (drains at the top of processBlock)
// Fixed capacity, preallocated, lock-free, wait-free on both sides.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>

template <typename T, size_t CapacityPow2>
class SpscQueue
{
    static_assert (std::is_trivially_copyable_v<T>, "queue payload must be POD");
    static_assert ((CapacityPow2 & (CapacityPow2 - 1)) == 0, "capacity must be a power of two");

public:
    // Worker side. Returns false (and drops) when full.
    bool push (const T& item) noexcept
    {
        const uint64_t w = _write.load (std::memory_order_relaxed);
        const uint64_t r = _read.load (std::memory_order_acquire);
        if (w - r >= CapacityPow2)
            return false;
        _slots[static_cast<size_t> (w & _mask)] = item;
        _write.store (w + 1, std::memory_order_release);
        return true;
    }

    // RT side. Returns false when empty.
    bool pop (T& out) noexcept
    {
        const uint64_t r = _read.load (std::memory_order_relaxed);
        const uint64_t w = _write.load (std::memory_order_acquire);
        if (w == r)
            return false;
        out = _slots[static_cast<size_t> (r & _mask)];
        _read.store (r + 1, std::memory_order_release);
        return true;
    }

    void reset() noexcept
    {
        _read.store (0);
        _write.store (0);
    }

private:
    static constexpr uint64_t _mask = CapacityPow2 - 1;
    std::array<T, CapacityPow2> _slots {};
    alignas (64) std::atomic<uint64_t> _write { 0 };
    alignas (64) std::atomic<uint64_t> _read { 0 };
};
