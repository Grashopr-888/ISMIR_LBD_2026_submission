// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/WorkerCore.h (changed: comment)
// The ~10.7 Hz analysis worker — T09: the real bar-loop instrument.
//
// Driven by one of three mechanisms (ELKGRANULAR_WORKER_MODE):
//   OwnThread   : plugin std::thread parked on a CV the RT thread signals at
//                 each hop boundary (desktop/DAW fallback; board would swap
//                 the CV for twine::RtConditionVariable).
//   ElkAsync    : sushi's host-managed requestAsyncWork worker (the chosen
//                 board pattern — no syscall from the RT domain).
//   Synchronous : drainPendingHops() runs inline on the audio thread
//                 (OFFLINE BOUNCES ONLY — deterministic renders; sushi -o
//                 runs faster than realtime, which would starve an async
//                 worker. Never use in a realtime session.)
//
// The instrument loop (T09):
//   * every hop (4480 frames @48k = 93.33 ms): read from the SPSC ring ->
//     input RMS/gate -> MelFrontend (44.1k mel80+extras on the training
//     grid) -> Embedder (student MLP / ridge / rawmel) -> stash per-slot.
//   * bars are whole hops: barHops = round(bar_seconds*sr/hop), sample-
//     counted internally — NEVER VST3 processContext transport (broken
//     under sushi). bar_seconds is read at each bar boundary.
//   * at each completed bar: Matcher (exact match.py port) picks one pack
//     grain per non-gated slot; commands are scheduled for the following bar at
//     absolute sample times (slot cores land on the next bar's hop grid,
//     each seg starting half a margin early for the equal-power crossfade).
//   * hold (SW1): while engaged, the previous ANSWER bar is re-emitted
//     verbatim each bar (no matching, no gate) — the cheap version of the
//     instrument's Hold. Input keeps being consumed and analysed.
//
// All file I/O, timestamps and mmap-page-touching stay on the worker side.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "SpscRing.h"
#include "SpscQueue.h"
#include "GrainEngine.h"
#include "ElkgrainPack.h"
#include "MelFrontend.h"
#include "Embedder.h"
#include "Matcher.h"

struct WorkerParams
{
    const std::atomic<float>* wet = nullptr;        // logging only
    const std::atomic<float>* tau = nullptr;        // 0..0.30
    const std::atomic<float>* gateDb = nullptr;     // -80..-20 dBFS
    const std::atomic<float>* barSeconds = nullptr; // 0.5..8
    const std::atomic<float>* stick = nullptr;      // 0..1
    const std::atomic<float>* variety = nullptr;    // 0..1 -> anti_hub = 0.25*variety
    const std::atomic<float>* gainMatch = nullptr;  // bool
    const std::atomic<float>* embedMode = nullptr;  // 0 student / 1 ridge / 2 rawmel
    const std::atomic<float>* seed = nullptr;       // int
    const std::atomic<float>* hold = nullptr;       // bool
    const std::atomic<float>* world = nullptr;      // 0..1 normalized -> round(v*(N-1)) = world index
};

class WorkerCore
{
public:
    enum class Mode { OwnThread, ElkAsync, Synchronous };

    static constexpr size_t kRingCapacity  = 32768; // frames (> 7 hops @ 4480)
    static constexpr size_t kQueueCapacity = 256;   // commands (> one 8 s bar)
    static constexpr int kMaxBarHops = 96;          // 8 s / 93.3 ms = 85.7

    using Ring  = SpscMonoRing<kRingCapacity>;
    using Queue = SpscQueue<GrainCommand, kQueueCapacity>;

    WorkerCore() = default;
    ~WorkerCore();

    // Non-RT. pack/embedder are owned by the processor and outlive us; pack
    // may be null (plugin then passes dry only). Starts the thread in
    // OwnThread mode.
    void prepare (double sampleRate, int hopSamples, Mode mode,
                  const ElkgrainPack* pack, Embedder* embedder,
                  const WorkerParams& params);

    void shutdown();

    // RT side --------------------------------------------------------------
    Ring&  ring()  noexcept { return _ring; }
    Queue& queue() noexcept { return _queue; }

    // RT: called at each hop boundary. Returns true when the caller must file
    // the elk requestAsyncWork request (ElkAsync mode). In Synchronous mode
    // the caller runs drainPendingHops() itself right after this.
    bool publishHopFromRt() noexcept
    {
        _pendingHops.fetch_add (1, std::memory_order_release);
        if (_mode == Mode::OwnThread)
        {
            _cv.notify_one();
            return false;
        }
        return _mode == Mode::ElkAsync;
    }

    void noteAsyncCompletion() noexcept { _asyncCompletions.fetch_add (1, std::memory_order_relaxed); }
    void noteAsyncRequestFailed() noexcept { _asyncRequestFailures.fetch_add (1, std::memory_order_relaxed); }

    // Worker side (our thread, sushi's Worker thread, or inline in sync mode)
    int drainPendingHops();

    Mode mode() const noexcept { return _mode; }
    int  hopSamples() const noexcept { return _hopSamples; }

    // ---- World switching (T10): one pack live at a time; N packs known by path.
    // The load happens on a helper thread; the swap happens at a hop boundary on the
    // worker; the RT side switches when it pops a command flagged kCmdFlagSwitchPack
    // (kill voices, take rtNextPack(), acknowledge); the previous pack is released only
    // after that acknowledgement plus a few hops. Call setWorlds() AFTER prepare().
    static constexpr uint32_t kCmdFlagSwitchPack = 1u << 0;
    void setWorlds (std::vector<std::string> paths, int currentIndex);
    const ElkgrainPack* rtNextPack() const noexcept { return _rtNextPack.load (std::memory_order_acquire); }
    void rtAckSwitch() noexcept { _rtSwitchAck.fetch_add (1, std::memory_order_release); }
    int worldIndex() const noexcept { return _worldIndex; }
    int worldCount() const noexcept { return static_cast<int> (_worldPaths.size()); }

private:
    void threadLoop();
    void analyzeHop (int64_t nowNs);
    void finalizeBar();
    void selectSlot (int h);          // per-hop matching (the bar boundary only queues)
    void serviceWorldSwitch();
    int  desiredWorldIndex() const noexcept;
    void writeSummaryLocked();
    float param (const std::atomic<float>* p, float fallback) const noexcept;

    // config
    Mode _mode = Mode::OwnThread;
    double _sampleRate = 48000.0;
    int _hopSamples = 4480;
    const ElkgrainPack* _pack = nullptr;
    Embedder* _embedder = nullptr;
    WorkerParams _params;

    // data path
    Ring _ring;
    Queue _queue;
    std::vector<float> _hopBuf;

    // signaling (OwnThread mode)
    std::thread _thread;
    std::mutex _m;
    std::condition_variable _cv;
    std::atomic<bool> _running { false };
    std::atomic<int> _pendingHops { 0 };

    // elk-async bookkeeping
    std::atomic<uint64_t> _asyncCompletions { 0 };
    std::atomic<uint64_t> _asyncRequestFailures { 0 };

    // analysis / bar state (worker thread only)
    MelFrontend _frontend;
    Matcher _matcher;
    uint64_t _hopAbs = 0;      // hops consumed since prepare (sample clock /hop)
    int _slot = 0;             // current slot within the capture bar
    int _barHops = 21;         // slots in the bar being captured
    uint64_t _barsDone = 0;
    uint32_t _seq = 0;
    uint32_t _lastSeed = 0;
    bool _embedWarned = false;

    struct SlotState
    {
        float emb[Embedder::kLatentDim];
        float feat[MelFrontend::kFeatDim];
        float rms;
        bool open;
    };
    std::vector<SlotState> _slots;      // kMaxBarHops

    struct ScheduledSlot
    {
        int32_t grain;                  // -1 = silent slot
        float gain;
    };
    std::vector<ScheduledSlot> _lastBar; // last ANSWER bar (for hold)
    int _lastBarHops = 0;
    std::vector<ScheduledSlot> _curBar;  // the bar being matched, slot by slot as hops arrive
    int _barGated = 0, _barPicked = 0;
    double _barSimAcc = 0.0, _barBestAcc = 0.0;

    uint64_t _hopsProcessed = 0;
    uint64_t _cmdDrops = 0;
    uint64_t _ringStalls = 0;

    // jitter measurement (worker-side only)
    std::vector<int64_t> _hopTimestampsNs;

    // E2b per-stage timers (worker-side only; int32 µs, reserved like the timestamps).
    //   fe = MelFrontend::processHop      st = Embedder::embed      se = selectSlot (pushed only
    //   when a search ran: slot open + pack loaded)   wk = the whole analyzeHop body (world-switch
    //   service + RMS/gate + fe + st + se)   sc = finalizeBar (scheduling the answer), per bar.
    std::vector<int32_t> _tFeUs, _tStUs, _tSeUs, _tWkUs, _tScUs;
    double _barFeUs = 0.0, _barStUs = 0.0, _barSeUs = 0.0; // sums for the per-bar log line
    int _barSeN = 0;
    FILE* _log = nullptr;
    bool _prepared = false;
    bool _summaryWritten = false;

    // world switching state (worker-owned unless noted)
    std::vector<std::string> _worldPaths;
    int _worldIndex = -1;
    int _loadTarget = -1;
    std::unique_ptr<ElkgrainPack> _loading;        // being opened on _loader
    std::thread _loader;
    std::atomic<int> _loadState { 0 };             // 0 idle, 1 loading, 2 ready, 3 failed
    std::unique_ptr<ElkgrainPack> _ownedCurrent;   // current pack when it came from a switch
    std::unique_ptr<ElkgrainPack> _retired;        // previous pack, awaiting RT ack + a few hops
    uint64_t _retireAtHop = 0;
    bool _switchCmdPending = false;
    std::atomic<const ElkgrainPack*> _rtNextPack { nullptr };
    std::atomic<uint32_t> _rtSwitchAck { 0 };
    uint32_t _switchSeen = 0;
};
