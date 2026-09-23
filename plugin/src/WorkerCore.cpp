// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/WorkerCore.cpp (changed: comment)
#include "WorkerCore.h"

#include <algorithm>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace
{
    int64_t nowNs()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds> (
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    int32_t usBetween (int64_t t0Ns, int64_t t1Ns)
    {
        return static_cast<int32_t> ((t1Ns - t0Ns) / 1000);
    }

    // E2b: sort-and-index percentiles (the SUMMARY line's method) over µs samples, in ms.
    struct StageMs { double p50 = 0.0, p99 = 0.0, mx = 0.0; size_t n = 0; };
    StageMs stageMs (std::vector<int32_t> v) // by value: sorts a copy
    {
        StageMs r;
        r.n = v.size();
        if (v.empty())
            return r;
        std::sort (v.begin(), v.end());
        auto at = [&] (double p) {
            return static_cast<double> (v[static_cast<size_t> (p * static_cast<double> (v.size() - 1))]) / 1000.0;
        };
        r.p50 = at (0.50);
        r.p99 = at (0.99);
        r.mx = static_cast<double> (v.back()) / 1000.0;
        return r;
    }
}

float WorkerCore::param (const std::atomic<float>* p, float fallback) const noexcept
{
    return p != nullptr ? p->load (std::memory_order_relaxed) : fallback;
}

WorkerCore::~WorkerCore()
{
    shutdown();
}

void WorkerCore::prepare (double sampleRate, int hopSamples, Mode mode,
                          const ElkgrainPack* pack, Embedder* embedder,
                          const WorkerParams& params)
{
    shutdown(); // idempotent re-prepare

    _sampleRate = sampleRate;
    _hopSamples = hopSamples;
    _mode = mode;
    _pack = pack;
    _embedder = embedder;
    _params = params;

    // world-switch state: nothing in flight, the caller's pack is current
    if (_loader.joinable())
        _loader.join();
    _loading.reset();
    _loadState.store (0);
    _retired.reset();
    _ownedCurrent.reset();
    _rtNextPack.store (nullptr);
    _switchCmdPending = false;
    _switchSeen = _rtSwitchAck.load();

    _hopBuf.assign (static_cast<size_t> (_hopSamples), 0.0f);
    _hopTimestampsNs.clear();
    _hopTimestampsNs.reserve (1 << 14);
    for (auto* v : { &_tFeUs, &_tStUs, &_tSeUs, &_tWkUs, &_tScUs })
    {
        v->clear();
        v->reserve (1 << 14);
    }
    _barFeUs = _barStUs = _barSeUs = 0.0;
    _barSeN = 0;
    _ring.reset();
    _queue.reset();
    _pendingHops.store (0);
    _asyncCompletions.store (0);
    _asyncRequestFailures.store (0);

    _frontend.prepare (_hopSamples);
    if (_pack != nullptr && _pack->isOpen())
        _matcher.prepare (_pack->emb(), _pack->centre(),
                          static_cast<int> (_pack->nGrains()),
                          static_cast<int> (_pack->grainDim()));

    _hopAbs = 0;
    _slot = 0;
    _barsDone = 0;
    _seq = 0;
    _embedWarned = false;
    _slots.assign (kMaxBarHops, SlotState {});
    _lastBar.assign (kMaxBarHops, ScheduledSlot { -1, 0.0f });
    _lastBarHops = 0;
    _curBar.assign (kMaxBarHops, ScheduledSlot { -1, 0.0f });
    _barGated = _barPicked = 0;
    _barSimAcc = _barBestAcc = 0.0;

    const float barSec = param (_params.barSeconds, 2.0f);
    _barHops = std::clamp (static_cast<int> (std::lround (
                               barSec * _sampleRate / static_cast<double> (_hopSamples))),
                           1, kMaxBarHops);

    _lastSeed = static_cast<uint32_t> (std::max (0.0f, param (_params.seed, 0.0f)));
    _matcher.setSeed (_lastSeed);

    _hopsProcessed = 0;
    _cmdDrops = 0;
    _ringStalls = 0;
    _summaryWritten = false;

    const char* logPath = std::getenv ("ELKGRANULAR_LOG");
    _log = std::fopen (logPath != nullptr ? logPath : "/tmp/elkgranular-worker.log", "w");
    if (_log != nullptr)
    {
        const char* modeName = _mode == Mode::OwnThread ? "thread"
                              : _mode == Mode::ElkAsync ? "elkasync" : "sync";
        std::fprintf (_log,
                      "# elkgranular T09 worker  mode=%s hop=%d sr=%.0f target_ms=%.3f bar_hops=%d "
                      "pack=%s grains=%u student=%d ridge=%d table=%s table_bytes=%zu\n",
                      modeName, _hopSamples, _sampleRate,
                      1000.0 * _hopSamples / _sampleRate, _barHops,
                      _pack != nullptr && _pack->isOpen() ? _pack->path().c_str() : "NONE",
                      _pack != nullptr ? _pack->nGrains() : 0u,
                      _embedder != nullptr && _embedder->studentReady() ? 1 : 0,
                      _embedder != nullptr && _embedder->ridgeReady() ? 1 : 0,
                      _matcher.tableName(), _matcher.tableBytes()); // D1: f32 | int8 (ELKGRANULAR_TABLE)
        std::fflush (_log);
    }

    _prepared = true;

    if (_mode == Mode::OwnThread)
    {
        _running.store (true);
        _thread = std::thread ([this] { threadLoop(); });
    }
}

void WorkerCore::shutdown()
{
    if (! _prepared)
        return;

    if (_running.exchange (false))
    {
        _cv.notify_one();
        if (_thread.joinable())
            _thread.join();
    }
    if (_loader.joinable())
        _loader.join(); // a world load in flight finishes before we go

    if (! _summaryWritten)
        writeSummaryLocked();

    if (_log != nullptr)
    {
        std::fclose (_log);
        _log = nullptr;
    }
    _prepared = false;
}

void WorkerCore::threadLoop()
{
    while (_running.load (std::memory_order_acquire))
    {
        {
            std::unique_lock<std::mutex> lk (_m);
            _cv.wait_for (lk, std::chrono::milliseconds (20), [this] {
                return _pendingHops.load (std::memory_order_acquire) > 0
                       || ! _running.load (std::memory_order_acquire);
            });
        }
        if (! _running.load (std::memory_order_acquire))
            break;
        drainPendingHops();
    }
}

int WorkerCore::drainPendingHops()
{
    int processed = 0;
    while (_pendingHops.load (std::memory_order_acquire) > 0)
    {
        const int64_t t = nowNs(); // measure BEFORE any work or I/O
        if (! _ring.readBlock (_hopBuf.data(), _hopSamples))
        {
            ++_ringStalls;
            _pendingHops.fetch_sub (1, std::memory_order_acq_rel);
            continue;
        }
        analyzeHop (t);
        _pendingHops.fetch_sub (1, std::memory_order_acq_rel);
        ++processed;
    }
    return processed;
}

void WorkerCore::analyzeHop (int64_t tNs)
{
    _hopTimestampsNs.push_back (tNs); // reserved; realloc practically never

    const int64_t tWk0 = nowNs(); // E2b: worker busy time for this hop starts here
    serviceWorldSwitch(); // may swap _pack/_matcher before this hop is used

    auto& slot = _slots[static_cast<size_t> (_slot)];

    // input hop RMS (double accumulation; render_clips.py's rms_q shape,
    // window = the hop) and the input gate (live_engine measures RMS in dB)
    double acc = 0.0;
    for (float s : _hopBuf)
        acc += static_cast<double> (s) * static_cast<double> (s);
    const double rms = std::sqrt (acc / static_cast<double> (_hopBuf.size()) + 1e-12);
    const double inDb = 20.0 * std::log10 (rms + 1e-9);
    slot.rms = static_cast<float> (rms);
    slot.open = inDb > static_cast<double> (param (_params.gateDb, -50.0f));

    // frontend runs on EVERY hop (spectral-flux + resampler state continuity)
    // E2b: steady_clock stamps around the three seams (worker side; never the RT path)
    const int64_t tFe0 = nowNs();
    _frontend.processHop (_hopBuf.data(), slot.feat);
    const int64_t tFe1 = nowNs();

    // embedding
    auto mode = static_cast<Embedder::Mode> (
        std::clamp (static_cast<int> (std::lround (param (_params.embedMode, 0.0f))), 0, 2));
    if (_embedder != nullptr)
    {
        const auto wanted = mode;
        if (mode == Embedder::Mode::Student && ! _embedder->studentReady())
            mode = Embedder::Mode::RawMel;
        if (mode == Embedder::Mode::Ridge && ! _embedder->ridgeReady())
            mode = Embedder::Mode::RawMel;
        if (static_cast<int> (mode) != static_cast<int> (wanted) && ! _embedWarned && _log != nullptr)
        {
            std::fprintf (_log, "WARN embed_mode %d unavailable (weights missing) -> rawmel debug\n",
                          static_cast<int> (wanted));
            _embedWarned = true;
        }
        _embedder->embed (mode, slot.feat, slot.emb);
    }
    else
    {
        std::memcpy (slot.emb, slot.feat, sizeof (slot.emb));
    }

    const int64_t tSt1 = nowNs();

    // Per-hop matching (2026-09-03): a 12k-grain world costs ~20 ms per slot on the A7; done as a
    // burst at the bar boundary that was 300-500 ms (8 s bars would overrun the input ring).
    // Same slot order, same per-bar matcher reset -> identical picks, spread over the bar.
    const bool searched = slot.open && _pack != nullptr && _pack->isOpen(); // gated hops skip the search
    selectSlot (_slot);
    const int64_t tSe1 = nowNs();

    const int32_t feUs = usBetween (tFe0, tFe1);
    const int32_t stUs = usBetween (tFe1, tSt1);
    const int32_t seUs = searched ? usBetween (tSt1, tSe1) : -1;
    _tFeUs.push_back (feUs);
    _tStUs.push_back (stUs);
    _barFeUs += feUs;
    _barStUs += stUs;
    if (searched)
    {
        _tSeUs.push_back (seUs);
        _barSeUs += seUs;
        ++_barSeN;
    }

    ++_hopAbs;
    ++_slot;
    ++_hopsProcessed;
    const int32_t wkUs = usBetween (tWk0, nowNs());
    _tWkUs.push_back (wkUs);

    if ((_hopsProcessed & 7u) == 0u && _log != nullptr)
    {
        const size_t n = _hopTimestampsNs.size();
        const double dtMs = n >= 2
            ? static_cast<double> (_hopTimestampsNs[n - 1] - _hopTimestampsNs[n - 2]) / 1e6
            : 0.0;
        std::fprintf (_log,
                      "hop %llu dt_ms=%.3f in_db=%.1f open=%d slot=%d/%d fe_us=%d st_us=%d se_us=%d "
                      "wk_us=%d wet=%.3f tau=%.3f ring_over=%llu cmd_drops=%llu async_done=%llu\n",
                      static_cast<unsigned long long> (_hopsProcessed), dtMs, inDb,
                      slot.open ? 1 : 0, _slot, _barHops, feUs, stUs, seUs, wkUs,
                      param (_params.wet, -1.0f), param (_params.tau, -1.0f),
                      static_cast<unsigned long long> (_ring.overruns()),
                      static_cast<unsigned long long> (_cmdDrops),
                      static_cast<unsigned long long> (_asyncCompletions.load (std::memory_order_relaxed)));
    }

    if (_slot >= _barHops)
    {
        const int64_t tSc0 = nowNs();
        finalizeBar(); // schedules the answer for the following bar
        _tScUs.push_back (usBetween (tSc0, nowNs()));
    }
}

void WorkerCore::selectSlot (int h)
{
    if (h < 0 || h >= kMaxBarHops)
        return;
    auto& out = _curBar[static_cast<size_t> (h)];
    out.grain = -1;
    out.gain = 0.0f;
    if (_pack == nullptr || ! _pack->isOpen())
        return;

    if (h == 0)
    {
        // bar start: (re)seed if asked, reset the per-bar matcher state, reset the bar stats
        const auto seedNow = static_cast<uint32_t> (std::max (0.0f, param (_params.seed, 0.0f)));
        if (seedNow != _lastSeed)
        {
            _lastSeed = seedNow;
            _matcher.setSeed (seedNow);
            if (_log != nullptr)
                std::fprintf (_log, "seed -> %u\n", seedNow);
        }
        _matcher.resetBar(); // counts/prev are per-bar, like one substitute() call
        _barGated = _barPicked = 0;
        _barSimAcc = _barBestAcc = 0.0;
    }

    const auto& s = _slots[static_cast<size_t> (h)];
    if (! s.open)
    {
        ++_barGated;
        return;
    }
    const float tau = param (_params.tau, 0.05f);
    const float antiHub = 0.25f * param (_params.variety, 0.0f);
    const float stick = param (_params.stick, 0.0f);
    const bool gainMatch = param (_params.gainMatch, 1.0f) >= 0.5f;

    const auto r = _matcher.selectOne (s.emb, tau, antiHub, stick);
    if (r.pick < 0)
        return;
    float g = 1.0f;
    if (gainMatch)
    {
        const float rmsG = std::max (_pack->grainCoreRms()[r.pick], 1e-6f);
        g = std::min (s.rms / rmsG, 4.0f); // render_clips.py clamp
    }
    out.grain = r.pick;
    out.gain = g;
    ++_barPicked;
    _barSimAcc += static_cast<double> (r.pickSim);
    _barBestAcc += static_cast<double> (r.bestSim);
}

void WorkerCore::finalizeBar()
{
    const int capturedHops = _slot;
    _slot = 0;

    // E2b: mean stage times over this bar's hops (µs) for the bar line, then reset
    const double feMeanUs = capturedHops > 0 ? _barFeUs / capturedHops : 0.0;
    const double stMeanUs = capturedHops > 0 ? _barStUs / capturedHops : 0.0;
    const double seMeanUs = _barSeN > 0 ? _barSeUs / _barSeN : 0.0;
    _barFeUs = _barStUs = _barSeUs = 0.0;
    _barSeN = 0;

    // bar clock: bar_seconds parameter ONLY (sushi's processContext transport
    // is broken) — re-read for the NEXT bar at every boundary
    const float barSec = param (_params.barSeconds, 2.0f);
    _barHops = std::clamp (static_cast<int> (std::lround (
                               barSec * _sampleRate / static_cast<double> (_hopSamples))),
                           1, kMaxBarHops);

    if (_pack == nullptr || ! _pack->isOpen())
        return;

    const uint64_t barBase = _hopAbs * static_cast<uint64_t> (_hopSamples); // next bar start
    const uint32_t half = _pack->margin48() / 2;
    const uint32_t segLen = _pack->grainLen48() + _pack->margin48();
    const bool hold = param (_params.hold, 0.0f) >= 0.5f;

    int queued = 0, gated = 0, picked = 0;
    double simAcc = 0.0, bestAcc = 0.0;

    if (hold && _lastBarHops > 0)
    {
        // ---- cheap Hold: repeat the last ANSWER bar verbatim -------------
        for (int h = 0; h < _lastBarHops; ++h)
        {
            const auto& s = _lastBar[static_cast<size_t> (h)];
            if (s.grain < 0)
                continue;
            GrainCommand c {};
            c.seq = _seq++;
            c.grainIndex = static_cast<uint32_t> (s.grain);
            c.startSample = static_cast<uint32_t> (
                barBase + static_cast<uint64_t> (h) * static_cast<uint64_t> (_hopSamples) - half);
            c.lengthFrames = segLen;
            c.gain = s.gain;
            c.pan = 0.0f;
            c.pitchRatio = 1.0f;
            if (_queue.push (c))
                ++queued;
            else
                ++_cmdDrops;
        }
        ++_barsDone;
        if (_log != nullptr)
        {
            std::fprintf (_log, "bar %llu HOLD replay slots=%d queued=%d\n",
                          static_cast<unsigned long long> (_barsDone), _lastBarHops, queued);
            std::fflush (_log);
        }
        return;
    }

    // ---- schedule the answer for the following bar (picks were made per hop in selectSlot) ----
    const float tau = param (_params.tau, 0.05f);
    const float antiHub = 0.25f * param (_params.variety, 0.0f); // instrument scaling (log only)
    const float stick = param (_params.stick, 0.0f);
    const bool gainMatch = param (_params.gainMatch, 1.0f) >= 0.5f;
    gated = _barGated;
    picked = _barPicked;
    simAcc = _barSimAcc;
    bestAcc = _barBestAcc;

    for (int h = 0; h < capturedHops; ++h)
    {
        const auto& cur = _curBar[static_cast<size_t> (h)];
        _lastBar[static_cast<size_t> (h)] = cur;
        if (cur.grain < 0)
            continue;
        GrainCommand c {};
        c.seq = _seq++;
        c.grainIndex = static_cast<uint32_t> (cur.grain);
        c.startSample = static_cast<uint32_t> (
            barBase + static_cast<uint64_t> (h) * static_cast<uint64_t> (_hopSamples) - half);
        c.lengthFrames = segLen;
        c.gain = cur.gain;
        c.pan = 0.0f;
        c.pitchRatio = 1.0f;
        if (_queue.push (c))
            ++queued;
        else
            ++_cmdDrops;
    }
    _lastBarHops = capturedHops;
    ++_barsDone;

    if (_log != nullptr)
    {
        char picksBuf[128];
        int off = 0;
        for (int h = 0; h < capturedHops && off < 110; ++h)
        {
            const int gI = _lastBar[static_cast<size_t> (h)].grain;
            off += std::snprintf (picksBuf + off, sizeof (picksBuf) - static_cast<size_t> (off),
                                  h == 0 ? "%d" : ",%d", gI);
        }
        picksBuf[sizeof (picksBuf) - 1] = 0;
        std::fprintf (_log,
                      "bar %llu slots=%d gated=%d picked=%d queued=%d tau=%.3f anti_hub=%.3f "
                      "stick=%.2f gm=%d cov=%.3f picked_sim=%.3f fe_us=%.0f st_us=%.0f se_us=%.0f "
                      "picks=[%s]\n",
                      static_cast<unsigned long long> (_barsDone), capturedHops, gated, picked,
                      queued, tau, antiHub, stick, gainMatch ? 1 : 0,
                      picked > 0 ? bestAcc / picked : 0.0,
                      picked > 0 ? simAcc / picked : 0.0, feMeanUs, stMeanUs, seMeanUs, picksBuf);
        std::fflush (_log);
    }
}

void WorkerCore::setWorlds (std::vector<std::string> paths, int currentIndex)
{
    _worldPaths = std::move (paths);
    _worldIndex = currentIndex;
    if (_log != nullptr)
    {
        std::fprintf (_log, "# worlds: %d", static_cast<int> (_worldPaths.size()));
        for (size_t i = 0; i < _worldPaths.size(); ++i)
            std::fprintf (_log, " [%zu]%s%s", i, _worldPaths[i].c_str(),
                          static_cast<int> (i) == _worldIndex ? "*" : "");
        std::fprintf (_log, "\n");
    }
}

int WorkerCore::desiredWorldIndex() const noexcept
{
    const int n = static_cast<int> (_worldPaths.size());
    if (n < 2)
        return _worldIndex;
    const float v = std::clamp (param (_params.world, 0.0f), 0.0f, 1.0f);
    return std::clamp (static_cast<int> (std::lround (v * static_cast<float> (n - 1))), 0, n - 1);
}

void WorkerCore::serviceWorldSwitch()
{
    if (_worldPaths.size() < 2)
        return;

    // 1) release the previous pack once the RT side acknowledged and a few hops passed
    if (_retired != nullptr && _rtSwitchAck.load (std::memory_order_acquire) > _switchSeen
        && _hopAbs >= _retireAtHop)
    {
        _switchSeen = _rtSwitchAck.load (std::memory_order_acquire);
        _retired.reset(); // munmap off the RT thread
        if (_log != nullptr)
            std::fprintf (_log, "world: previous pack released at hop %llu\n",
                          static_cast<unsigned long long> (_hopAbs));
    }

    // 2) the RT side never got the switch command (queue full) -> retry
    if (_switchCmdPending)
    {
        GrainCommand c {};
        c.seq = _seq++;
        c.flags = kCmdFlagSwitchPack;
        if (_queue.push (c))
            _switchCmdPending = false;
        return;
    }

    // 3) a load finished -> swap on the worker side, then tell the RT side
    const int st = _loadState.load (std::memory_order_acquire);
    if (st == 2 || st == 3)
    {
        if (_loader.joinable())
            _loader.join();
        _loadState.store (0);
        if (st == 3 || _loading == nullptr || ! _loading->isOpen())
        {
            if (_log != nullptr)
                std::fprintf (_log, "world: load of [%d] %s FAILED: %s\n", _loadTarget,
                              _loadTarget >= 0 ? _worldPaths[static_cast<size_t> (_loadTarget)].c_str() : "?",
                              _loading != nullptr ? _loading->error().c_str() : "no pack");
            _loading.reset();
            _worldIndex = _loadTarget; // do not retry the same broken pack every hop
            return;
        }
        auto np = std::move (_loading);
        _matcher.prepare (np->emb(), np->centre(),
                          static_cast<int> (np->nGrains()), static_cast<int> (np->grainDim()));
        _matcher.setSeed (_lastSeed);
        _matcher.resetBar();
        _slot = 0;          // restart the capture bar
        _lastBarHops = 0;   // hold must not replay grains of another pack
        _rtNextPack.store (np.get(), std::memory_order_release);
        _retired = std::move (_ownedCurrent); // null when the current pack is the processor's own
        _ownedCurrent = std::move (np);
        _pack = _ownedCurrent.get();
        _worldIndex = _loadTarget;
        _retireAtHop = _hopAbs + 4;
        GrainCommand c {};
        c.seq = _seq++;
        c.flags = kCmdFlagSwitchPack;
        if (! _queue.push (c))
            _switchCmdPending = true;
        if (_log != nullptr)
            std::fprintf (_log, "world %d -> %s (%u grains) at hop %llu table=%s table_bytes=%zu\n", _worldIndex,
                          _pack->path().c_str(), _pack->nGrains(),
                          static_cast<unsigned long long> (_hopAbs),
                          _matcher.tableName(), _matcher.tableBytes());
        std::fprintf (stderr, "ElkGranular: world %d -> %s (%u grains)\n", _worldIndex,
                      _pack->path().c_str(), _pack->nGrains());
        return;
    }

    // 4) idle and nothing retired -> start a load if the knob asks for another world
    if (st == 0 && _retired == nullptr)
    {
        const int want = desiredWorldIndex();
        if (want != _worldIndex && want >= 0 && want < static_cast<int> (_worldPaths.size()))
        {
            _loadTarget = want;
            _loading = std::make_unique<ElkgrainPack>();
            _loadState.store (1, std::memory_order_release);
            _loader = std::thread ([this]
            {
#ifdef __linux__
                // The board has two cores: audio on CPU1, everything else on CPU0. A 113 MB
                // mmap + pre-touch + RMS pass at normal priority starves the hop worker
                // (p99 340 ms measured 2026-09-03). Idle scheduling: the load only takes
                // cycles nobody else wants; the switch just arrives a little later.
                sched_param sp {};
                sp.sched_priority = 0;
                pthread_setschedparam (pthread_self(), SCHED_IDLE, &sp);
#endif
                const bool ok = _loading->open (_worldPaths[static_cast<size_t> (_loadTarget)])
                                && _loading->unit() == 1;
                _loadState.store (ok ? 2 : 3, std::memory_order_release);
            });
            if (_log != nullptr)
                std::fprintf (_log, "world: loading [%d] %s\n", want,
                              _worldPaths[static_cast<size_t> (want)].c_str());
        }
    }
}

void WorkerCore::writeSummaryLocked()
{
    _summaryWritten = true;
    if (_log == nullptr)
        return;

    std::vector<double> dts;
    dts.reserve (_hopTimestampsNs.size());
    for (size_t i = 1; i < _hopTimestampsNs.size(); ++i)
        dts.push_back (static_cast<double> (_hopTimestampsNs[i] - _hopTimestampsNs[i - 1]) / 1e6);

    if (dts.empty())
    {
        std::fprintf (_log, "SUMMARY no hops processed\n");
        std::fflush (_log);
        return;
    }

    std::sort (dts.begin(), dts.end());
    auto pct = [&] (double p) {
        const auto idx = static_cast<size_t> (p * static_cast<double> (dts.size() - 1));
        return dts[idx];
    };
    const double target = 1000.0 * _hopSamples / _sampleRate;
    const char* modeName = _mode == Mode::OwnThread ? "thread"
                          : _mode == Mode::ElkAsync ? "elkasync" : "sync";

    std::fprintf (_log,
                  "SUMMARY mode=%s hops=%llu bars=%llu target_ms=%.3f p50=%.3f p90=%.3f p99=%.3f "
                  "min=%.3f max=%.3f ring_overruns=%llu ring_stalls=%llu cmd_drops=%llu "
                  "async_done=%llu async_req_failed=%llu\n",
                  modeName,
                  static_cast<unsigned long long> (_hopsProcessed),
                  static_cast<unsigned long long> (_barsDone), target,
                  pct (0.50), pct (0.90), pct (0.99), dts.front(), dts.back(),
                  static_cast<unsigned long long> (_ring.overruns()),
                  static_cast<unsigned long long> (_ringStalls),
                  static_cast<unsigned long long> (_cmdDrops),
                  static_cast<unsigned long long> (_asyncCompletions.load()),
                  static_cast<unsigned long long> (_asyncRequestFailures.load()));

    // E2b: per-stage p50/p99/max over the run (ms). n_open = hops whose slot was open (a search
    // ran); the front-end and student run on every hop; sc is per bar.
    const StageMs fe = stageMs (_tFeUs), st = stageMs (_tStUs), se = stageMs (_tSeUs),
                  wk = stageMs (_tWkUs), sc = stageMs (_tScUs);
    std::fprintf (_log,
                  "STAGES grains=%u n_hops=%zu n_open=%zu n_bars=%zu "
                  "fe_p50=%.3f fe_p99=%.3f fe_max=%.3f st_p50=%.3f st_p99=%.3f st_max=%.3f "
                  "se_p50=%.3f se_p99=%.3f se_max=%.3f wk_p50=%.3f wk_p99=%.3f wk_max=%.3f "
                  "sc_p50=%.3f sc_p99=%.3f sc_max=%.3f unit=ms\n",
                  _pack != nullptr && _pack->isOpen() ? _pack->nGrains() : 0u,
                  fe.n, se.n, sc.n,
                  fe.p50, fe.p99, fe.mx, st.p50, st.p99, st.mx,
                  se.p50, se.p99, se.mx, wk.p50, wk.p99, wk.mx,
                  sc.p50, sc.p99, sc.mx);
    std::fflush (_log);
}
