// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/PluginProcessor.cpp (changed: default paths)
#include "PluginProcessor.h"
#include <algorithm>

#if ELKGRANULAR_WITH_ELK_ASYNC
#include "ElkAsync.h"
#endif

#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Debug RT alloc guard: any operator new/delete while the RT section is open
// trips an assert (see RtGuard.h). One-TU definition.
// ---------------------------------------------------------------------------
#if ELKGRANULAR_RT_ALLOC_GUARD

#include <cassert>
#include <cstdio>
#include <new>

namespace rtguard
{
    thread_local int depth = 0;

    static void checkRtViolation (const char* what) noexcept
    {
        if (depth > 0)
        {
            std::fprintf (stderr, "ELKGRANULAR RT VIOLATION: %s inside the RT section\n", what);
            assert (false && "heap operation inside the RT audio section");
        }
    }
}

void* operator new (std::size_t sz)
{
    rtguard::checkRtViolation ("operator new");
    if (void* p = std::malloc (sz > 0 ? sz : 1))
        return p;
    throw std::bad_alloc {};
}

void* operator new[] (std::size_t sz)
{
    rtguard::checkRtViolation ("operator new[]");
    if (void* p = std::malloc (sz > 0 ? sz : 1))
        return p;
    throw std::bad_alloc {};
}

void operator delete (void* p) noexcept
{
    rtguard::checkRtViolation ("operator delete");
    std::free (p);
}

void operator delete[] (void* p) noexcept
{
    rtguard::checkRtViolation ("operator delete[]");
    std::free (p);
}

void operator delete (void* p, std::size_t) noexcept { operator delete (p); }
void operator delete[] (void* p, std::size_t) noexcept { operator delete[] (p); }

#endif // ELKGRANULAR_RT_ALLOC_GUARD

// ---------------------------------------------------------------------------

juce::AudioProcessorValueTreeState::ParameterLayout ElkGranularProcessor::createLayout()
{
    using PF = juce::AudioParameterFloat;
    using PB = juce::AudioParameterBool;
    using PC = juce::AudioParameterChoice;
    using PI = juce::AudioParameterInt;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> ps;

    // SUSHI addresses VST3 parameters by DISPLAY NAME — the display names
    // below are the snake_case ids VERBATIM. Do not prettify them.
    // Ranges/defaults follow docs/controls_reference.md + the T06 decisions
    // (tau default 0.05; below ~0.03 cheap embeddings hub hard unless
    // variety > 0 — documented, not clamped).
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "wet", 1 }, "wet",
                                        juce::NormalisableRange<float> (0.0f, 1.0f), 1.0f));
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "tau", 1 }, "tau",
                                        juce::NormalisableRange<float> (0.0f, 0.30f), 0.05f));
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "gate_db", 1 }, "gate_db",
                                        juce::NormalisableRange<float> (-80.0f, -20.0f), -50.0f));
    // Bar clock comes ONLY from this parameter — VST3 processContext
    // transport is broken under sushi and is never read. Snap-free.
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "bar_seconds", 1 }, "bar_seconds",
                                        juce::NormalisableRange<float> (0.5f, 8.0f), 2.0f));
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "stick", 1 }, "stick",
                                        juce::NormalisableRange<float> (0.0f, 1.0f), 0.0f));
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "variety", 1 }, "variety",
                                        juce::NormalisableRange<float> (0.0f, 1.0f), 0.0f));
    ps.push_back (std::make_unique<PB> (juce::ParameterID { "gain_match", 1 }, "gain_match", true));
    ps.push_back (std::make_unique<PC> (juce::ParameterID { "embed_mode", 1 }, "embed_mode",
                                        juce::StringArray { "student", "ridge", "rawmel" }, 0));
    ps.push_back (std::make_unique<PI> (juce::ParameterID { "seed", 1 }, "seed", 0, 9999, 0));
    ps.push_back (std::make_unique<PB> (juce::ParameterID { "hold", 1 }, "hold", false));
    // T10: which world (pack) answers. 0..1 normalized -> round(v*(N-1)) over the packs
    // found next to the initial pack (worlds.txt order, else *.elkgrain sorted). The
    // 4-position rotary maps to 0, 1/3, 2/3, 1 -> worlds 0..3 exactly when N = 4.
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "world", 1 }, "world",
                                        juce::NormalisableRange<float> (0.0f, 1.0f), 0.0f));
    // Output trim in dB (the Stomp outs are line level; headphones need less)
    ps.push_back (std::make_unique<PF> (juce::ParameterID { "out_gain", 1 }, "out_gain",
                                        juce::NormalisableRange<float> (-24.0f, 12.0f), 0.0f));
    return { ps.begin(), ps.end() };
}

ElkGranularProcessor::ElkGranularProcessor()
    : juce::AudioProcessor (BusesProperties()
                                .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                                .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      _apvts (*this, nullptr, "params", createLayout())
{
    _pWet = _apvts.getRawParameterValue ("wet");
    _wp.wet = _pWet;
    _wp.tau = _apvts.getRawParameterValue ("tau");
    _wp.gateDb = _apvts.getRawParameterValue ("gate_db");
    _wp.barSeconds = _apvts.getRawParameterValue ("bar_seconds");
    _wp.stick = _apvts.getRawParameterValue ("stick");
    _wp.variety = _apvts.getRawParameterValue ("variety");
    _wp.gainMatch = _apvts.getRawParameterValue ("gain_match");
    _wp.embedMode = _apvts.getRawParameterValue ("embed_mode");
    _wp.seed = _apvts.getRawParameterValue ("seed");
    _wp.hold = _apvts.getRawParameterValue ("hold");
    _wp.world = _apvts.getRawParameterValue ("world");
    _pOutGain = _apvts.getRawParameterValue ("out_gain");

#if ELKGRANULAR_WITH_ELK_ASYNC
    _bridge = ElkAsyncBridge::create (_worker);
#endif
}

ElkGranularProcessor::~ElkGranularProcessor()
{
    _worker.shutdown();
}

#if ELKGRANULAR_WITH_ELK_ASYNC
juce::VST3ClientExtensions* ElkGranularProcessor::getVST3ClientExtensions()
{
    return _bridge != nullptr ? _bridge->clientExtensions() : nullptr;
}
#endif

bool ElkGranularProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& in = layouts.getMainInputChannelSet();
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    if (in != juce::AudioChannelSet::mono() && in != juce::AudioChannelSet::stereo())
        return false;
    return true;
}

// T10: the list of worlds (packs) this instance can switch between. Order = rotary order.
//   ELKGRANULAR_WORLDS=<file>   explicit list, one pack per line (absolute or relative to the pack dir)
//   <packdir>/worlds.txt        same, found next to the initial pack
//   else                        every *.elkgrain in the pack dir, sorted, minus demo*/test*
// The initial pack is always in the list (inserted first if absent).
static std::vector<std::string> discoverWorlds (const std::string& initialPackPath)
{
    const juce::File initial = juce::File::getCurrentWorkingDirectory().getChildFile (juce::String (initialPackPath));
    const juce::File dir = initial.getParentDirectory();
    std::vector<std::string> out;
    auto addList = [&] (const juce::File& listFile)
    {
        juce::StringArray lines;
        lines.addLines (listFile.loadFileAsString());
        for (const auto& line : lines)
        {
            const auto t = line.trim();
            if (t.isEmpty() || t.startsWith ("#"))
                continue;
            const juce::File f = juce::File::isAbsolutePath (t) ? juce::File (t) : dir.getChildFile (t);
            if (f.existsAsFile())
                out.push_back (f.getFullPathName().toStdString());
        }
    };
    if (const char* e = std::getenv ("ELKGRANULAR_WORLDS"); e != nullptr && juce::File (juce::String (e)).existsAsFile())
        addList (juce::File (juce::String (e)));
    else if (dir.getChildFile ("worlds.txt").existsAsFile())
        addList (dir.getChildFile ("worlds.txt"));
    else
    {
        for (const auto& f : dir.findChildFiles (juce::File::findFiles, false, "*.elkgrain"))
        {
            const auto name = f.getFileNameWithoutExtension();
            if (name.startsWithIgnoreCase ("demo") || name.startsWithIgnoreCase ("test"))
                continue;
            out.push_back (f.getFullPathName().toStdString());
        }
        std::sort (out.begin(), out.end());
    }
    const auto initialFull = initial.getFullPathName().toStdString();
    if (std::find (out.begin(), out.end(), initialFull) == out.end())
        out.insert (out.begin(), initialFull);
    return out;
}

void ElkGranularProcessor::loadPackAndWeights()
{
    // Non-RT (prepareToPlay). Env override (board/run_board.sh sets all three),
    // then relative to SUSHI's working directory.
    auto firstExisting = [] (const char* env, std::initializer_list<const char*> candidates) {
        if (const char* e = std::getenv (env))
            return std::string (e);
        struct stat st {};
        for (const char* c : candidates)
            if (::stat (c, &st) == 0)
                return std::string (c);
        return std::string (*candidates.begin());
    };

    const auto packPath = firstExisting ("ELKGRANULAR_PACK",
        { "packs/field.elkgrain" });
    const auto studentPath = firstExisting ("ELKGRANULAR_STUDENT",
        { "packs/student_mlp_mel80.json" });
    const auto ridgePath = firstExisting ("ELKGRANULAR_RIDGE",
        { "packs/ridge_pooled.json" });

    if (! _pack.isOpen() || _pack.path() != packPath)
    {
        if (_pack.open (packPath))
        {
            std::fprintf (stderr,
                          "ElkGranular: pack %s (%u grains, unit=%u, hop48=%u, margin48=%u)\n",
                          packPath.c_str(), _pack.nGrains(), _pack.unit(), _pack.hop48(),
                          _pack.margin48());
            if (_pack.unit() != 1)
            {
                std::fprintf (stderr, "ElkGranular: unit=%u packs are not supported by the T09 "
                                      "bar loop yet (unit=1 only) — pack disabled\n", _pack.unit());
                _pack.close();
            }
        }
        else
        {
            std::fprintf (stderr, "ElkGranular: PACK LOAD FAILED: %s (dry passthrough only)\n",
                          _pack.error().c_str());
        }
    }

    std::string err;
    if (! _embedder.studentReady() && ! _embedder.loadStudent (studentPath, err))
        std::fprintf (stderr, "ElkGranular: student weights unavailable: %s\n", err.c_str());
    if (! _embedder.ridgeReady() && ! _embedder.loadRidge (ridgePath, err))
        std::fprintf (stderr, "ElkGranular: ridge map unavailable: %s\n", err.c_str());

    // T10: worlds = the packs next to this one (see discoverWorlds)
    _worldPaths.clear();
    _worldIndex = 0;
    if (_pack.isOpen())
    {
        _worldPaths = discoverWorlds (_pack.path());
        const auto full = juce::File::getCurrentWorkingDirectory().getChildFile (juce::String (_pack.path())).getFullPathName().toStdString();
        const auto it = std::find (_worldPaths.begin(), _worldPaths.end(), full);
        _worldIndex = it == _worldPaths.end() ? 0 : static_cast<int> (it - _worldPaths.begin());
        std::fprintf (stderr, "ElkGranular: %d world(s), current [%d]\n",
                      static_cast<int> (_worldPaths.size()), _worldIndex);
    }
}

void ElkGranularProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // All allocation and pack mmap/pre-touch happen HERE (non-RT).
    loadPackAndWeights();
    if (_pack.isOpen())
        _pool.preparePack (_pack);

    // Analysis hop: 4480 @48k == 93.33 ms (the codec frame grid). The
    // devkit/board always run 48 kHz; other rates would shift the grid.
    _hopSamples = juce::roundToInt (sampleRate * 4480.0 / 48000.0);
    _hopPhase = 0;
    _rtPos = 0;

    const auto scratchLen = static_cast<size_t> (juce::jmax (4096, samplesPerBlock));
    _scratchL.assign (scratchLen, 0.0f);
    _scratchR.assign (scratchLen, 0.0f);

    // Worker mode (T08 decision): AUTO — host-managed requestAsyncWork when
    // the host exposes IElkHostExtension (sushi), own thread + CV everywhere
    // else. ELKGRANULAR_WORKER_MODE=thread|elkasync|sync forces a mode
    // (sync = inline on the audio thread, OFFLINE BOUNCES ONLY).
    const char* m = std::getenv ("ELKGRANULAR_WORKER_MODE");
    const bool forceThread = m != nullptr && std::strcmp (m, "thread") == 0;
    const bool forceElk = m != nullptr && std::strcmp (m, "elkasync") == 0;
    const bool forceSync = m != nullptr && std::strcmp (m, "sync") == 0;

    auto mode = WorkerCore::Mode::OwnThread;
    _elkAsyncActive = false;
    _syncWorker = false;
    if (forceSync)
    {
        mode = WorkerCore::Mode::Synchronous;
        _syncWorker = true;
    }
#if ELKGRANULAR_WITH_ELK_ASYNC
    else if (! forceThread && _bridge != nullptr && _bridge->hostSupportsAsyncWork())
    {
        mode = WorkerCore::Mode::ElkAsync;
        _elkAsyncActive = true;
    }
#endif
    if (forceElk && ! _elkAsyncActive)
        std::fprintf (stderr, "ElkGranular: elkasync requested but unavailable "
                              "(host lacks IElkHostExtension or not compiled in) — using own thread\n");

    _rtPack = _pack.isOpen() ? &_pack : nullptr;
    _worker.prepare (sampleRate, _hopSamples, mode,
                     _pack.isOpen() ? &_pack : nullptr, &_embedder, _wp);
    _worker.setWorlds (_worldPaths, _worldIndex);
    // reflect the initial world on the parameter so knobs and displays agree
    if (auto* wp = _apvts.getParameter ("world"); wp != nullptr && _worldPaths.size() > 1)
        wp->setValueNotifyingHost (static_cast<float> (_worldIndex) / static_cast<float> (_worldPaths.size() - 1));
    _preparedOk = true;
}

void ElkGranularProcessor::releaseResources()
{
    _preparedOk = false;
    _worker.shutdown(); // joins the thread (if any) and writes the summary
}

void ElkGranularProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    ELKGRANULAR_RT_SECTION; // debug builds assert on any allocation below
    juce::ScopedNoDenormals noDenormals;

    const int n = buffer.getNumSamples();
    const int nCh = juce::jmin (buffer.getNumChannels(), 2);
    if (! _preparedOk || n <= 0 || nCh <= 0)
        return;
    const auto un = static_cast<size_t> (n);
    if (un > _scratchL.size())
        return; // never touch unallocated memory

    // 1) drain worker commands into the voice pool (SPSC pop, wait-free).
    //    _rtPos is the absolute sample clock at the START of this block —
    //    commands carry absolute start times, so scheduling is exact here.
    GrainCommand cmd;
    while (_worker.queue().pop (cmd))
    {
        if ((cmd.flags & WorkerCore::kCmdFlagSwitchPack) != 0u)
        {
            // T10 world switch: voices of the old pack die here, before the swap
            _pool.killAll();
            _rtPack = _worker.rtNextPack();
            if (_rtPack != nullptr)
                _pool.preparePackIfChanged (*_rtPack);
            _worker.rtAckSwitch();
            continue;
        }
        _pool.activate (cmd, _rtPack, _rtPos);
    }

    // 2) feed the analysis ring BEFORE the buffer is overwritten
    const float* chans[2] = { buffer.getReadPointer (0),
                              nCh > 1 ? buffer.getReadPointer (1) : buffer.getReadPointer (0) };
    _worker.ring().writeMonoFrom (chans, 2, n);

    // 3) hop boundaries -> signal the worker (CV notify / elk async / inline)
    _hopPhase += n;
    while (_hopPhase >= _hopSamples)
    {
        _hopPhase -= _hopSamples;
        const bool fileElkRequest = _worker.publishHopFromRt();
        if (fileElkRequest)
        {
#if ELKGRANULAR_WITH_ELK_ASYNC
            if (_bridge == nullptr || ! _bridge->requestWorkFromRt())
                _worker.noteAsyncRequestFailed();
#else
            _worker.noteAsyncRequestFailed();
#endif
        }
        else if (_syncWorker)
        {
            // OFFLINE ONLY: run the whole analysis inline (deterministic
            // renders; violates RT hygiene by design — never realtime).
            _worker.drainPendingHops();
        }
    }

    // 4) render the answer bus (reads pre-touched mmap'd pack audio only)
    std::memset (_scratchL.data(), 0, un * sizeof (float));
    std::memset (_scratchR.data(), 0, un * sizeof (float));
    _pool.render (_scratchL.data(), _scratchR.data(), n);

    // 5) wet/dry: LIVE dry vs the one-bar-late answer
    const float wet = _pWet != nullptr ? _pWet->load (std::memory_order_relaxed) : 1.0f;
    const float dry = 1.0f - wet;
    const float og = _pOutGain != nullptr
        ? juce::Decibels::decibelsToGain (_pOutGain->load (std::memory_order_relaxed), -24.0f) : 1.0f;
    const float* g[2] = { _scratchL.data(), _scratchR.data() };

    for (int c = 0; c < buffer.getNumChannels(); ++c)
    {
        auto* out = buffer.getWritePointer (c);
        const float* answer = g[juce::jmin (c, 1)];
        for (int i = 0; i < n; ++i)
            out[i] = (out[i] * dry + answer[i] * wet) * og;
    }

    _rtPos += static_cast<uint64_t> (n);
}

void ElkGranularProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = _apvts.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void ElkGranularProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        _apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

// JUCE plugin factory entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ElkGranularProcessor();
}
