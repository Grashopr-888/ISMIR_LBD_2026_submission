// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/PluginProcessor.h (changed: comment)
// ElkGranular — T09 MVP: the latent-granular instrument under SUSHI.
// Live input is re-voiced through a .elkgrain corpus and played back in the following bar.
//
//   RT thread (processBlock, 64-frame chunks under sushi):
//     pop GrainCommands -> voice pool (pack int16 windows, equal-power
//     margin crossfades); write input mixdown -> SPSC ring; publish hop
//     boundaries; render answer bus; out = in*(1-wet) + answer*wet.
//     ZERO allocation / locks / file IO (debug builds assert on alloc).
//   Worker (~10.7 Hz): WorkerCore — mel80 frontend on the T06 training grid,
//     student MLP (RTNeural) / ridge / rawmel embedding, exact match.py
//     matcher, bar assembly + one-bar-late scheduling. Modes: own thread +
//     CV, sushi requestAsyncWork (auto under sushi), or synchronous
//     (offline bounces), via ELKGRANULAR_WORKER_MODE=thread|elkasync|sync.
//
// SUSHI contracts honored here:
//   * parameter DISPLAY NAMES are the snake_case ids verbatim:
//     wet, tau, gate_db, bar_seconds, stick, variety, gain_match,
//     embed_mode, seed, hold — sushi addresses by name.
//   * no VST3 processContext transport (broken under sushi) — the bar clock
//     is the bar_seconds parameter only, sample-counted internally.
//
// Pack / weights loading (non-RT, prepareToPlay; string-property loading is
// the eventual Elk pattern — env vars stand in for it on the desktop):
//   ELKGRANULAR_PACK    .elkgrain path   (default: packs/field.elkgrain)
//   ELKGRANULAR_STUDENT student JSON     (default: packs/student_mlp_mel80.json)
//   ELKGRANULAR_RIDGE   ridge JSON       (default: packs/ridge_pooled.json)
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <string>
#include <vector>
#include "ElkgrainPack.h"
#include "Embedder.h"
#include "GrainEngine.h"
#include "RtGuard.h"
#include "WorkerCore.h"

#if ELKGRANULAR_WITH_ELK_ASYNC
class ElkAsyncBridge; // defined in ElkAsync.cpp (Steinberg types kept out of here)
#endif

class ElkGranularProcessor : public juce::AudioProcessor
{
public:
    ElkGranularProcessor();
    ~ElkGranularProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }

    const juce::String getName() const override { return "ElkGranular"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

#if ELKGRANULAR_WITH_ELK_ASYNC
    juce::VST3ClientExtensions* getVST3ClientExtensions() override;
#endif

    WorkerCore& worker() noexcept { return _worker; }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void loadPackAndWeights(); // non-RT; env-var paths with defaults

    juce::AudioProcessorValueTreeState _apvts;

    // raw-value atomics (RT/worker-safe reads)
    std::atomic<float>* _pWet = nullptr;
    std::atomic<float>* _pOutGain = nullptr;
    const ElkgrainPack* _rtPack = nullptr;      // RT-only: the pack voices read from (switches on a flagged command)
    std::vector<std::string> _worldPaths;       // discovered next to the initial pack (worlds.txt or *.elkgrain)
    int _worldIndex = 0;
    WorkerParams _wp;

    ElkgrainPack _pack;
    Embedder _embedder;
    GrainVoicePool _pool;
    WorkerCore _worker;

    // RT state
    int _hopSamples = 4480;
    int _hopPhase = 0;
    uint64_t _rtPos = 0; // absolute sample clock (samples processed)
    std::vector<float> _scratchL, _scratchR;
    bool _preparedOk = false;
    bool _elkAsyncActive = false;
    bool _syncWorker = false;

#if ELKGRANULAR_WITH_ELK_ASYNC
    std::unique_ptr<ElkAsyncBridge> _bridge;
#endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ElkGranularProcessor)
};
