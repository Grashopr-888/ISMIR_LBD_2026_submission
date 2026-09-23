// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/ElkAsync.h (changed: paths in comments)
// Bridge to the elk-plugin-extensions VST3 async-work extension
// (github.com/elk-audio/elk-plugin-extensions, fetched by CMake at commit
// 23edab4, not redistributed). Compiled only when
// ELKGRANULAR_WITH_ELK_ASYNC=1; all Steinberg types stay inside ElkAsync.cpp.
//
// Flow (sushi 1.3.0, verified against SUSHI's source):
//   RT: bridge->requestWorkFromRt()
//     -> IElkHostExtension::requestAsyncWork (gated on the calling thread
//        being flagged RT — AudioEngine::process_chunk sets twine's
//        ThreadRtFlag for every frontend, CoreAudio included)
//     -> lock-free RtEvent to sushi's dispatcher -> Worker thread (1 ms poll)
//        runs our callback -> WorkerCore::drainPendingHops()
//     -> completion RtEvent -> IElkProcessorExtension::asyncWorkCompleted on
//        the audio thread (sushi queried our IAudioProcessor for the
//        extension at load: "Plugin supports Elk Processor Extension").
#pragma once

#if ELKGRANULAR_WITH_ELK_ASYNC

#include <memory>
#include <juce_audio_processors/juce_audio_processors.h>

class WorkerCore;

class ElkAsyncBridge
{
public:
    virtual ~ElkAsyncBridge() = default;

    // true once setIHostApplication delivered a context exposing
    // IElkHostExtension (i.e. we are inside sushi).
    virtual bool hostSupportsAsyncWork() const noexcept = 0;

    // RT-safe: files one async-work request. false if unsupported/failed.
    virtual bool requestWorkFromRt() noexcept = 0;

    // Hook for AudioProcessor::getVST3ClientExtensions().
    virtual juce::VST3ClientExtensions* clientExtensions() noexcept = 0;

    static std::unique_ptr<ElkAsyncBridge> create (WorkerCore& worker);
};

#endif // ELKGRANULAR_WITH_ELK_ASYNC
