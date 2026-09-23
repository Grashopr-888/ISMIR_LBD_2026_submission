// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/ElkAsync.cpp (verbatim)
#if ELKGRANULAR_WITH_ELK_ASYNC

#include "ElkAsync.h"

#include <atomic>

// Steinberg pluginterfaces come from JUCE's bundled VST3 SDK copy (see
// CMakeLists include dirs); the elk header pulls funknown.h + vsttypes.h.
#include <elk_vst3_extensions/elk_vst3_extensions.h>

#include "WorkerCore.h"

// The FUID storage for the elk interfaces must be defined in exactly one TU
// on the plugin side (sushi does the same on the host side).
DEF_CLASS_IID (elk::IElkHostExtension)
DEF_CLASS_IID (elk::IElkProcessorExtension)

namespace
{

class ElkAsyncBridgeImpl final : public ElkAsyncBridge,
                                 public juce::VST3ClientExtensions,
                                 public elk::IElkProcessorExtension
{
public:
    explicit ElkAsyncBridgeImpl (WorkerCore& worker) : _worker (worker) {}

    ~ElkAsyncBridgeImpl() override
    {
        if (auto* h = _hostExt.exchange (nullptr))
            h->release();
    }

    // ---- ElkAsyncBridge --------------------------------------------------
    bool hostSupportsAsyncWork() const noexcept override
    {
        return _hostExt.load (std::memory_order_acquire) != nullptr;
    }

    bool requestWorkFromRt() noexcept override
    {
        auto* h = _hostExt.load (std::memory_order_acquire);
        if (h == nullptr)
            return false;
        Steinberg::int32 requestId = 0;
        return h->requestAsyncWork (&ElkAsyncBridgeImpl::workCallback, this, requestId)
               == Steinberg::kResultOk;
    }

    juce::VST3ClientExtensions* clientExtensions() noexcept override { return this; }

    // ---- juce::VST3ClientExtensions -------------------------------------
    void setIHostApplication (Steinberg::FUnknown* host) override
    {
        if (host == nullptr || _hostExt.load() != nullptr)
            return;
        void* out = nullptr;
        if (host->queryInterface (elk::IElkHostExtension::iid.toTUID(), &out) == Steinberg::kResultOk
            && out != nullptr)
        {
            _hostExt.store (static_cast<elk::IElkHostExtension*> (out), std::memory_order_release);
        }
    }

    int32_t queryIAudioProcessor (const Steinberg::TUID tuid, void** obj) override
    {
        if (Steinberg::FUnknownPrivate::iidEqual (tuid, elk::IElkProcessorExtension::iid.toTUID()))
        {
            *obj = static_cast<elk::IElkProcessorExtension*> (this);
            addRef();
            return Steinberg::kResultOk;
        }
        *obj = nullptr;
        return Steinberg::kNoInterface;
    }

    // ---- elk::IElkProcessorExtension (FUnknown) --------------------------
    Steinberg::tresult PLUGIN_API queryInterface (const Steinberg::TUID tuid, void** obj) override
    {
        if (Steinberg::FUnknownPrivate::iidEqual (tuid, elk::IElkProcessorExtension::iid.toTUID())
            || Steinberg::FUnknownPrivate::iidEqual (tuid, Steinberg::FUnknown::iid.toTUID()))
        {
            *obj = static_cast<elk::IElkProcessorExtension*> (this);
            addRef();
            return Steinberg::kResultOk;
        }
        *obj = nullptr;
        return Steinberg::kNoInterface;
    }

    // Statically owned by the processor — refcounting is a no-op.
    Steinberg::uint32 PLUGIN_API addRef() override  { return 1000; }
    Steinberg::uint32 PLUGIN_API release() override { return 1000; }

    void PLUGIN_API propertyValueChanged (Steinberg::int32, const elk::PropertyValue&) override {}

    // Runs on the AUDIO thread, before process(): keep it trivial.
    void PLUGIN_API asyncWorkCompleted (Steinberg::int32, Steinberg::int32) override
    {
        _worker.noteAsyncCompletion();
    }

private:
    // Runs on sushi's dispatcher Worker thread (non-RT).
    static Steinberg::int32 workCallback (void* data, Steinberg::uint16 /*requestId*/)
    {
        auto* self = static_cast<ElkAsyncBridgeImpl*> (data);
        return static_cast<Steinberg::int32> (self->_worker.drainPendingHops());
    }

    WorkerCore& _worker;
    std::atomic<elk::IElkHostExtension*> _hostExt { nullptr };
};

} // namespace

std::unique_ptr<ElkAsyncBridge> ElkAsyncBridge::create (WorkerCore& worker)
{
    return std::make_unique<ElkAsyncBridgeImpl> (worker);
}

#endif // ELKGRANULAR_WITH_ELK_ASYNC
