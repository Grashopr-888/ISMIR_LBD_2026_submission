// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/RtGuard.h (verbatim)
// RT-hygiene guard: in Debug builds (ELKGRANULAR_RT_ALLOC_GUARD=1) any
// operator new/delete executed while an RtSection is alive on the current
// thread trips an assert. Scope covers OUR processBlock body (everything the
// JUCE VST3 wrapper does around it is host code and not ours to guard).
// The global operator new/delete overrides live in PluginProcessor.cpp.
#pragma once

#if ELKGRANULAR_RT_ALLOC_GUARD

namespace rtguard
{
    // >0 while the current thread is inside an RT section.
    extern thread_local int depth;

    struct RtSection
    {
        RtSection() noexcept  { ++depth; }
        ~RtSection() noexcept { --depth; }
        RtSection (const RtSection&) = delete;
        RtSection& operator= (const RtSection&) = delete;
    };
}

#define ELKGRANULAR_RT_SECTION ::rtguard::RtSection rtSectionGuard_

#else

#define ELKGRANULAR_RT_SECTION do {} while (false)

#endif
