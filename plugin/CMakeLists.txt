# Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/CMakeLists.txt (changed: RTNeural and Elk's extension header fetched at pinned commits; paths)
# ElkGranular: the latent granular instrument as a VST3 plugin for SUSHI on the Elk Stomp (and the desktop).
#
# Desktop build (this Mac):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Board builds (Elk Stomp, armv7l) — same file, ELKGRANULAR_SYSTEM_JUCE=ON:
#   cross, no vendor SDK:  board/crossbuild/build.sh   (Docker, Ubuntu armhf GCC 13 + JUCE 8.0.12)
#   native ON the board:   cmake -B /udata/build -DELKGRANULAR_SYSTEM_JUCE=ON -DELKGRANULAR_BUILD_GOLDEN=OFF
#                          (the dev image ships gcc 13.4, cmake 3.28 and JUCE 8.0.12 in /usr/lib/cmake)
#   vendor SDK:            source environment-setup-cortexa7t2hf-neon-vfpv4-elk-linux-gnueabi, then as cross
cmake_minimum_required(VERSION 3.24)
project(ElkGranular VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "" FORCE)

# Compile-in the elk-plugin-extensions requestAsyncWork worker path (mode is
# still chosen at runtime via ELKGRANULAR_WORKER_MODE=thread|elkasync).
option(ELKGRANULAR_WITH_ELK_ASYNC "Build the elk requestAsyncWork worker path" ON)

# ---------------------------------------------------------------------------
# JUCE 8 — three sources, one target graph:
#   default (desktop)            FetchContent, pinned 8.0.8
#   ELKGRANULAR_SYSTEM_JUCE=ON   find_package(JUCE CONFIG): the Elk SDK sysroot, the board's own
#                                /usr/lib/cmake/JUCE-8.0.12, or board/crossbuild's /opt/juce-host
#   JUCE_SOURCE_DIR=<tree>       a pre-fetched source tree (offline)
# ---------------------------------------------------------------------------
option(ELKGRANULAR_SYSTEM_JUCE "Use an installed JUCE package (Elk SDK / board / crossbuild image)" OFF)
if(ELKGRANULAR_SYSTEM_JUCE)
    find_package(JUCE CONFIG REQUIRED)
    set(ELKGRANULAR_VST3_SDK_DIR ${JUCE_MODULES_DIR}/juce_audio_processors/format_types/VST3_SDK)
    message(STATUS "ElkGranular: system JUCE ${JUCE_VERSION} (modules: ${JUCE_MODULES_DIR})")
elseif(DEFINED JUCE_SOURCE_DIR)
    add_subdirectory(${JUCE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/juce EXCLUDE_FROM_ALL)
    set(ELKGRANULAR_VST3_SDK_DIR ${JUCE_SOURCE_DIR}/modules/juce_audio_processors/format_types/VST3_SDK)
elseif(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Cross build: pass -DELKGRANULAR_SYSTEM_JUCE=ON (installed JUCE) or -DJUCE_SOURCE_DIR=<tree>")
else()
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.8            # pin: JUCE 8.0.x
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(JUCE)
    set(ELKGRANULAR_VST3_SDK_DIR ${juce_SOURCE_DIR}/modules/juce_audio_processors/format_types/VST3_SDK)
endif()

# LTO: on for the desktop; off by default for board/cross builds (link memory on a 1 GB
# board, plain-ar toolchains) until measured there.
if(CMAKE_CROSSCOMPILING OR ELKGRANULAR_SYSTEM_JUCE)
    set(_elkgranular_lto_default OFF)
else()
    set(_elkgranular_lto_default ON)
endif()
option(ELKGRANULAR_LTO "Link-time optimisation" ${_elkgranular_lto_default})

# ---------------------------------------------------------------------------
# RTNeural (student MLP inference), BSD-3-Clause, pinned to the commit the paper's
# build used (95c3c0f). Fetched at configure time (submodules skipped: the STL
# backend needs neither xsimd nor Eigen; 0.103 M MACs at 10.7 Hz needs no SIMD),
# or taken from -DRTNEURAL_SOURCE_DIR=<checkout> for offline and board builds.
# Its vendored nlohmann json is used by the tolerant weights loader.
# ---------------------------------------------------------------------------
set(ELKGRANULAR_RTNEURAL_COMMIT 95c3c0f987a6fe903e7eec71e797405dbed7caf7)
if(NOT DEFINED RTNEURAL_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(rtneural
        GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
        GIT_TAG        ${ELKGRANULAR_RTNEURAL_COMMIT}
        GIT_SUBMODULES ""
        SOURCE_SUBDIR  populate-only)   # no CMakeLists.txt there: populate, add_subdirectory below
    FetchContent_MakeAvailable(rtneural)
    set(RTNEURAL_SOURCE_DIR ${rtneural_SOURCE_DIR})
endif()
if(NOT EXISTS ${RTNEURAL_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "RTNeural tree missing at ${RTNEURAL_SOURCE_DIR}: clone "
                        "https://github.com/jatinchowdhury18/RTNeural at ${ELKGRANULAR_RTNEURAL_COMMIT}")
endif()
set(RTNEURAL_STL ON CACHE BOOL "" FORCE)
add_subdirectory(${RTNEURAL_SOURCE_DIR} ${CMAKE_BINARY_DIR}/rtneural EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Plugin target. NOTE the SUSHI contracts:
#   * sushi addresses VST3 params by DISPLAY NAME -> our display names are the
#     snake_case ids verbatim (wet, tau, gate, grain_gain, bar_seconds).
#   * sushi config uid = the VST3 factory class name = PRODUCT_NAME here.
#   * VST3_AUTO_MANIFEST FALSE per the Elk pattern (no moduleinfo.json step;
#     also required for cross builds where the binary can't run on the host).
# ---------------------------------------------------------------------------
juce_add_plugin(ElkGranular
    COMPANY_NAME                "mutek-latent"
    PRODUCT_NAME                "ElkGranular"
    PLUGIN_MANUFACTURER_CODE    Mltk
    PLUGIN_CODE                 Elkg
    FORMATS                     VST3
    IS_SYNTH                    FALSE
    NEEDS_MIDI_INPUT            FALSE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    VST3_AUTO_MANIFEST          FALSE)

target_sources(ElkGranular PRIVATE
    src/PluginProcessor.cpp
    src/WorkerCore.cpp
    src/ElkgrainPack.cpp
    src/MelFrontend.cpp
    src/Embedder.cpp
    src/Matcher.cpp)

target_compile_definitions(ElkGranular PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    # RT hygiene: debug builds assert on any operator new/delete inside the
    # processBlock RT section (see RtGuard.h).
    ELKGRANULAR_RT_ALLOC_GUARD=$<IF:$<CONFIG:Debug>,1,0>)

target_include_directories(ElkGranular PRIVATE
    ${RTNEURAL_SOURCE_DIR}/modules/json)

# 32-bit ARM + -D_FILE_OFFSET_BITS=64 (Yocto's default, and the Elk SDK's): JUCE's bundled
# zlib turns adler32_combine/crc32_combine into duplicate *_combine64 definitions (a hackathon
# teammate's field note). Use the sysroot's zlib instead — every board/cross sysroot has it.
if(CMAKE_CROSSCOMPILING OR ELKGRANULAR_SYSTEM_JUCE)
    target_compile_definitions(ElkGranular PRIVATE JUCE_INCLUDE_ZLIB_CODE=0)
    target_link_libraries(ElkGranular PRIVATE z)
endif()

if(ELKGRANULAR_WITH_ELK_ASYNC)
    # elk_vst3_extensions.h (Copyright Elk Audio AB) is not redistributed here: it is fetched from
    # https://github.com/elk-audio/elk-plugin-extensions at the commit the paper's build used
    # (23edab4, byte-identical), or taken from -DELK_PLUGIN_EXTENSIONS_DIR=<checkout> for offline
    # and board builds. Steinberg pluginterfaces come from JUCE's bundled VST3 SDK copy.
    if(NOT DEFINED ELK_PLUGIN_EXTENSIONS_DIR)
        include(FetchContent)
        FetchContent_Declare(elk_plugin_extensions
            GIT_REPOSITORY https://github.com/elk-audio/elk-plugin-extensions.git
            GIT_TAG        23edab4c38813bb6966304d613fa4f32fb866030
            SOURCE_SUBDIR  populate-only)   # header only: populate, include it below
        FetchContent_MakeAvailable(elk_plugin_extensions)
        set(ELK_PLUGIN_EXTENSIONS_DIR ${elk_plugin_extensions_SOURCE_DIR})
    endif()
    if(NOT EXISTS ${ELK_PLUGIN_EXTENSIONS_DIR}/include/elk_vst3_extensions/elk_vst3_extensions.h)
        message(FATAL_ERROR "elk_vst3_extensions.h missing under ${ELK_PLUGIN_EXTENSIONS_DIR}/include")
    endif()
    target_sources(ElkGranular PRIVATE src/ElkAsync.cpp)
    target_compile_definitions(ElkGranular PRIVATE
        ELKGRANULAR_WITH_ELK_ASYNC=1
        $<IF:$<CONFIG:Debug>,DEVELOPMENT=1,RELEASE=1>)
    target_include_directories(ElkGranular PRIVATE
        ${ELK_PLUGIN_EXTENSIONS_DIR}/include
        ${ELKGRANULAR_VST3_SDK_DIR})
endif()

target_link_libraries(ElkGranular
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        RTNeural
    PUBLIC
        juce::juce_recommended_config_flags
        $<$<BOOL:${ELKGRANULAR_LTO}>:juce::juce_recommended_lto_flags>
        juce::juce_recommended_warning_flags)

# ---------------------------------------------------------------------------
# elkgrain_golden — offline parity CLI (C++ stages vs corpus_builder/match.py,
# distillation/features.py, the student JSON and distillation/ridgefit.py). Desktop by default; also buildable
# natively on the board (ELKGRANULAR_BUILD_GOLDEN=ON) — never in a cross build.
# ---------------------------------------------------------------------------
if(CMAKE_CROSSCOMPILING)
    set(_elkgranular_golden_default OFF)
else()
    set(_elkgranular_golden_default ON)
endif()
option(ELKGRANULAR_BUILD_GOLDEN "Build the elkgrain_golden parity CLI" ${_elkgranular_golden_default})
if(ELKGRANULAR_BUILD_GOLDEN)
    juce_add_console_app(ElkgrainGolden PRODUCT_NAME "elkgrain_golden")
    target_sources(ElkgrainGolden PRIVATE
        src/GoldenMain.cpp
        src/ElkgrainPack.cpp
        src/MelFrontend.cpp
        src/Embedder.cpp
        src/Matcher.cpp)
    target_compile_definitions(ElkgrainGolden PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0)
    target_include_directories(ElkgrainGolden PRIVATE
        ${RTNEURAL_SOURCE_DIR}/modules/json)
    target_link_libraries(ElkgrainGolden PRIVATE
        juce::juce_dsp
        RTNeural
        juce::juce_recommended_config_flags)
endif()

# ---------------------------------------------------------------------------
# elkgrain_bench — microbench for the board session (E3 student µs/frame; Matcher::selectOne per
# pack, the E2b cross-check). JUCE-free — Embedder/Matcher/ElkgrainPack use none — so unlike
# elkgrain_golden it IS built in cross builds (board/crossbuild/build.sh) and runs on the Stomp.
# -O3 = the plugin's Release level (juce_recommended_config_flags); the toolchain file carries
# -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard.
# ---------------------------------------------------------------------------
option(ELKGRANULAR_BUILD_BENCH "Build the elkgrain_bench microbench CLI" ON)
if(ELKGRANULAR_BUILD_BENCH)
    add_executable(elkgrain_bench
        src/BenchMain.cpp
        src/ElkgrainPack.cpp
        src/Embedder.cpp
        src/Matcher.cpp)
    target_include_directories(elkgrain_bench PRIVATE
        ${RTNEURAL_SOURCE_DIR}/modules/json)
    target_link_libraries(elkgrain_bench PRIVATE RTNeural)
    target_compile_options(elkgrain_bench PRIVATE $<$<CONFIG:Release>:-O3>)
    set_target_properties(elkgrain_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
