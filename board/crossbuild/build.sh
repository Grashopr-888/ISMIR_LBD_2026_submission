#!/bin/bash -e
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/crossbuild/build.sh (changed: repository paths)
# Cross-compile ElkGranular.vst3 for the Elk Stomp in the board/crossbuild Docker image.
#   board/crossbuild/build.sh            # Release build -> plugin/build-stomp/
# Prereq (once, ~10 min): docker build --platform linux/arm64 -t elk-stomp-crossbuild board/crossbuild
# RTNeural and Elk's extension header are fetched by CMake inside the container (git, network).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLUGIN="$ROOT/plugin"
docker run --rm --platform linux/arm64 -v "$PLUGIN":/work -w /work elk-stomp-crossbuild bash -lc '
  set -e
  # The host-installed JUCE package is tagged 64-bit; CMake would refuse it for the 32-bit target.
  # The Elk SDK installs JUCE into the target sysroot with a host juceaide, the same layout as here,
  # so drop the pointer-size check (juceaide is a host tool, the modules are sources).
  # NOTE: this block runs inside bash -lc with single quotes: no apostrophes in comments here.
  sed -i "s/set(PACKAGE_VERSION_UNSUITABLE TRUE)/# arch check disabled: host juceaide + target modules/" \
      /opt/juce-host/lib/cmake/JUCE-8.0.12/JUCEConfigVersion.cmake
  cmake -S . -B build-stomp -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain-stomp.cmake \
        -DCMAKE_BUILD_TYPE=Release -DELKGRANULAR_SYSTEM_JUCE=ON -DCMAKE_PREFIX_PATH=/opt/juce-host \
        -DELKGRANULAR_BUILD_GOLDEN=OFF
  cmake --build build-stomp --target ElkGranular_VST3 elkgrain_bench -j"$(nproc)"
  SO=$(find build-stomp/ElkGranular_artefacts -name "ElkGranular.so" | head -1)
  arm-linux-gnueabihf-strip --strip-unneeded "$SO"
  echo "== $SO"; file "$SO"
  arm-linux-gnueabihf-nm -gD "$SO" | grep -c GetPluginFactory | sed "s/^/GetPluginFactory exports: /"
  arm-linux-gnueabihf-objdump -p "$SO" | grep NEEDED
  ls -la "$(dirname "$SO")"
  # E3 microbench (JUCE-free, see plugin/CMakeLists.txt): board-side counterpart of the STAGES timers
  BENCH=build-stomp/elkgrain_bench
  arm-linux-gnueabihf-strip --strip-unneeded "$BENCH"
  echo "== $BENCH"; file "$BENCH"
  arm-linux-gnueabihf-objdump -p "$BENCH" | grep NEEDED
  ls -la "$BENCH"
'
