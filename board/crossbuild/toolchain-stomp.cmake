# Provenance: Grashopr-888/elk-mutek@3baef53:tools/crossbuild/toolchain-stomp.cmake (verbatim)
# CMake toolchain: Elk Stomp (STM32MP157, 2x Cortex-A7, NEON-VFPv4, hard-float), Yocto tune
# "cortexa7t2hf-neon-vfpv4". Matches the vendor SDK's flags; the board's own gcc reports
# -march=armv7ve+fp -mfloat-abi=hard, and its libs were built with 64-bit off_t/time_t.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)          # -> JUCE names the VST3 arch folder armv7l-linux
set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc-13)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++-13)
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf /usr/lib/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)   # JUCE's package lives in /opt/juce-host (host prefix)
set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} "")
# -Wno-psabi: silence the "parameter passing changed in GCC 7.1" notes (armhf ABI history, harmless)
set(STOMP_FLAGS "-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -mthumb -D_FILE_OFFSET_BITS=64 -D_TIME_BITS=64 -Wno-psabi")
set(CMAKE_C_FLAGS_INIT   "${STOMP_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${STOMP_FLAGS}")
