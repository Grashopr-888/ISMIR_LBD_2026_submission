<!-- Provenance: Grashopr-888/elk-mutek@3baef53:tools/crossbuild/README.md (changed: paths, references) -->
# crossbuild — Elk Stomp plugin builds without the vendor SDK installer

Three ways to get an `armv7l-linux` VST3 for the Stomp; this folder is the first one.

| Route | What it needs | Notes |
|---|---|---|
| **Docker (this folder)** | Docker Desktop, ~10 min once | Ubuntu 24.04 arm64 image with `g++-13-arm-linux-gnueabihf` (glibc 2.39, libstdc++ 6.0.32 = the board's), armhf dev libs for JUCE (freetype, fontconfig, X11 headers, ALSA, zlib), JUCE 8.0.12 installed Elk-style: host `juceaide` + module sources + CMake config in `/opt/juce-host` |
| Native on the board | nothing extra | the dev image ships gcc 13.4, cmake 3.28, JUCE 8.0.12 + juceaide; build in `/udata` (rootfs ~200 MB free), `-j1`, no LTO, and it lacks freetype/fontconfig/X11 *headers* — untested |
| Vendor SDK | Elk's `elk-stomp-toolchain-*.sh` | source its environment script, then configure as for a cross build; colima at 8 GB |

```bash
# once
docker build --platform linux/arm64 -t elk-stomp-crossbuild board/crossbuild
# every time (Release, stripped, checks file/nm/NEEDED)
board/crossbuild/build.sh
# copy plugin + board configs + run script + weights (+ packs) over the USB-TO-UART console (~8 KB/s)
board/crossbuild/deploy_serial.sh [pack.elkgrain ...]
```

Then on the board: `/udata/elkgranular/run_board.sh` (sensei → `sushi -r`; Ctrl-C only). See `board/README.md`.

## Things that bit, so you don't have to rediscover them
- **JUCE's installed package is tagged 64-bit** and CMake refuses it for the 32-bit target.
  `build.sh` drops the pointer-size check in `JUCEConfigVersion.cmake` inside the container
  (the SDK avoids this by installing JUCE into the target sysroot with a host juceaide).
- **JUCE's bundled zlib does not compile on 32-bit with `-D_FILE_OFFSET_BITS=64`** (Yocto's and
  the SDK's default): `JUCE_INCLUDE_ZLIB_CODE=0` + link `z` (`plugin/CMakeLists.txt`, board/cross only).
- `-Wno-psabi` silences ~180 harmless "parameter passing changed in GCC 7.1" notes.
- The container script in `build.sh` runs under `bash -lc '…'` — no apostrophes inside it.
- Resulting binary (2026-09-02): ELF 32-bit ARM EABI5 hard-float, Thumb-2, VFPv4 + NEON,
  needs only libz / libfontconfig / libfreetype / libstdc++ (≤ GLIBCXX_3.4.32) — all on the board.
