# Provenance: Grashopr-888/elk-mutek@3baef53:tools/crossbuild/Dockerfile (verbatim)
# Elk Stomp cross-build image WITHOUT the vendor SDK installer.
# Host: Ubuntu 24.04 arm64 (runs natively on Apple Silicon). Target: armv7hf, glibc 2.39,
# GCC 13 — the same glibc/libstdc++ generation as the board's Elk Audio OS v1.2.1 (scarthgap).
# JUCE 8.0.12 = the version in Elk's SDK sysroot, installed the way Elk does it (host juceaide
# + module sources + CMake config) so plugins use find_package(JUCE CONFIG).
FROM ubuntu:24.04   # build with: docker build --platform linux/arm64 …
ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture armhf && apt-get update -q && apt-get install -qy --no-install-recommends \
      build-essential g++-13-arm-linux-gnueabihf gcc-13-arm-linux-gnueabihf binutils-arm-linux-gnueabihf \
      cmake ninja-build pkg-config git ca-certificates wget unzip file \
      libfreetype-dev libfontconfig-dev libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libxrender-dev libasound2-dev zlib1g-dev \
      libfreetype-dev:armhf libfontconfig-dev:armhf libx11-dev:armhf libxext-dev:armhf libxrandr-dev:armhf libxinerama-dev:armhf libxcursor-dev:armhf libxrender-dev:armhf libasound2-dev:armhf zlib1g-dev:armhf \
    && rm -rf /var/lib/apt/lists/*
ARG JUCE_URL="https://github.com/juce-framework/JUCE/releases/download/8.0.12/juce-8.0.12-linux.zip"
RUN wget -qO /tmp/juce.zip "$JUCE_URL" && unzip -q /tmp/juce.zip -d /opt && rm /tmp/juce.zip \
    && cmake -S /opt/JUCE -B /tmp/juce-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DJUCE_BUILD_EXTRAS=OFF -DJUCE_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/opt/juce-host \
    && cmake --build /tmp/juce-build && cmake --install /tmp/juce-build && rm -rf /tmp/juce-build
COPY toolchain-stomp.cmake /opt/toolchain-stomp.cmake
WORKDIR /work
