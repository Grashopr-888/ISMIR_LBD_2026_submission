// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/MelFrontend.h (changed: paths in comments)
// T09 analysis frontend: 48 kHz hop -> feat89 on the student's training grid.
//
// The T06 student (and ridge map) were trained on features from
// distillation/features.py: 44.1 kHz audio, 4096-sample start-aligned
// windows on the ~93 ms codec frame grid, np.hanning window, rFFT power,
// librosa mel (slaney scale + slaney norm, 80 bands, 20 Hz..16 kHz),
// logmel = ln(mel_power + 1e-10), plus 9 scalar spectral features
// (= feat89; the student consumes feat89[0:80] = the log-mels).
//
// A mismatched frontend silently degrades the student (T06 finding), so this
// class reproduces that pipeline: the 48 kHz input hop stream is resampled
// 147/160 to 44.1 kHz by a streaming polyphase FIR designed exactly like
// scipy.signal.resample_poly's default (firwin kaiser beta=5.0,
// 2*10*160+1 taps, cutoff 1/160, DC-normalised, *up) and each 4480-sample
// hop yields exactly 4116 samples at 44.1 kHz, of which the first 4096 form
// the analysis window (training windows are start-aligned with hop ~4106 and
// length 4096 — same shape). The resampler runs continuously across hops
// (state carried), so there are no per-hop filter transients. Constant
// group-delay (~0.23 ms) is not compensated — irrelevant at 93 ms granularity.
//
// Worker-thread only. No RT use. Allocation only in prepare().
#pragma once

#include <juce_dsp/juce_dsp.h>

#include <cstdint>
#include <vector>

class MelFrontend
{
public:
    static constexpr int kNFft = 4096;
    static constexpr int kNumBins = kNFft / 2 + 1;    // 2049
    static constexpr int kNumMels = 80;
    static constexpr int kFeatDim = 89;               // [mel80, logrms, centroid, bw,
                                                      //  skew, kurt, logflat, roll85,
                                                      //  flux, zcr]
    static constexpr int kUp = 147, kDown = 160;      // 48000 * 147/160 = 44100
    static constexpr int kOut44PerHop = 4116;         // 4480 * 147/160 exactly

    // hopIn must be 4480 (48 kHz). Allocates everything.
    void prepare (int hopInSamples);
    void reset();

    // Consume one 48 kHz hop, produce feat89 (float32). Streaming state
    // (resampler history, spectral-flux previous spectrum) is carried.
    void processHop (const float* hop48, float* feat89Out);

    // Golden-test entry: analysis stage only, from a raw 44.1 kHz
    // 4096-sample window (exactly what processHop runs after resampling).
    void analyzeWindow44 (const float* win4096, float* feat89Out);

    // Golden-test entry: run only the resampler on a buffer.
    // Returns number of output samples written (out must hold n*kUp/kDown+8).
    int resampleOnly (const float* in48, int n, float* out44);

    // librosa.filters.mel(sr=44100, n_fft=4096, n_mels=80, fmin=20,
    // fmax=16000, htk=False, norm='slaney') — computed in double, cast f32.
    // fbOut is [kNumMels][kNumBins] row-major.
    static void buildMelFilterbank (std::vector<float>& fbOut);

private:
    void designResampler();
    void feedResampler (float x);   // may emit 0..2 outputs into _fifo44

    // resampler
    std::vector<double> _taps;            // 3201 taps, already *kUp
    static constexpr int kHistLen = 32;   // >= ceil(3201/147)+1 = 23, pow2
    double _hist[kHistLen] {};
    uint64_t _inCount = 0;                // inputs consumed
    // up-domain position of the next output, pos = _outQ*kUp + _outR, kept as quotient and remainder so
    // feedResampler never divides a 64-bit number (a software library call on the 32-bit board; 16 Sep 2026)
    uint64_t _outQ = 0;                   // floor(pos / kUp): the newest input this output needs
    int _outR = 0;                        // pos % kUp, 0..kUp-1
    std::vector<float> _fifo44;           // output FIFO (ring)
    size_t _fifoW = 0, _fifoR = 0, _fifoMask = 0;

    // analysis
    int _hopIn = 4480;
    std::vector<float> _win44;            // kOut44PerHop staging
    std::vector<float> _hann;             // np.hanning(4096) (symmetric)
    std::vector<float> _fftBuf;           // 2*kNFft for juce FFT
    std::vector<float> _power;            // kNumBins
    std::vector<float> _pn;               // normalised spectrum (flux)
    std::vector<float> _pnPrev;
    bool _havePrev = false;
    std::vector<float> _fb;               // [80][2049]
    std::vector<int> _fbStart, _fbEnd;    // nonzero span per band
    std::vector<float> _freqsKhz;         // kNumBins
    std::unique_ptr<juce::dsp::FFT> _fft;
};
