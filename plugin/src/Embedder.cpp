// Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/plugin/src/Embedder.cpp (verbatim)
#include "Embedder.h"

#include <cstdio>
#include <fstream>
#include <vector>

#include <RTNeural/RTNeural.h>
#include <json.hpp> // nlohmann, vendored inside RTNeural (modules/json)

using nlohmann::json;

struct Embedder::Impl
{
    using StudentNet = RTNeural::ModelT<float, Embedder::kMelDim, Embedder::kLatentDim,
                                        RTNeural::DenseT<float, 80, 256>,
                                        RTNeural::ReLuActivationT<float, 256>,
                                        RTNeural::DenseT<float, 256, 256>,
                                        RTNeural::ReLuActivationT<float, 256>,
                                        RTNeural::DenseT<float, 256, 64>>;

    std::unique_ptr<StudentNet> student;
    std::array<float, Embedder::kMelDim> stuMean {}, stuStd {};
    bool studentOk = false;

    // ridge: y = ((x - mu)/sd) @ W[89][64] + b
    std::array<float, Embedder::kFeatDim> ridgeMu {}, ridgeSd {};
    std::vector<float> ridgeW; // [89][64] row-major (input-major, as exported)
    std::array<float, Embedder::kLatentDim> ridgeB {};
    bool ridgeOk = false;

    std::array<float, Embedder::kMelDim> xStd {}; // scratch
    std::array<float, Embedder::kFeatDim> xStd89 {};
};

Embedder::Embedder() : _impl (std::make_unique<Impl>()) {}
Embedder::~Embedder() = default;

bool Embedder::studentReady() const noexcept { return _impl->studentOk; }
bool Embedder::ridgeReady() const noexcept { return _impl->ridgeOk; }

namespace
{
    bool readJsonFile (const std::string& path, json& out, std::string& err)
    {
        std::ifstream f (path);
        if (! f.good())
        {
            err = path + ": cannot open";
            return false;
        }
        try
        {
            f >> out;
        }
        catch (const std::exception& e)
        {
            err = path + ": JSON parse failed: " + e.what();
            return false;
        }
        return true;
    }

    template <typename Vec>
    bool readFloatArray (const json& j, size_t expected, Vec& out, std::string& err,
                         const char* what)
    {
        if (! j.is_array() || j.size() != expected)
        {
            err = std::string (what) + ": expected array of " + std::to_string (expected);
            return false;
        }
        for (size_t i = 0; i < expected; ++i)
            out[i] = j[i].get<float>();
        return true;
    }
}

bool Embedder::loadStudent (const std::string& jsonPath, std::string& errOut)
{
    auto& im = *_impl;
    im.studentOk = false;

    json doc;
    if (! readJsonFile (jsonPath, doc, errOut))
        return false;

    // tolerant: only "layers" (3 dense) and input.standardize.{mean,std} matter
    if (! doc.contains ("layers") || ! doc["layers"].is_array() || doc["layers"].size() != 3)
    {
        errOut = jsonPath + ": expected exactly 3 layers (mel80->256->256->64)";
        return false;
    }

    const int inSizes[3] = { 80, 256, 256 };
    const int outSizes[3] = { 256, 256, 64 };

    auto net = std::make_unique<Impl::StudentNet>();

    for (int li = 0; li < 3; ++li)
    {
        const auto& L = doc["layers"][static_cast<size_t> (li)];
        if (L.value ("type", "") != std::string ("dense"))
        {
            errOut = jsonPath + ": layer " + std::to_string (li) + " is not dense";
            return false;
        }
        const auto& w = L["weights"];
        if (! w.is_array() || w.size() < 2)
        {
            errOut = jsonPath + ": layer weights must be [kernel, bias]";
            return false;
        }
        const auto& kernel = w[0]; // [in][out]
        const auto& bias = w[1];   // [out]
        const int nIn = inSizes[li], nOut = outSizes[li];
        if (static_cast<int> (kernel.size()) != nIn
            || static_cast<int> (kernel[0].size()) != nOut
            || static_cast<int> (bias.size()) != nOut)
        {
            errOut = jsonPath + ": layer " + std::to_string (li) + " shape mismatch (want ["
                     + std::to_string (nIn) + "][" + std::to_string (nOut) + "])";
            return false;
        }

        // RTNeural DenseT::setWeights wants [out][in]; JSON kernel is [in][out]
        std::vector<std::vector<float>> wt (static_cast<size_t> (nOut),
                                            std::vector<float> (static_cast<size_t> (nIn)));
        for (int i = 0; i < nIn; ++i)
            for (int o = 0; o < nOut; ++o)
                wt[static_cast<size_t> (o)][static_cast<size_t> (i)] =
                    kernel[static_cast<size_t> (i)][static_cast<size_t> (o)].get<float>();
        std::vector<float> b (static_cast<size_t> (nOut));
        for (int o = 0; o < nOut; ++o)
            b[static_cast<size_t> (o)] = bias[static_cast<size_t> (o)].get<float>();

        if (li == 0)      { net->get<0>().setWeights (wt); net->get<0>().setBias (b.data()); }
        else if (li == 1) { net->get<2>().setWeights (wt); net->get<2>().setBias (b.data()); }
        else              { net->get<4>().setWeights (wt); net->get<4>().setBias (b.data()); }
    }

    // input standardization (required by the training frontend)
    try
    {
        const auto& st = doc.at ("input").at ("standardize");
        if (! readFloatArray (st.at ("mean"), kMelDim, im.stuMean, errOut, "standardize.mean")
            || ! readFloatArray (st.at ("std"), kMelDim, im.stuStd, errOut, "standardize.std"))
            return false;
    }
    catch (const std::exception&)
    {
        errOut = jsonPath + ": missing input.standardize.{mean,std}";
        return false;
    }

    net->reset();
    im.student = std::move (net);
    im.studentOk = true;
    return true;
}

bool Embedder::loadRidge (const std::string& jsonPath, std::string& errOut)
{
    auto& im = *_impl;
    im.ridgeOk = false;

    json doc;
    if (! readJsonFile (jsonPath, doc, errOut))
        return false;

    try
    {
        if (! readFloatArray (doc.at ("mu"), kFeatDim, im.ridgeMu, errOut, "mu")
            || ! readFloatArray (doc.at ("sd"), kFeatDim, im.ridgeSd, errOut, "sd")
            || ! readFloatArray (doc.at ("b"), kLatentDim, im.ridgeB, errOut, "b"))
            return false;
        const auto& W = doc.at ("W"); // [89][64]
        if (static_cast<int> (W.size()) != kFeatDim
            || static_cast<int> (W[0].size()) != kLatentDim)
        {
            errOut = jsonPath + ": W must be [89][64]";
            return false;
        }
        im.ridgeW.assign (static_cast<size_t> (kFeatDim) * kLatentDim, 0.0f);
        for (int i = 0; i < kFeatDim; ++i)
            for (int o = 0; o < kLatentDim; ++o)
                im.ridgeW[static_cast<size_t> (i) * kLatentDim + o] =
                    W[static_cast<size_t> (i)][static_cast<size_t> (o)].get<float>();
    }
    catch (const std::exception& e)
    {
        errOut = jsonPath + ": " + e.what();
        return false;
    }

    im.ridgeOk = true;
    return true;
}

void Embedder::embed (Mode m, const float* feat89, float* latent64) noexcept
{
    auto& im = *_impl;

    if (m == Mode::Student && im.studentOk)
    {
        for (int i = 0; i < kMelDim; ++i)
            im.xStd[static_cast<size_t> (i)] =
                (feat89[i] - im.stuMean[static_cast<size_t> (i)])
                / im.stuStd[static_cast<size_t> (i)];
        im.student->forward (im.xStd.data());
        const float* y = im.student->getOutputs();
        for (int o = 0; o < kLatentDim; ++o)
            latent64[o] = y[o];
        return;
    }

    if (m == Mode::Ridge && im.ridgeOk)
    {
        for (int i = 0; i < kFeatDim; ++i)
            im.xStd89[static_cast<size_t> (i)] =
                (feat89[i] - im.ridgeMu[static_cast<size_t> (i)])
                / im.ridgeSd[static_cast<size_t> (i)];
        for (int o = 0; o < kLatentDim; ++o)
        {
            double acc = static_cast<double> (im.ridgeB[static_cast<size_t> (o)]);
            for (int i = 0; i < kFeatDim; ++i)
                acc += static_cast<double> (im.xStd89[static_cast<size_t> (i)])
                       * static_cast<double> (im.ridgeW[static_cast<size_t> (i) * kLatentDim + o]);
            latent64[o] = static_cast<float> (acc);
        }
        return;
    }

    // rawmel plumbing debug (also the fallback when weights failed to load)
    for (int o = 0; o < kLatentDim; ++o)
        latent64[o] = feat89[o];
}
