//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv)
#error These examples require a RISC-V target.
#endif

#include <torq_tile/tqt_compile_config.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

// Depthwise multiplier two: compare CF weights with CL HWIO weights.
#if TORQ_TILE_HAS_RVV
#include "torq_tile/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32_f32_f32/tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32_f32_f32/tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32_f32_f32t/tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32_f32_f32t/tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv.h"
#endif

namespace
{

// Keep reference computation and checks visible in this standalone teaching source.
float input_value(size_t index)
{
    return (static_cast<int>(index % 17) - 8) * 0.0625f;
}

// Canonical reference uses CF activations and [OC,IC/group,KH,KW] weights.
// Each layout below is prepared independently from these logical tensors.
struct Data
{
    tqt_conv_params p{};
    std::vector<float> cf, cl, wcf, wcl, bias, reference;
    Data(bool depthwise)
    {
        p.input_shape[0] = p.output_shape[0] = p.kernel_shape[0] = 1;
        p.input_shape[1] = p.output_shape[1] = 5;
        p.input_shape[2] = p.output_shape[2] = 7;
        p.kernel_shape[1] = p.kernel_shape[2] = 3;
        for (size_t d = 0; d < 3; ++d)
            p.stride[d] = p.dilation[d] = 1;
        p.pad[1] = p.pad[2] = 1;
        p.input_channel = 6;
        p.output_channel = depthwise ? 12 : 10;
        p.groups = depthwise ? p.input_channel : 2;
        const size_t ic = p.input_channel, oc = p.output_channel;
        const size_t spatial = 35, icg = ic / p.groups, ocg = oc / p.groups, k = icg * 9;
        cf.resize(ic * spatial);
        cl.resize(cf.size());
        wcf.resize(oc * k);
        wcl.resize(wcf.size());
        bias.resize(oc);
        reference.resize(oc * spatial);
        for (size_t c = 0; c < ic; ++c)
            for (size_t xy = 0; xy < spatial; ++xy)
                cf[c * spatial + xy] = cl[xy * ic + c] = input_value(c * spatial + xy);
        for (size_t c = 0; c < oc; ++c) {
            bias[c] = input_value(c);
            for (size_t q = 0; q < icg; ++q)
                for (size_t xy = 0; xy < 9; ++xy) {
                    const float value = input_value(c * k + q * 9 + xy + 5);
                    wcf[c * k + q * 9 + xy] = value;
                    // Conv/IGEMM CL uses [OC,KH,KW,IC/group]; DWConv CL uses [KH,KW,OC].
                    wcl[depthwise ? xy * oc + c : c * k + xy * icg + q] = value;
                }
            for (size_t y = 0; y < 5; ++y)
                for (size_t x = 0; x < 7; ++x) {
                    float sum = bias[c];
                    for (size_t q = 0; q < icg; ++q)
                        for (int ky = 0; ky < 3; ++ky)
                            for (int kx = 0; kx < 3; ++kx) {
                                const int iy = static_cast<int>(y) + ky - 1;
                                const int ix = static_cast<int>(x) + kx - 1;
                                if (iy >= 0 && iy < 5 && ix >= 0 && ix < 7) {
                                    const size_t channel = (c / ocg) * icg + q;
                                    sum = std::fma(cf[channel * spatial + iy * 7 + ix],
                                                   wcf[c * k + q * 9 + ky * 3 + kx], sum);
                                }
                            }
                    reference[c * spatial + y * 7 + x] = std::min(0.75f, std::max(-0.5f, sum));
                }
        }
    }
    bool check(const std::vector<float> &output, bool channel_last) const
    {
        for (size_t c = 0; c < p.output_channel; ++c)
            for (size_t xy = 0; xy < 35; ++xy) {
                const float value = output[channel_last ? xy * p.output_channel + c : c * 35 + xy];
                const float expected = reference[c * 35 + xy];
                if (!std::isfinite(value) || std::fabs(value - expected) > 1e-4f) {
                    std::fprintf(stderr, "Mismatch channel=%zu spatial=%zu: %g vs %g\n", c, xy,
                                 value, expected);
                    return false;
                }
            }
        return true;
    }
};

namespace channel_first
{
using Kernel = tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv",
     {tqt_get_output_size_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
      tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv}},
#endif
};

bool run(const Data &data)
{
    const auto &p = data.p;
    std::vector<float> output(data.reference.size());
    for (const auto &v : variants) {
        const auto &u = v.kernel;
        if (u.get_output_size(&p) != output.size() * sizeof(float))
            return false;
        u.run_dwconv(data.cf.data(), data.wcf.data(), data.bias.data(), output.data(), &p, -0.5f,
                     0.75f);
        std::printf("%s groups=%zu\n", v.name, p.groups);
        if (!data.check(output, false))
            return false;
    }
    return true;
}
}  // namespace channel_first

namespace channel_last
{
using Kernel = tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv",
     {tqt_get_output_size_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv,
      tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv}},
#endif
};

bool run(const Data &data)
{
    const auto &p = data.p;
    std::vector<float> output(data.reference.size());
    for (const auto &v : variants) {
        const auto &u = v.kernel;
        if (u.get_output_size(&p) != output.size() * sizeof(float))
            return false;
        u.run_dwconv(data.cl.data(), data.wcl.data(), data.bias.data(), output.data(), &p, -0.5f,
                     0.75f);
        std::printf("%s groups=%zu\n", v.name, p.groups);
        if (!data.check(output, true))
            return false;
    }
    return true;
}
}  // namespace channel_last

bool run_channel_first(const Data &data)
{
    return channel_first::run(data);
}
bool run_channel_last(const Data &data)
{
    return channel_last::run(data);
}

}  // namespace

int main()
{
    try {
        const Data data(true);
        const bool cf = run_channel_first(data), cl = run_channel_last(data);
        std::printf("Status: %s\n", cf && cl ? "PASSED" : "FAILED");
        return cf && cl ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
