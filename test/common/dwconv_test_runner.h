//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cfloat>
#include <cstring>
#include <string>
#include <vector>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "test/common/conv_test_runner.h"  // reuse ConvTestParams + ConvLayout
#include "test/common/data_generator.h"
#include "test/common/reference_dwconv.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// DWConv Test Function-Pointer Types
// ============================================================================
//
// 3 categories of dwconv micro-kernel ukernel struct:
//   1. NonPacked      : raw cf or cl input/output, only get_output_size + run_dwconv
//   2. GenericPacked  : packed (NC1DHWC0), with pack/unpack helpers + full-tensor run_dwconv
//   3. SpecPacked     : packed, with get_oh_step + sliced run_dwconv (oh_start/oh_end)
//
// All 3 are templated on ClampT (float for fp32 kernels, float16_t for fp16).
// ============================================================================

template <typename ClampT>
struct NonPackedDwconvFuncs
{
    using GetOutputSizeFunc = size_t (*)(const tqt_conv_params *);
    using RunDwconvFunc = void (*)(const void *input, const void *weight, const void *bias,
                                   void *output, const tqt_conv_params *params, ClampT clamp_min,
                                   ClampT clamp_max);

    GetOutputSizeFunc get_output_size;
    RunDwconvFunc run_dwconv;
};

template <typename ClampT>
struct GenericPackedDwconvFuncs
{
    using GetOutputSizeFunc = size_t (*)(const tqt_conv_params *);
    using GetSizeFunc = size_t (*)(const tqt_conv_params *);
    using RunPackFunc = void (*)(const void *src, void *dst, const tqt_conv_params *params);
    using RunDwconvFunc = void (*)(const void *packed_input, const void *packed_weight,
                                   const void *bias, void *packed_output,
                                   const tqt_conv_params *params, ClampT clamp_min,
                                   ClampT clamp_max);

    GetOutputSizeFunc get_output_size;
    GetSizeFunc get_packed_input_size;
    GetSizeFunc get_packed_weight_size;
    RunPackFunc run_pack_input;
    RunPackFunc run_pack_weight;
    RunPackFunc run_unpack_output;
    RunDwconvFunc run_dwconv;
};

template <typename ClampT>
struct SpecPackedDwconvFuncs
{
    using GetOutputSizeFunc = size_t (*)(const tqt_conv_params *);
    using GetSizeFunc = size_t (*)(const tqt_conv_params *);
    using GetOhStepFunc = size_t (*)(void);
    using RunPackFunc = void (*)(const void *src, void *dst, const tqt_conv_params *params);
    using RunDwconvFunc = void (*)(size_t oh_start, size_t oh_end, const void *packed_input,
                                   const void *packed_weight, const void *bias, void *packed_output,
                                   const tqt_conv_params *params, ClampT clamp_min,
                                   ClampT clamp_max);

    GetOutputSizeFunc get_output_size;
    GetSizeFunc get_packed_input_size;
    GetSizeFunc get_packed_weight_size;
    GetOhStepFunc get_oh_step;
    RunPackFunc run_pack_input;
    RunPackFunc run_pack_weight;
    RunPackFunc run_unpack_output;
    RunDwconvFunc run_dwconv;
};

// ============================================================================
// Helpers: weight / output element-counts (single batch)
// ============================================================================

inline size_t dwconv_input_elements(const tqt_conv_params &p)
{
    return p.input_channel * p.input_shape[0] * p.input_shape[1] * p.input_shape[2];
}

inline size_t dwconv_output_elements(const tqt_conv_params &p)
{
    return p.output_channel * p.output_shape[0] * p.output_shape[1] * p.output_shape[2];
}

inline size_t dwconv_weight_elements(const tqt_conv_params &p)
{
    // [OC, 1, KD, KH, KW] (cf) or [KD, KH, KW, 1, OC] (cl) — same total count.
    return p.output_channel * p.kernel_shape[0] * p.kernel_shape[1] * p.kernel_shape[2];
}

inline size_t dwconv_bias_elements(const tqt_conv_params &p)
{
    return p.output_channel;
}

// Layout helpers: convert between [C, D, H, W] (cf) and [D, H, W, C] (cl).
template <typename T>
inline void cf_to_cl(const T *src, T *dst, size_t C, size_t D, size_t H, size_t W)
{
    const size_t spatial = D * H * W;
    for (size_t pos = 0; pos < spatial; ++pos) {
        for (size_t c = 0; c < C; ++c) {
            dst[pos * C + c] = src[c * spatial + pos];
        }
    }
}

template <typename T>
inline void cl_to_cf(const T *src, T *dst, size_t C, size_t D, size_t H, size_t W)
{
    const size_t spatial = D * H * W;
    for (size_t pos = 0; pos < spatial; ++pos) {
        for (size_t c = 0; c < C; ++c) {
            dst[c * spatial + pos] = src[pos * C + c];
        }
    }
}

// Convert weight [OC, 1, KD, KH, KW] (cf) -> [KD, KH, KW, 1, OC] (cl).
template <typename T>
inline void weight_cf_to_cl(const T *src, T *dst, size_t OC, size_t KD, size_t KH, size_t KW)
{
    const size_t k_size = KD * KH * KW;
    for (size_t k = 0; k < k_size; ++k) {
        for (size_t oc = 0; oc < OC; ++oc) {
            dst[k * OC + oc] = src[oc * k_size + k];
        }
    }
}

// ============================================================================
// Runner: non-packed dwconv (covers cf and cl via ConvLayout argument)
// ============================================================================

template <typename T>
void run_nonpacked_dwconv_test(const NonPackedDwconvFuncs<T> &funcs, const ConvTestParams &tp,
                               ConvLayout layout)
{
    tqt_conv_params p = tp.to_conv_params();

    // dwconv hard constraint
    ASSERT_EQ(p.groups, p.input_channel);
    ASSERT_EQ(p.output_channel % p.input_channel, 0u);

    const size_t in_elems = dwconv_input_elements(p);
    const size_t wt_elems = dwconv_weight_elements(p);
    const size_t bs_elems = dwconv_bias_elements(p);
    const size_t out_elems = dwconv_output_elements(p);

    // Generate input/weight/bias in CF representation; convert to CL when needed.
    UniformRandomGenerator<T> gen(
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(-1),
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(1),
        /*seed=*/42);

    std::vector<T> input_cf(in_elems);
    std::vector<T> weight_cf(wt_elems);
    std::vector<T> bias(bs_elems);
    gen.fill_matrix(input_cf.data(), 1, in_elems);
    gen.fill_matrix(weight_cf.data(), 1, wt_elems);
    if (tp.has_bias) {
        gen.fill_matrix(bias.data(), 1, bs_elems);
    }
    const T *bias_ptr = tp.has_bias ? bias.data() : nullptr;

    // Reference: compute in matching layout
    std::vector<T> ref_output(out_elems, T(0));
    std::vector<T> actual_output(out_elems, T(0));

    if (layout == ConvLayout::kChannelFirst) {
        reference_dwconv<T>(input_cf.data(), weight_cf.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelFirst, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        funcs.run_dwconv(input_cf.data(), weight_cf.data(), bias_ptr, actual_output.data(), &p,
                         static_cast<T>(tp.clamp_min), static_cast<T>(tp.clamp_max));
    } else {
        // CL: convert input + weight, then run + reference both in CL representation
        std::vector<T> input_cl(in_elems);
        std::vector<T> weight_cl(wt_elems);
        cf_to_cl<T>(input_cf.data(), input_cl.data(), p.input_channel, p.input_shape[0],
                    p.input_shape[1], p.input_shape[2]);
        weight_cf_to_cl<T>(weight_cf.data(), weight_cl.data(), p.output_channel, p.kernel_shape[0],
                           p.kernel_shape[1], p.kernel_shape[2]);
        reference_dwconv<T>(input_cl.data(), weight_cl.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelLast, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        funcs.run_dwconv(input_cl.data(), weight_cl.data(), bias_ptr, actual_output.data(), &p,
                         static_cast<T>(tp.clamp_min), static_cast<T>(tp.clamp_max));
    }

    auto vr =
        verify_gemm_result<T>(ref_output.data(), actual_output.data(), out_elems,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(vr.passed) << vr.error_message << " (max_abs=" << vr.max_abs_error
                           << ", cos=" << vr.cosine_similarity << ")";
}

// ============================================================================
// Runner: generic packed dwconv
// ============================================================================

template <typename T>
void run_generic_packed_dwconv_test(const GenericPackedDwconvFuncs<T> &funcs,
                                    const ConvTestParams &tp, ConvLayout layout)
{
    tqt_conv_params p = tp.to_conv_params();

    ASSERT_EQ(p.groups, p.input_channel);
    ASSERT_EQ(p.output_channel % p.input_channel, 0u);

    const size_t in_elems = dwconv_input_elements(p);
    const size_t wt_elems = dwconv_weight_elements(p);
    const size_t bs_elems = dwconv_bias_elements(p);
    const size_t out_elems = dwconv_output_elements(p);

    UniformRandomGenerator<T> gen(
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(-1),
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(1),
        /*seed=*/42);

    std::vector<T> input_cf(in_elems);
    std::vector<T> weight_cf(wt_elems);
    std::vector<T> bias(bs_elems);
    gen.fill_matrix(input_cf.data(), 1, in_elems);
    gen.fill_matrix(weight_cf.data(), 1, wt_elems);
    if (tp.has_bias) {
        gen.fill_matrix(bias.data(), 1, bs_elems);
    }
    const T *bias_ptr = tp.has_bias ? bias.data() : nullptr;

    std::vector<T> ref_output(out_elems, T(0));
    std::vector<T> actual_output_cf(out_elems, T(0));

    // Source data in matching layout
    std::vector<T> input_cl, weight_cl;
    const T *src_input = nullptr;
    const T *src_weight = nullptr;
    if (layout == ConvLayout::kChannelFirst) {
        reference_dwconv<T>(input_cf.data(), weight_cf.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelFirst, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        src_input = input_cf.data();
        src_weight = weight_cf.data();
    } else {
        input_cl.resize(in_elems);
        weight_cl.resize(wt_elems);
        cf_to_cl<T>(input_cf.data(), input_cl.data(), p.input_channel, p.input_shape[0],
                    p.input_shape[1], p.input_shape[2]);
        weight_cf_to_cl<T>(weight_cf.data(), weight_cl.data(), p.output_channel, p.kernel_shape[0],
                           p.kernel_shape[1], p.kernel_shape[2]);
        reference_dwconv<T>(input_cl.data(), weight_cl.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelLast, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        src_input = input_cl.data();
        src_weight = weight_cl.data();
    }

    // Pack -> compute -> unpack
    const size_t packed_in_bytes = funcs.get_packed_input_size(&p);
    const size_t packed_wt_bytes = funcs.get_packed_weight_size(&p);
    const size_t packed_out_bytes = funcs.get_output_size(&p);

    std::vector<uint8_t> packed_in(packed_in_bytes);
    std::vector<uint8_t> packed_wt(packed_wt_bytes);
    std::vector<uint8_t> packed_out(packed_out_bytes);

    funcs.run_pack_input(src_input, packed_in.data(), &p);
    funcs.run_pack_weight(src_weight, packed_wt.data(), &p);
    funcs.run_dwconv(packed_in.data(), packed_wt.data(), bias_ptr, packed_out.data(), &p,
                     static_cast<T>(tp.clamp_min), static_cast<T>(tp.clamp_max));

    // Unpack into the layout matching reference
    std::vector<T> actual_output(out_elems, T(0));
    funcs.run_unpack_output(packed_out.data(), actual_output.data(), &p);

    auto vr =
        verify_gemm_result<T>(ref_output.data(), actual_output.data(), out_elems,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(vr.passed) << vr.error_message << " (max_abs=" << vr.max_abs_error
                           << ", cos=" << vr.cosine_similarity << ")";
}

// ============================================================================
// Runner: specialized packed dwconv (with oh_step slicing)
// ============================================================================

template <typename T>
void run_spec_packed_dwconv_test(const SpecPackedDwconvFuncs<T> &funcs, const ConvTestParams &tp,
                                 ConvLayout layout)
{
    tqt_conv_params p = tp.to_conv_params();

    ASSERT_EQ(p.groups, p.input_channel);
    ASSERT_EQ(p.output_channel % p.input_channel, 0u);
    // Specialized kernels are 2D: KD=ID=OD=1, dilation=1.
    ASSERT_EQ(p.kernel_shape[0], 1u);
    ASSERT_EQ(p.input_shape[0], 1u);
    ASSERT_EQ(p.output_shape[0], 1u);
    ASSERT_EQ(p.dilation[0], 1u);
    ASSERT_EQ(p.dilation[1], 1u);
    ASSERT_EQ(p.dilation[2], 1u);

    const size_t in_elems = dwconv_input_elements(p);
    const size_t wt_elems = dwconv_weight_elements(p);
    const size_t bs_elems = dwconv_bias_elements(p);
    const size_t out_elems = dwconv_output_elements(p);

    UniformRandomGenerator<T> gen(
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(-1),
        static_cast<typename std::conditional<std::is_integral<T>::value, T, float>::type>(1),
        /*seed=*/42);

    std::vector<T> input_cf(in_elems);
    std::vector<T> weight_cf(wt_elems);
    std::vector<T> bias(bs_elems);
    gen.fill_matrix(input_cf.data(), 1, in_elems);
    gen.fill_matrix(weight_cf.data(), 1, wt_elems);
    if (tp.has_bias) {
        gen.fill_matrix(bias.data(), 1, bs_elems);
    }
    const T *bias_ptr = tp.has_bias ? bias.data() : nullptr;

    std::vector<T> ref_output(out_elems, T(0));
    std::vector<T> input_cl, weight_cl;
    const T *src_input = nullptr;
    const T *src_weight = nullptr;
    if (layout == ConvLayout::kChannelFirst) {
        reference_dwconv<T>(input_cf.data(), weight_cf.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelFirst, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        src_input = input_cf.data();
        src_weight = weight_cf.data();
    } else {
        input_cl.resize(in_elems);
        weight_cl.resize(wt_elems);
        cf_to_cl<T>(input_cf.data(), input_cl.data(), p.input_channel, p.input_shape[0],
                    p.input_shape[1], p.input_shape[2]);
        weight_cf_to_cl<T>(weight_cf.data(), weight_cl.data(), p.output_channel, p.kernel_shape[0],
                           p.kernel_shape[1], p.kernel_shape[2]);
        reference_dwconv<T>(input_cl.data(), weight_cl.data(), bias_ptr, ref_output.data(), &p,
                            ConvLayout::kChannelLast, static_cast<T>(tp.clamp_min),
                            static_cast<T>(tp.clamp_max));
        src_input = input_cl.data();
        src_weight = weight_cl.data();
    }

    const size_t packed_in_bytes = funcs.get_packed_input_size(&p);
    const size_t packed_wt_bytes = funcs.get_packed_weight_size(&p);
    const size_t packed_out_bytes = funcs.get_output_size(&p);

    std::vector<uint8_t> packed_in(packed_in_bytes);
    std::vector<uint8_t> packed_wt(packed_wt_bytes);
    std::vector<uint8_t> packed_out(packed_out_bytes);

    funcs.run_pack_input(src_input, packed_in.data(), &p);
    funcs.run_pack_weight(src_weight, packed_wt.data(), &p);

    // Specialized: slice OH by get_oh_step()
    const size_t oh_step = funcs.get_oh_step();
    const size_t OH = p.output_shape[1];
    for (size_t oh = 0; oh < OH; oh += oh_step) {
        const size_t oh_end = std::min(oh + oh_step, OH);
        funcs.run_dwconv(oh, oh_end, packed_in.data(), packed_wt.data(), bias_ptr,
                         packed_out.data(), &p, static_cast<T>(tp.clamp_min),
                         static_cast<T>(tp.clamp_max));
    }

    std::vector<T> actual_output(out_elems, T(0));
    funcs.run_unpack_output(packed_out.data(), actual_output.data(), &p);

    auto vr =
        verify_gemm_result<T>(ref_output.data(), actual_output.data(), out_elems,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(vr.passed) << vr.error_message << " (max_abs=" << vr.max_abs_error
                           << ", cos=" << vr.cosine_similarity << ")";
}

}  // namespace test
}  // namespace torq_tile
