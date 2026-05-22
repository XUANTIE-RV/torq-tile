//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/common/tqt_conv.h"
#include "test/common/data_generator.h"
#include "test/common/reference_conv.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Conv Test Parameters
// ============================================================================

struct ConvTestParams
{
    size_t input_channel;
    size_t output_channel;
    size_t groups;
    size_t input_shape[3];   // {D, H, W}
    size_t kernel_shape[3];  // {KD, KH, KW}
    size_t stride[3];        // {SD, SH, SW}
    size_t dilation[3];      // {DD, DH, DW}
    size_t pad[3];           // {PD, PH, PW}
    bool has_bias;
    float clamp_min;
    float clamp_max;
    std::string name;

    ConvTestParams(size_t ic, size_t oc, size_t g, size_t ih, size_t iw, size_t kh, size_t kw,
                   std::string n = "")
        : input_channel(ic),
          output_channel(oc),
          groups(g),
          input_shape{1, ih, iw},
          kernel_shape{1, kh, kw},
          stride{1, 1, 1},
          dilation{1, 1, 1},
          pad{0, 0, 0},
          has_bias(true),
          clamp_min(-FLT_MAX),
          clamp_max(FLT_MAX),
          name(std::move(n))
    {
    }

    ConvTestParams &set_stride(size_t sh, size_t sw)
    {
        stride[1] = sh;
        stride[2] = sw;
        return *this;
    }
    ConvTestParams &set_pad(size_t ph, size_t pw)
    {
        pad[1] = ph;
        pad[2] = pw;
        return *this;
    }
    ConvTestParams &set_dilation(size_t dh, size_t dw)
    {
        dilation[1] = dh;
        dilation[2] = dw;
        return *this;
    }
    ConvTestParams &set_bias(bool v)
    {
        has_bias = v;
        return *this;
    }
    ConvTestParams &set_clamp(float lo, float hi)
    {
        clamp_min = lo;
        clamp_max = hi;
        return *this;
    }

    tqt_conv_params to_conv_params() const
    {
        tqt_conv_params p;
        memcpy(p.input_shape, input_shape, sizeof(input_shape));
        memcpy(p.kernel_shape, kernel_shape, sizeof(kernel_shape));
        memcpy(p.stride, stride, sizeof(stride));
        memcpy(p.dilation, dilation, sizeof(dilation));
        memcpy(p.pad, pad, sizeof(pad));
        p.input_channel = input_channel;
        p.output_channel = output_channel;
        p.groups = groups;
        // Compute output shape
        for (int i = 0; i < 3; ++i) {
            p.output_shape[i] =
                (input_shape[i] + 2 * pad[i] - dilation[i] * (kernel_shape[i] - 1) - 1) /
                    stride[i] +
                1;
        }
        return p;
    }
};

// ============================================================================
// Function pointer types for non-packed CF conv
// ============================================================================

template <typename ClampT>
struct NonPackedConvCfFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t m_idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetBIm2colSizeFunc = size_t (*)(size_t k, size_t n);
    using GetBIm2colOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb_im2col);
    using RunIm2colFunc = void (*)(size_t group_idx, const void *activation, void *im2col_buf,
                                   const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetBIm2colSizeFunc get_b_im2col_size;
    RunIm2colFunc run_im2col;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for non-packed CL conv
// ============================================================================

template <typename ClampT>
struct NonPackedConvClFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetBiasOffsetFunc = size_t (*)(size_t n_idx);
    using GetAIm2colSizeFunc = size_t (*)(size_t m, size_t k);
    using RunIm2colFunc = void (*)(size_t group_idx, const void *activation, void *im2col_buf,
                                   const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAIm2colSizeFunc get_a_im2col_size;
    RunIm2colFunc run_im2col;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for packed CF conv
// ============================================================================

template <typename ClampT>
struct PackedConvCfFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBIm2colpackSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBIm2colpackFunc = void (*)(size_t group_idx, const void *activation, size_t ldb_packed,
                                        size_t k_idx, void *B_packed,
                                        const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBIm2colpackSizeFunc get_b_im2colpack_size;
    RunAPackFunc run_a_pack;
    RunBIm2colpackFunc run_b_im2colpack;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for packed CL conv
// ============================================================================

template <typename ClampT>
struct PackedConvClFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAIm2colpackSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunAIm2colpackFunc = void (*)(size_t group_idx, const void *activation, size_t lda_packed,
                                        size_t k_idx, void *A_packed,
                                        const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAIm2colpackSizeFunc get_a_im2colpack_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunBtPackFunc run_bt_pack;
    RunAIm2colpackFunc run_a_im2colpack;
    RunConvFunc run_conv;
};

// ============================================================================
// Non-packed CF conv test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_convcf_test(const ConvTestParams &test_params,
                                      const NonPackedConvCfFuncs<ClampT> &funcs)
{
    tqt_conv_params params = test_params.to_conv_params();

    const size_t IC = params.input_channel;
    const size_t OC_per_group = params.output_channel / params.groups;
    const size_t IC_per_group = IC / params.groups;
    const size_t ID = params.input_shape[0], IH = params.input_shape[1], IW = params.input_shape[2];
    const size_t OD = params.output_shape[0], OH = params.output_shape[1],
                 OW = params.output_shape[2];
    const size_t KD = params.kernel_shape[0], KH = params.kernel_shape[1],
                 KW = params.kernel_shape[2];

    const size_t M = OC_per_group;
    const size_t N = OD * OH * OW;
    const size_t K = IC_per_group * KD * KH * KW;

    UniformRandomGenerator<T> gen(-1.0f, 1.0f, 42);

    // Activation: [IC, ID, IH, IW] (CF layout)
    std::vector<T> activation(IC * ID * IH * IW);
    gen.fill_matrix(activation.data(), IC, ID * IH * IW);

    // Weight: [OC_per_group, IC_per_group * KD * KH * KW] = [M, K] row-major
    std::vector<T> weight(OC_per_group * K);
    gen.fill_matrix(weight.data(), OC_per_group, K);

    // Bias: [OC_per_group]
    std::vector<T> bias_data(OC_per_group);
    if (test_params.has_bias) {
        gen.fill_matrix(bias_data.data(), OC_per_group, 1);
    }
    const T *bias_ptr = test_params.has_bias ? bias_data.data() : nullptr;

    // Reference output: [OC_per_group, OD, OH, OW]
    std::vector<T> ref_output(OC_per_group * N, static_cast<T>(0.0f));
    // Weight needs to be in [OC_per_group, IC_per_group, KD, KH, KW] for reference
    reference_conv<T>(0, activation.data(), weight.data(), bias_ptr, ref_output.data(), &params,
                      ConvLayout::kChannelFirst, to_target_type<T>(test_params.clamp_min),
                      to_target_type<T>(test_params.clamp_max));

    // Actual output via ukernel
    // Step 1: im2col B (activation) -> [K, N]
    size_t im2col_size = funcs.get_b_im2col_size(K, N);
    std::vector<uint8_t> im2col_buf(im2col_size);
    funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);

    // Step 2: run conv: A=weight[M,K], B=im2col[K,N]
    std::vector<T> act_output(OC_per_group * N, static_cast<T>(0.0f));
    funcs.run_conv(M, N, K, weight.data(), K, 0, im2col_buf.data(), N, 0, nullptr, N,
                   act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                   to_target_type<T>(test_params.clamp_max));

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CF NonPacked: " << test_params.name << "\n  IC=" << IC
                               << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Non-packed CL conv test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_convcl_test(const ConvTestParams &test_params,
                                      const NonPackedConvClFuncs<ClampT> &funcs)
{
    tqt_conv_params params = test_params.to_conv_params();

    const size_t IC = params.input_channel;
    const size_t OC_per_group = params.output_channel / params.groups;
    const size_t IC_per_group = IC / params.groups;
    const size_t ID = params.input_shape[0], IH = params.input_shape[1], IW = params.input_shape[2];
    const size_t OD = params.output_shape[0], OH = params.output_shape[1],
                 OW = params.output_shape[2];
    const size_t KD = params.kernel_shape[0], KH = params.kernel_shape[1],
                 KW = params.kernel_shape[2];

    const size_t M = OD * OH * OW;
    const size_t N = OC_per_group;
    const size_t K = KD * KH * KW * IC_per_group;

    UniformRandomGenerator<T> gen(-1.0f, 1.0f, 42);

    // Activation: [ID, IH, IW, IC] (CL layout)
    std::vector<T> activation(ID * IH * IW * IC);
    gen.fill_matrix(activation.data(), ID * IH * IW, IC);

    // Weight: [OC_per_group, K] = [N, K] row-major (transposed for GEMM)
    std::vector<T> weight(OC_per_group * K);
    gen.fill_matrix(weight.data(), OC_per_group, K);

    // Bias: [OC_per_group]
    std::vector<T> bias_data(OC_per_group);
    if (test_params.has_bias) {
        gen.fill_matrix(bias_data.data(), OC_per_group, 1);
    }
    const T *bias_ptr = test_params.has_bias ? bias_data.data() : nullptr;

    // Reference output: [OD, OH, OW, OC_per_group]
    std::vector<T> ref_output(M * N, static_cast<T>(0.0f));
    reference_conv<T>(0, activation.data(), weight.data(), bias_ptr, ref_output.data(), &params,
                      ConvLayout::kChannelLast, to_target_type<T>(test_params.clamp_min),
                      to_target_type<T>(test_params.clamp_max));

    // Actual output via ukernel
    // Step 1: im2col A (activation) -> [M, K]
    size_t im2col_size = funcs.get_a_im2col_size(M, K);
    std::vector<uint8_t> im2col_buf(im2col_size);
    funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);

    // Step 2: run conv: A=im2col[M,K], B=weight[N,K] (transposed)
    std::vector<T> act_output(M * N, static_cast<T>(0.0f));
    funcs.run_conv(M, N, K, im2col_buf.data(), K, 0, weight.data(), K, 0, nullptr, N,
                   act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                   to_target_type<T>(test_params.clamp_max));

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CL NonPacked: " << test_params.name << "\n  IC=" << IC
                               << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed CF conv test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_convcf_test(const ConvTestParams &test_params,
                                   const PackedConvCfFuncs<ClampT> &funcs)
{
    tqt_conv_params params = test_params.to_conv_params();

    const size_t IC = params.input_channel;
    const size_t OC_per_group = params.output_channel / params.groups;
    const size_t IC_per_group = IC / params.groups;
    const size_t ID = params.input_shape[0], IH = params.input_shape[1], IW = params.input_shape[2];
    const size_t OD = params.output_shape[0], OH = params.output_shape[1],
                 OW = params.output_shape[2];
    const size_t KD = params.kernel_shape[0], KH = params.kernel_shape[1],
                 KW = params.kernel_shape[2];

    const size_t M = OC_per_group;
    const size_t N = OD * OH * OW;
    const size_t K = IC_per_group * KD * KH * KW;

    UniformRandomGenerator<T> gen(-1.0f, 1.0f, 42);

    std::vector<T> activation(IC * ID * IH * IW);
    gen.fill_matrix(activation.data(), IC, ID * IH * IW);

    std::vector<T> weight(OC_per_group * K);
    gen.fill_matrix(weight.data(), OC_per_group, K);

    std::vector<T> bias_data(OC_per_group);
    if (test_params.has_bias) {
        gen.fill_matrix(bias_data.data(), OC_per_group, 1);
    }
    const T *bias_ptr = test_params.has_bias ? bias_data.data() : nullptr;

    std::vector<T> ref_output(OC_per_group * N, static_cast<T>(0.0f));
    reference_conv<T>(0, activation.data(), weight.data(), bias_ptr, ref_output.data(), &params,
                      ConvLayout::kChannelFirst, to_target_type<T>(test_params.clamp_min),
                      to_target_type<T>(test_params.clamp_max));

    // Pack A (weight) [M, K]
    size_t a_packed_size = funcs.get_a_packed_size(M, K);
    std::vector<uint8_t> A_packed(a_packed_size);
    funcs.run_a_pack(M, K, K, K, 0, weight.data(), A_packed.data());

    // Fused im2col+pack B (activation) -> B_packed
    size_t b_packed_size = funcs.get_b_im2colpack_size(N, K);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_b_im2colpack(0, activation.data(), K, 0, B_packed.data(), &params);

    std::vector<T> act_output(OC_per_group * N, static_cast<T>(0.0f));
    funcs.run_conv(M, N, K, A_packed.data(), K, 0, B_packed.data(), K, 0, nullptr, N,
                   act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                   to_target_type<T>(test_params.clamp_max));

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Conv CF Packed: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed CL conv test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_convcl_test(const ConvTestParams &test_params,
                                   const PackedConvClFuncs<ClampT> &funcs)
{
    tqt_conv_params params = test_params.to_conv_params();

    const size_t IC = params.input_channel;
    const size_t OC_per_group = params.output_channel / params.groups;
    const size_t IC_per_group = IC / params.groups;
    const size_t ID = params.input_shape[0], IH = params.input_shape[1], IW = params.input_shape[2];
    const size_t OD = params.output_shape[0], OH = params.output_shape[1],
                 OW = params.output_shape[2];
    const size_t KD = params.kernel_shape[0], KH = params.kernel_shape[1],
                 KW = params.kernel_shape[2];

    const size_t M = OD * OH * OW;
    const size_t N = OC_per_group;
    const size_t K = KD * KH * KW * IC_per_group;

    UniformRandomGenerator<T> gen(-1.0f, 1.0f, 42);

    std::vector<T> activation(ID * IH * IW * IC);
    gen.fill_matrix(activation.data(), ID * IH * IW, IC);

    std::vector<T> weight(OC_per_group * K);
    gen.fill_matrix(weight.data(), OC_per_group, K);

    std::vector<T> bias_data(OC_per_group);
    if (test_params.has_bias) {
        gen.fill_matrix(bias_data.data(), OC_per_group, 1);
    }
    const T *bias_ptr = test_params.has_bias ? bias_data.data() : nullptr;

    std::vector<T> ref_output(M * N, static_cast<T>(0.0f));
    reference_conv<T>(0, activation.data(), weight.data(), bias_ptr, ref_output.data(), &params,
                      ConvLayout::kChannelLast, to_target_type<T>(test_params.clamp_min),
                      to_target_type<T>(test_params.clamp_max));

    // Fused im2col+pack A (activation) -> A_packed
    size_t a_packed_size = funcs.get_a_im2colpack_size(M, K);
    std::vector<uint8_t> A_packed(a_packed_size);
    funcs.run_a_im2colpack(0, activation.data(), K, 0, A_packed.data(), &params);

    // Pack B (weight) [N, K] transposed -> packed
    size_t b_packed_size = funcs.get_b_packed_size(N, K);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_bt_pack(N, K, K, K, 0, weight.data(), B_packed.data());

    std::vector<T> act_output(M * N, static_cast<T>(0.0f));
    funcs.run_conv(M, N, K, A_packed.data(), K, 0, B_packed.data(), K, 0, nullptr, N,
                   act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                   to_target_type<T>(test_params.clamp_max));

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Conv CL Packed: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// GTest parameter name generator
// ============================================================================

struct ConvTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<ConvTestParams> &info) const
    {
        const auto &p = info.param;
        std::string test_name =
            "IC" + std::to_string(p.input_channel) + "_OC" + std::to_string(p.output_channel) +
            "_IH" + std::to_string(p.input_shape[1]) + "_IW" + std::to_string(p.input_shape[2]) +
            "_KH" + std::to_string(p.kernel_shape[1]) + "_KW" + std::to_string(p.kernel_shape[2]);
        if (!p.name.empty()) {
            test_name += "_" + p.name;
        }
        // Sanitize
        for (auto &c : test_name) {
            if (!std::isalnum(c) && c != '_')
                c = '_';
        }
        return test_name;
    }
};

}  // namespace test
}  // namespace torq_tile
