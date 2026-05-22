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
#include "test/common/conv_test_runner.h"
#include "test/common/data_generator.h"
#include "test/common/reference_conv.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Function pointer types for igemmcf (channel-first)
// ============================================================================

template <typename ClampT>
struct NonPackedIgemmcfFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunIgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                                  size_t n_idx, size_t k_idx_b, const void *A, size_t lda,
                                  size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
                                  const void *bias, ClampT clamp_min, ClampT clamp_max,
                                  const tqt_conv_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunIgemmFunc run_igemm;
};

template <typename ClampT>
struct PackedIgemmcfFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                                  const void *weight, void *weight_packed);
    using RunIgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                                  size_t n_idx, size_t k_idx_b, const void *A_packed, size_t lda,
                                  size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
                                  const void *bias, ClampT clamp_min, ClampT clamp_max,
                                  const tqt_conv_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    RunAPackFunc run_a_pack;
    RunIgemmFunc run_igemm;
};

// ============================================================================
// Function pointer types for igemmcl (channel-last)
// ============================================================================

template <typename ClampT>
struct NonPackedIgemmclFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunIgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                                  size_t m_idx, size_t k_idx_a, const void *B, size_t ldb,
                                  size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                  const void *bias, ClampT clamp_min, ClampT clamp_max,
                                  const tqt_conv_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunIgemmFunc run_igemm;
};

template <typename ClampT>
struct PackedIgemmclFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                                   const void *weight, void *weight_packed);
    using RunIgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                                  size_t m_idx, size_t k_idx_a, const void *B_packed, size_t ldb,
                                  size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                  const void *bias, ClampT clamp_min, ClampT clamp_max,
                                  const tqt_conv_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetBPackedSizeFunc get_b_packed_size;
    RunBtPackFunc run_bt_pack;
    RunIgemmFunc run_igemm;
};

// ============================================================================
// Non-packed igemmcf test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_igemmcf_test(const ConvTestParams &test_params,
                                       const NonPackedIgemmcfFuncs<ClampT> &funcs)
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

    std::vector<T> act_output(OC_per_group * N, static_cast<T>(0.0f));
    funcs.run_igemm(M, N, K, activation.data(), 0, 0, 0, weight.data(), K, 0, nullptr, N,
                    act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                    to_target_type<T>(test_params.clamp_max), &params);

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Igemmcf NonPacked: " << test_params.name << "\n  IC=" << IC
                               << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed igemmcf test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_igemmcf_test(const ConvTestParams &test_params,
                                    const PackedIgemmcfFuncs<ClampT> &funcs)
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

    size_t w_packed_size = funcs.get_a_packed_size(M, K);
    std::vector<uint8_t> w_packed(w_packed_size);
    funcs.run_a_pack(M, K, K, K, 0, weight.data(), w_packed.data());

    std::vector<T> act_output(OC_per_group * N, static_cast<T>(0.0f));
    funcs.run_igemm(M, N, K, activation.data(), 0, 0, 0, w_packed.data(), K, 0, nullptr, N,
                    act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                    to_target_type<T>(test_params.clamp_max), &params);

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Igemmcf Packed: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Non-packed igemmcl test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_igemmcl_test(const ConvTestParams &test_params,
                                       const NonPackedIgemmclFuncs<ClampT> &funcs)
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

    std::vector<T> act_output(M * N, static_cast<T>(0.0f));
    funcs.run_igemm(M, N, K, activation.data(), 0, 0, 0, weight.data(), K, 0, nullptr, N,
                    act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                    to_target_type<T>(test_params.clamp_max), &params);

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Igemmcl NonPacked: " << test_params.name << "\n  IC=" << IC
                               << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed igemmcl test runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_igemmcl_test(const ConvTestParams &test_params,
                                    const PackedIgemmclFuncs<ClampT> &funcs)
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

    size_t w_packed_size = funcs.get_b_packed_size(N, K);
    std::vector<uint8_t> w_packed(w_packed_size);
    funcs.run_bt_pack(N, K, K, K, 0, weight.data(), w_packed.data());

    std::vector<T> act_output(M * N, static_cast<T>(0.0f));
    funcs.run_igemm(M, N, K, activation.data(), 0, 0, 0, w_packed.data(), K, 0, nullptr, N,
                    act_output.data(), N, bias_ptr, to_target_type<T>(test_params.clamp_min),
                    to_target_type<T>(test_params.clamp_max), &params);

    auto result =
        verify_gemm_result<T>(ref_output.data(), act_output.data(), M * N,
                              DefaultThreshold<T>::abs_error, DefaultThreshold<T>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Igemmcl Packed: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

}  // namespace test
}  // namespace torq_tile
