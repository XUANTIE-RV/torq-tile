//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"
#include "test/common/conv_test_runner.h"
#include "test/common/data_generator.h"
#include "test/common/quantized_gemm_test_runner.h"
#include "test/common/reference_conv.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Function pointer types for non-packed requantized CF conv
// ============================================================================

template <typename QParams>
struct NonPackedRequantizedConvCfFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetBIm2colSizeFunc = size_t (*)(size_t k, size_t n);
    using RunIm2colFunc = void (*)(size_t group_idx, const void *activation, void *im2col_buf,
                                   const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetBIm2colSizeFunc get_b_im2col_size;
    RunIm2colFunc run_im2col;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for non-packed requantized CL conv
// ============================================================================

template <typename QParams>
struct NonPackedRequantizedConvClFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAIm2colSizeFunc = size_t (*)(size_t m, size_t k);
    using RunIm2colFunc = void (*)(size_t group_idx, const void *activation, void *im2col_buf,
                                   const tqt_conv_params *params);
    using RunConvFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAIm2colSizeFunc get_a_im2col_size;
    RunIm2colFunc run_im2col;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for packed requantized CF conv
// ============================================================================

template <typename QParams>
struct PackedRequantizedConvCfFuncs
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
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBIm2colpackSizeFunc get_b_im2colpack_size;
    RunAPackFunc run_a_pack;
    RunBIm2colpackFunc run_b_im2colpack;
    RunConvFunc run_conv;
};

// ============================================================================
// Function pointer types for packed requantized CL conv
// ============================================================================

template <typename QParams>
struct PackedRequantizedConvClFuncs
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
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAIm2colpackSizeFunc get_a_im2colpack_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunBtPackFunc run_bt_pack;
    RunAIm2colpackFunc run_a_im2colpack;
    RunConvFunc run_conv;
};

// ============================================================================
// Non-packed CF conv test runner: requantize int32 -> int8
// CF layout: A = weight [OC_pg, K], B = im2col(activation) [K, N]
// Mx1 bias pattern: A per-channel symmetric, B global asymmetric
// ============================================================================

inline void run_nonpacked_convcf_reqi32toi8_test(
    const ConvTestParams &test_params,
    const NonPackedRequantizedConvCfFuncs<tqt_qai8_qsi8dx_qai8_params> &funcs)
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

    const size_t M = OC_per_group;  // output channels
    const size_t N = OD * OH * OW;  // spatial output
    const size_t K = IC_per_group * KD * KH * KW;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);

    // Activation: [IC, ID, IH, IW] (CF layout)
    std::vector<float> activation_fp32(IC * ID * IH * IW);
    gen.fill_matrix(activation_fp32.data(), IC, ID * IH * IW);

    // Weight: [OC_per_group, K] = [M, K] row-major
    std::vector<float> weight_fp32(M * K);
    gen.fill_matrix(weight_fp32.data(), M, K);

    // Bias: [OC_per_group] = [M]
    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(M);
    bias_gen.fill_matrix(bias_fp32.data(), M, 1);

    // Compute fp32 reference: using reference_conv<float> with CF layout
    std::vector<float> D_ref_fp32(M * N, 0.0f);
    reference_conv<float>(0, activation_fp32.data(), weight_fp32.data(), bias_fp32.data(),
                          D_ref_fp32.data(), &params, ConvLayout::kChannelFirst, -FLT_MAX, FLT_MAX);

    // Compute quantization parameters
    // A (weight): per-channel symmetric, each row is a channel
    std::vector<float> scale_a(M);
    for (size_t ch = 0; ch < M; ++ch) {
        compute_sym_quant_params_i8(&weight_fp32[ch * K], K, scale_a[ch]);
    }

    // B (activation): global asymmetric
    float scale_b;
    int32_t zp_b;
    compute_asym_quant_params_i8(activation_fp32.data(), IC * ID * IH * IW, scale_b, zp_b);

    // Quantize weight (per-channel symmetric)
    std::vector<int8_t> weight_i8(M * K);
    for (size_t ch = 0; ch < M; ++ch) {
        quantize_matrix_i8(&weight_fp32[ch * K], &weight_i8[ch * K], K, scale_a[ch], 0);
    }

    // Quantize activation (global asymmetric)
    std::vector<int8_t> activation_i8(IC * ID * IH * IW);
    quantize_matrix_i8(activation_fp32.data(), activation_i8.data(), IC * ID * IH * IW, scale_b,
                       zp_b);

    // Quantize bias (Mx1, scale = scale_a[ch] * scale_b, zp = 0)
    std::vector<int32_t> bias_i32(M);
    for (size_t ch = 0; ch < M; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_b into bias: bias_fused[m] = bias[m] - zp_b * sum(weight_i8[m, 0..K-1])
    std::vector<int32_t> bias_fused(M);
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), weight_i8.data(), zp_b, M, K);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), M * N, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qsi8dx_qai8_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.quant_channel_a = static_cast<int32_t>(M);
    qparams.scale_b = scale_b;
    qparams.zero_point_b = zp_b;
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Im2col activation
    size_t im2col_size = funcs.get_b_im2col_size(K, N);
    std::vector<uint8_t> im2col_buf(im2col_size);
    funcs.run_im2col(0, activation_i8.data(), im2col_buf.data(), &params);

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(M * N, 0);

    funcs.run_conv(M, N, K, weight_i8.data(), K, 0, im2col_buf.data(), N, 0, nullptr, N,
                   D_i8.data(), N, bias_fused.data(), clamp_min, clamp_max, &qparams);

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(M * N);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), M * N, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), M * N, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CF NonPacked ReqI32toI8: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Non-packed CL conv test runner: requantize int32 -> int8
// CL layout: A = im2col(activation) [M, K], B = weight [N, K] (transposed)
// 1xN bias pattern: A global asymmetric, B per-channel symmetric
// ============================================================================

inline void run_nonpacked_convcl_reqi32toi8_test(
    const ConvTestParams &test_params,
    const NonPackedRequantizedConvClFuncs<tqt_qai8_qai8_qsi8cx_params> &funcs)
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

    const size_t M = OD * OH * OW;  // spatial output
    const size_t N = OC_per_group;  // output channels
    const size_t K = KD * KH * KW * IC_per_group;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);

    // Activation: [ID, IH, IW, IC] (CL layout)
    std::vector<float> activation_fp32(ID * IH * IW * IC);
    gen.fill_matrix(activation_fp32.data(), ID * IH * IW, IC);

    // Weight: [OC_per_group, K] = [N, K] row-major (transposed for GEMM)
    std::vector<float> weight_fp32(N * K);
    gen.fill_matrix(weight_fp32.data(), N, K);

    // Bias: [OC_per_group] = [N]
    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(N);
    bias_gen.fill_matrix(bias_fp32.data(), 1, N);

    // Compute fp32 reference: using reference_conv<float> with CL layout
    std::vector<float> D_ref_fp32(M * N, 0.0f);
    reference_conv<float>(0, activation_fp32.data(), weight_fp32.data(), bias_fp32.data(),
                          D_ref_fp32.data(), &params, ConvLayout::kChannelLast, -FLT_MAX, FLT_MAX);

    // Compute quantization parameters
    // A (activation): global asymmetric
    float scale_a;
    int32_t zp_a;
    compute_asym_quant_params_i8(activation_fp32.data(), ID * IH * IW * IC, scale_a, zp_a);

    // B (weight): per-channel symmetric, each row of B[N,K] is a channel
    std::vector<float> scale_b(N);
    for (size_t ch = 0; ch < N; ++ch) {
        compute_sym_quant_params_i8(&weight_fp32[ch * K], K, scale_b[ch]);
    }

    // Quantize activation (global asymmetric)
    std::vector<int8_t> activation_i8(ID * IH * IW * IC);
    quantize_matrix_i8(activation_fp32.data(), activation_i8.data(), ID * IH * IW * IC, scale_a,
                       zp_a);

    // Quantize weight (per-channel symmetric, each row of B[N,K] is a channel)
    std::vector<int8_t> weight_i8(N * K);
    for (size_t ch = 0; ch < N; ++ch) {
        quantize_matrix_i8(&weight_fp32[ch * K], &weight_i8[ch * K], K, scale_b[ch], 0);
    }

    // Quantize bias (1xN, scale = scale_a * scale_b[ch], zp = 0)
    std::vector<int32_t> bias_i32(N);
    for (size_t ch = 0; ch < N; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_a into bias: bias_fused[n] = bias[n] - zp_a * sum(weight_i8[n, 0..K-1])
    std::vector<int32_t> bias_fused(N);
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), weight_i8.data(), zp_a, N, K);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), M * N, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qai8_qsi8cx_params qparams;
    qparams.scale_a = scale_a;
    qparams.zero_point_a = zp_a;
    qparams.scale_b = scale_b.data();
    qparams.quant_channel_b = static_cast<int32_t>(N);
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Im2col activation
    size_t im2col_size = funcs.get_a_im2col_size(M, K);
    std::vector<uint8_t> im2col_buf(im2col_size);
    funcs.run_im2col(0, activation_i8.data(), im2col_buf.data(), &params);

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(M * N, 0);

    funcs.run_conv(M, N, K, im2col_buf.data(), K, 0, weight_i8.data(), K, 0, nullptr, N,
                   D_i8.data(), N, bias_fused.data(), clamp_min, clamp_max, &qparams);

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(M * N);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), M * N, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), M * N, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CL NonPacked ReqI32toI8: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed CF conv test runner: requantize int32 -> int8
// CF layout: A = weight [OC_pg, K] packed, B = im2col(activation) [K, N] packed
// Mx1 bias pattern: A per-channel symmetric, B global asymmetric
// ============================================================================

inline void run_packed_convcf_reqi32toi8_test(
    const ConvTestParams &test_params,
    const PackedRequantizedConvCfFuncs<tqt_qai8_qsi8dx_qai8_params> &funcs)
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

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);

    std::vector<float> activation_fp32(IC * ID * IH * IW);
    gen.fill_matrix(activation_fp32.data(), IC, ID * IH * IW);

    std::vector<float> weight_fp32(M * K);
    gen.fill_matrix(weight_fp32.data(), M, K);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(M);
    bias_gen.fill_matrix(bias_fp32.data(), M, 1);

    // Compute fp32 reference
    std::vector<float> D_ref_fp32(M * N, 0.0f);
    reference_conv<float>(0, activation_fp32.data(), weight_fp32.data(), bias_fp32.data(),
                          D_ref_fp32.data(), &params, ConvLayout::kChannelFirst, -FLT_MAX, FLT_MAX);

    // Compute quantization parameters
    std::vector<float> scale_a(M);
    for (size_t ch = 0; ch < M; ++ch) {
        compute_sym_quant_params_i8(&weight_fp32[ch * K], K, scale_a[ch]);
    }

    float scale_b;
    int32_t zp_b;
    compute_asym_quant_params_i8(activation_fp32.data(), IC * ID * IH * IW, scale_b, zp_b);

    // Quantize
    std::vector<int8_t> weight_i8(M * K);
    for (size_t ch = 0; ch < M; ++ch) {
        quantize_matrix_i8(&weight_fp32[ch * K], &weight_i8[ch * K], K, scale_a[ch], 0);
    }

    std::vector<int8_t> activation_i8(IC * ID * IH * IW);
    quantize_matrix_i8(activation_fp32.data(), activation_i8.data(), IC * ID * IH * IW, scale_b,
                       zp_b);

    std::vector<int32_t> bias_i32(M);
    for (size_t ch = 0; ch < M; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    std::vector<int32_t> bias_fused(M);
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), weight_i8.data(), zp_b, M, K);

    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), M * N, scale_d, zp_d);

    struct tqt_qai8_qsi8dx_qai8_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.quant_channel_a = static_cast<int32_t>(M);
    qparams.scale_b = scale_b;
    qparams.zero_point_b = zp_b;
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Pack A (weight) [M, K]
    size_t a_packed_size = funcs.get_a_packed_size(M, K);
    std::vector<uint8_t> A_packed(a_packed_size);
    funcs.run_a_pack(M, K, K, K, 0, weight_i8.data(), A_packed.data());

    // Fused im2col+pack B (activation) -> B_packed
    size_t b_packed_size = funcs.get_b_im2colpack_size(N, K);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_b_im2colpack(0, activation_i8.data(), K, 0, B_packed.data(), &params);

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(M * N, 0);

    funcs.run_conv(M, N, K, A_packed.data(), K, 0, B_packed.data(), K, 0, nullptr, N, D_i8.data(),
                   N, bias_fused.data(), clamp_min, clamp_max, &qparams);

    // Dequantize and verify
    std::vector<float> D_act_fp32(M * N);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), M * N, scale_d, zp_d);

    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), M * N, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CF Packed ReqI32toI8: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed CL conv test runner: requantize int32 -> int8
// CL layout: A = im2col(activation) [M, K] packed, B = weight [N, K] (transposed) packed
// 1xN bias pattern: A global asymmetric, B per-channel symmetric
// ============================================================================

inline void run_packed_convcl_reqi32toi8_test(
    const ConvTestParams &test_params,
    const PackedRequantizedConvClFuncs<tqt_qai8_qai8_qsi8cx_params> &funcs)
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

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);

    std::vector<float> activation_fp32(ID * IH * IW * IC);
    gen.fill_matrix(activation_fp32.data(), ID * IH * IW, IC);

    std::vector<float> weight_fp32(N * K);
    gen.fill_matrix(weight_fp32.data(), N, K);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(N);
    bias_gen.fill_matrix(bias_fp32.data(), 1, N);

    // Compute fp32 reference
    std::vector<float> D_ref_fp32(M * N, 0.0f);
    reference_conv<float>(0, activation_fp32.data(), weight_fp32.data(), bias_fp32.data(),
                          D_ref_fp32.data(), &params, ConvLayout::kChannelLast, -FLT_MAX, FLT_MAX);

    // Compute quantization parameters
    float scale_a;
    int32_t zp_a;
    compute_asym_quant_params_i8(activation_fp32.data(), ID * IH * IW * IC, scale_a, zp_a);

    std::vector<float> scale_b(N);
    for (size_t ch = 0; ch < N; ++ch) {
        compute_sym_quant_params_i8(&weight_fp32[ch * K], K, scale_b[ch]);
    }

    // Quantize
    std::vector<int8_t> activation_i8(ID * IH * IW * IC);
    quantize_matrix_i8(activation_fp32.data(), activation_i8.data(), ID * IH * IW * IC, scale_a,
                       zp_a);

    std::vector<int8_t> weight_i8(N * K);
    for (size_t ch = 0; ch < N; ++ch) {
        quantize_matrix_i8(&weight_fp32[ch * K], &weight_i8[ch * K], K, scale_b[ch], 0);
    }

    std::vector<int32_t> bias_i32(N);
    for (size_t ch = 0; ch < N; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    std::vector<int32_t> bias_fused(N);
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), weight_i8.data(), zp_a, N, K);

    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), M * N, scale_d, zp_d);

    struct tqt_qai8_qai8_qsi8cx_params qparams;
    qparams.scale_a = scale_a;
    qparams.zero_point_a = zp_a;
    qparams.scale_b = scale_b.data();
    qparams.quant_channel_b = static_cast<int32_t>(N);
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Fused im2col+pack A (activation) -> A_packed
    size_t a_packed_size = funcs.get_a_im2colpack_size(M, K);
    std::vector<uint8_t> A_packed(a_packed_size);
    funcs.run_a_im2colpack(0, activation_i8.data(), K, 0, A_packed.data(), &params);

    // Pack B (weight) [N, K] transposed -> packed
    size_t b_packed_size = funcs.get_b_packed_size(N, K);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_bt_pack(N, K, K, K, 0, weight_i8.data(), B_packed.data());

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(M * N, 0);

    funcs.run_conv(M, N, K, A_packed.data(), K, 0, B_packed.data(), K, 0, nullptr, N, D_i8.data(),
                   N, bias_fused.data(), clamp_min, clamp_max, &qparams);

    // Dequantize and verify
    std::vector<float> D_act_fp32(M * N);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), M * N, scale_d, zp_d);

    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), M * N, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message
                               << "\n  Conv CL Packed ReqI32toI8: " << test_params.name
                               << "\n  IC=" << IC << " OC=" << params.output_channel
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

}  // namespace test
}  // namespace torq_tile
