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
#include <vector>

#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"
#include "test/common/data_generator.h"
#include "test/common/gemm_test_runner.h"
#include "test/common/quantize_utils.h"
#include "test/common/reference_gemm.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Quantization parameter computation helpers
// ============================================================================

/// Compute asymmetric int8 quantization parameters from data
/// scale = (max - min) / 255, zp = round(-128 - min/scale)
inline void compute_asym_quant_params_i8(const float *data, size_t size, float &scale, int32_t &zp)
{
    float fmin = FLT_MAX;
    float fmax = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < fmin)
            fmin = data[i];
        if (data[i] > fmax)
            fmax = data[i];
    }

    fmax = fmax > 0.0f ? fmax : 0.0f;
    fmin = fmin < 0.0f ? fmin : 0.0f;

    float scale_tmp = (fmax - fmin) / 255.0f;
    if (scale_tmp == 0.0f) {
        scale = 1.0f;
        zp = 0;
        return;
    }

    float zp_tmp = -128.0f - fmin / scale_tmp;
    int32_t zp_val = static_cast<int32_t>(std::nearbyintf(zp_tmp));
    if (zp_val == -128)
        zp_val = -127;

    scale = scale_tmp;
    zp = zp_val;
}

/// Compute symmetric int8 quantization parameters from data
/// scale = 2 * abs_max / 255 (zero_point = 0 implied)
inline void compute_sym_quant_params_i8(const float *data, size_t size, float &scale)
{
    float fmin = FLT_MAX;
    float fmax = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < fmin)
            fmin = data[i];
        if (data[i] > fmax)
            fmax = data[i];
    }

    float abs_max = (std::fabsf(fmax) >= std::fabsf(fmin)) ? std::fabsf(fmax) : std::fabsf(fmin);
    float scale_tmp = 2.0f * abs_max / 255.0f;
    if (scale_tmp == 0.0f)
        scale_tmp = 1.0f;

    scale = scale_tmp;
}

// ============================================================================
// Function pointer types for requantized non-packed variants
// ============================================================================

template <typename QParams>
struct NonPackedRequantizedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: gemm_clamp_i32_i8_i8 (int8 inputs -> int32 output with clamp)
// ============================================================================

/// Test runner for gemm_clamp_i32_i8_i8 (int8 inputs -> int32 output with clamp)
inline void run_clamp_i32_i8_gemm_test(const GemmTestParams &params,
                                       const NonPackedGemmFuncs<int32_t> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate random int8/int32 data
    std::vector<int8_t> A(m * k);
    std::vector<int8_t> B(k * n);
    std::vector<int32_t> C(m * n);
    std::vector<int32_t> D_ref(m * n);
    std::vector<int32_t> D_act(m * n, 0);

    UniformRandomGenerator<int8_t> gen_i8(-50, 50, 42);
    gen_i8.fill_matrix(A.data(), m, k);
    gen_i8.fill_matrix(B.data(), k, n);
    if (params.has_c) {
        UniformRandomGenerator<int32_t> gen_i32(-1000, 1000, 42);
        gen_i32.fill_matrix(C.data(), m, n);
    }

    const int32_t clamp_min = -100000;
    const int32_t clamp_max = 100000;

    // Reference: int8 -> int32 with clamp
    const int32_t *c_ptr = params.has_c ? C.data() : nullptr;
    reference_gemm_i32_i8(m, n, k, A.data(), k, B.data(), n, BLayout::kRowMajor, c_ptr, n,
                          D_ref.data(), n, clamp_min, clamp_max);

    // Actual: using tiling loop
    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_offset(m_idx, 0, k);
            const size_t b_offset = funcs.get_b_offset(n_idx, 0, n);
            const size_t c_offset = funcs.get_c_offset(m_idx, n_idx, n);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const char *>(B.data()) + b_offset;
            const void *c_tile_ptr =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_offset : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, n, 0, c_tile_ptr, n, d_ptr, n,
                           nullptr, clamp_min, clamp_max);
        }
    }

    // Verify
    auto result = verify_gemm_result<int32_t>(D_ref.data(), D_act.data(), m * n,
                                              DefaultThreshold<int32_t>::abs_error,
                                              DefaultThreshold<int32_t>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Test runner: 1xN bias requantized GEMM
// A: asymmetric int8, B: per-channel symmetric int8 transposed [N,K], D: asymmetric int8
// ============================================================================

/// Test runner for 1xN bias requantized GEMM
/// A: asymmetric int8, B: per-channel symmetric int8 transposed [N,K], D: asymmetric int8
inline void run_1xnbias_reqi32toi8_gemm_test(
    const GemmTestParams &params,
    const NonPackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);  // B is [N, K] (transposed)
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(n);
    bias_gen.fill_matrix(bias_fp32.data(), 1, n);

    // Compute quantization parameters from data
    float scale_a;
    int32_t zp_a;
    compute_asym_quant_params_i8(A_fp32.data(), m * k, scale_a, zp_a);

    std::vector<float> scale_b(n);
    for (size_t ch = 0; ch < n; ++ch) {
        compute_sym_quant_params_i8(&B_fp32[ch * k], k, scale_b[ch]);
    }

    // Quantize inputs
    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(n * k);  // [N, K]
    std::vector<int32_t> bias_i32(n);
    std::vector<int32_t> bias_fused(n);

    // Quantize A (global asymmetric)
    quantize_matrix_i8(A_fp32.data(), A_i8.data(), m * k, scale_a, zp_a);

    // Quantize B (per-channel symmetric, each row of B[N,K] is a channel)
    for (size_t ch = 0; ch < n; ++ch) {
        quantize_matrix_i8(&B_fp32[ch * k], &B_i8[ch * k], k, scale_b[ch], 0);
    }

    // Quantize bias (1xN, scale = scale_a * scale_b[ch], zp = 0)
    for (size_t ch = 0; ch < n; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_a into bias: bias_fused[n] = bias[n] - zp_a * sum(B_i8[n, 0..K-1])
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), B_i8.data(), zp_a, n, k);

    // Compute fp32 reference: D_ref = A[M,K] * B[N,K]^T + bias[1,N]
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), k, BLayout::kTransposed,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::k1xN,
                                 -FLT_MAX, FLT_MAX);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qai8_qsi8cx_params qparams;
    qparams.scale_a = scale_a;
    qparams.zero_point_a = zp_a;
    qparams.scale_b = scale_b.data();
    qparams.quant_channel_b = static_cast<int32_t>(n);
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_offset(m_idx, 0, k);
            const size_t b_offset = funcs.get_b_offset(n_idx, 0, k);
            const size_t bias_offset = funcs.get_bias_offset(n_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A_i8.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const char *>(B_i8.data()) + b_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_fused.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_i8.data()) + d_offset;

            // Note: ldb = k because B is [N,K] transposed format
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Test runner: Mx1 bias requantized GEMM
// A: per-channel symmetric int8, B: asymmetric int8 [K,N], D: asymmetric int8
// ============================================================================

/// Test runner for Mx1 bias requantized GEMM
/// A: per-channel symmetric int8, B: asymmetric int8 [K,N], D: asymmetric int8
inline void run_mx1bias_reqi32toi8_gemm_test(
    const GemmTestParams &params,
    const NonPackedRequantizedGemmFuncs<tqt_qai8_qsi8dx_qai8_params> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(k * n);  // B is [K, N] (row major)
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), k, n);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(m);
    bias_gen.fill_matrix(bias_fp32.data(), m, 1);

    // Compute quantization parameters from data
    float scale_b;
    int32_t zp_b;
    compute_asym_quant_params_i8(B_fp32.data(), k * n, scale_b, zp_b);

    std::vector<float> scale_a(m);
    for (size_t ch = 0; ch < m; ++ch) {
        compute_sym_quant_params_i8(&A_fp32[ch * k], k, scale_a[ch]);
    }

    // Quantize inputs
    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(k * n);
    std::vector<int32_t> bias_i32(m);
    std::vector<int32_t> bias_fused(m);

    // Quantize B (global asymmetric)
    quantize_matrix_i8(B_fp32.data(), B_i8.data(), k * n, scale_b, zp_b);

    // Quantize A (per-channel symmetric, each row is a channel)
    for (size_t ch = 0; ch < m; ++ch) {
        quantize_matrix_i8(&A_fp32[ch * k], &A_i8[ch * k], k, scale_a[ch], 0);
    }

    // Quantize bias (Mx1, scale = scale_a[ch] * scale_b, zp = 0)
    for (size_t ch = 0; ch < m; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_b into bias: bias_fused[m] = bias[m] - zp_b * sum(A_i8[m, 0..K-1])
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), A_i8.data(), zp_b, m, k);

    // Compute fp32 reference: D_ref = A[M,K] * B[K,N] + bias[M,1]
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), n, BLayout::kRowMajor,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::kMx1,
                                 -FLT_MAX, FLT_MAX);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qsi8dx_qai8_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.quant_channel_a = static_cast<int32_t>(m);
    qparams.scale_b = scale_b;
    qparams.zero_point_b = zp_b;
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Execute kernel
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_offset(m_idx, 0, k);
            const size_t b_offset = funcs.get_b_offset(n_idx, 0, n);
            const size_t bias_offset = funcs.get_bias_offset(m_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A_i8.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const char *>(B_i8.data()) + b_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_fused.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_i8.data()) + d_offset;

            // Note: ldb = n because B is [K,N] row major format
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, n, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Function pointer types for packed requantized variants
// ============================================================================

template <typename QParams>
struct PackedRequantizedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda,
                                            size_t actual_m);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                  const void *B, void *B_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBPackFunc run_b_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: packed gemm_clamp_i32_i8p_i8p (int8 packed inputs -> int32 output)
// ============================================================================

/// Test runner for packed gemm_clamp_i32_i8p_i8p
/// A: int8 [M,K] row-major -> packed, B: int8 [K,N] row-major -> packed, D: int32
inline void run_packed_clamp_i32_i8_gemm_test(const GemmTestParams &params,
                                              const PackedGemmFuncs<int32_t> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate random int8/int32 data
    std::vector<int8_t> A(m * k);
    std::vector<int8_t> B(k * n);
    std::vector<int32_t> C(m * n);
    std::vector<int32_t> D_ref(m * n);
    std::vector<int32_t> D_act(m * n, 0);

    UniformRandomGenerator<int8_t> gen_i8(-50, 50, 42);
    gen_i8.fill_matrix(A.data(), m, k);
    gen_i8.fill_matrix(B.data(), k, n);
    if (params.has_c) {
        UniformRandomGenerator<int32_t> gen_i32(-1000, 1000, 42);
        gen_i32.fill_matrix(C.data(), m, n);
    }

    const int32_t clamp_min = -100000;
    const int32_t clamp_max = 100000;

    // Reference: int8 -> int32 with clamp, B is [K,N] row-major
    const int32_t *c_ptr = params.has_c ? C.data() : nullptr;
    reference_gemm_i32_i8(m, n, k, A.data(), k, B.data(), n, BLayout::kRowMajor, c_ptr, n,
                          D_ref.data(), n, clamp_min, clamp_max);

    // Pack A and B
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, k, k, 0, static_cast<const void *>(A.data()),
                     static_cast<void *>(A_packed.data()));
    funcs.run_b_pack(n, k, n, k, 0, static_cast<const void *>(B.data()),
                     static_cast<void *>(B_packed.data()));

    // Actual: using tiling loop with packed matrices
    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_packed_offset = funcs.get_a_packed_offset(m_idx, 0, k, m);
            const size_t b_packed_offset = funcs.get_b_packed_offset(n_idx, 0, k, n);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = A_packed.data() + a_packed_offset;
            const void *b_ptr = B_packed.data() + b_packed_offset;

            const void *c_tile_ptr = nullptr;
            if (params.has_c) {
                const size_t c_offset = funcs.get_c_offset(m_idx, n_idx, n);
                c_tile_ptr = reinterpret_cast<const char *>(C.data()) + c_offset;
            }

            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, c_tile_ptr, n, d_ptr, n,
                           nullptr, clamp_min, clamp_max);
        }
    }

    // Verify
    auto result = verify_gemm_result<int32_t>(D_ref.data(), D_act.data(), m * n,
                                              DefaultThreshold<int32_t>::abs_error,
                                              DefaultThreshold<int32_t>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Test runner: packed 1xN bias requantized GEMM
// A: asymmetric int8 [M,K] -> packed, B: per-channel symmetric int8 transposed [N,K] -> packed
// D: asymmetric int8
// ============================================================================

/// Test runner for packed 1xN bias requantized GEMM
inline void run_packed_1xnbias_reqi32toi8_gemm_test(
    const GemmTestParams &params,
    const PackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);  // B is [N, K] (transposed)
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(n);
    bias_gen.fill_matrix(bias_fp32.data(), 1, n);

    // Compute quantization parameters from data
    float scale_a;
    int32_t zp_a;
    compute_asym_quant_params_i8(A_fp32.data(), m * k, scale_a, zp_a);

    std::vector<float> scale_b(n);
    for (size_t ch = 0; ch < n; ++ch) {
        compute_sym_quant_params_i8(&B_fp32[ch * k], k, scale_b[ch]);
    }

    // Quantize inputs
    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(n * k);  // [N, K]
    std::vector<int32_t> bias_i32(n);
    std::vector<int32_t> bias_fused(n);

    // Quantize A (global asymmetric)
    quantize_matrix_i8(A_fp32.data(), A_i8.data(), m * k, scale_a, zp_a);

    // Quantize B (per-channel symmetric, each row of B[N,K] is a channel)
    for (size_t ch = 0; ch < n; ++ch) {
        quantize_matrix_i8(&B_fp32[ch * k], &B_i8[ch * k], k, scale_b[ch], 0);
    }

    // Quantize bias (1xN, scale = scale_a * scale_b[ch], zp = 0)
    for (size_t ch = 0; ch < n; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_a into bias: bias_fused[n] = bias[n] - zp_a * sum(B_i8[n, 0..K-1])
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), B_i8.data(), zp_a, n, k);

    // Compute fp32 reference: D_ref = A[M,K] * B[N,K]^T + bias[1,N]
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), k, BLayout::kTransposed,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::k1xN,
                                 -FLT_MAX, FLT_MAX);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qai8_qsi8cx_params qparams;
    qparams.scale_a = scale_a;
    qparams.zero_point_a = zp_a;
    qparams.scale_b = scale_b.data();
    qparams.quant_channel_b = static_cast<int32_t>(n);
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Pack A and B (B is [N,K] transposed -> use bt_pack)
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, k, k, 0, static_cast<const void *>(A_i8.data()),
                     static_cast<void *>(A_packed.data()));
    funcs.run_bt_pack(n, k, k, k, 0, static_cast<const void *>(B_i8.data()),
                      static_cast<void *>(B_packed.data()));

    // Execute kernel with packed matrices
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_packed_offset = funcs.get_a_packed_offset(m_idx, 0, k, m);
            const size_t b_packed_offset = funcs.get_b_packed_offset(n_idx, 0, k, n);
            const size_t bias_offset = funcs.get_bias_offset(n_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = A_packed.data() + a_packed_offset;
            const void *b_ptr = B_packed.data() + b_packed_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_fused.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_i8.data()) + d_offset;

            // Note: ldb = k because B is [N,K] transposed format
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Test runner: packed Mx1 bias requantized GEMM
// A: per-channel symmetric int8 [M,K] -> packed, B: asymmetric int8 [K,N] -> packed
// D: asymmetric int8
// ============================================================================

/// Test runner for packed Mx1 bias requantized GEMM
inline void run_packed_mx1bias_reqi32toi8_gemm_test(
    const GemmTestParams &params,
    const PackedRequantizedGemmFuncs<tqt_qai8_qsi8dx_qai8_params> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(k * n);  // B is [K, N] (row major)
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), k, n);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(m);
    bias_gen.fill_matrix(bias_fp32.data(), m, 1);

    // Compute quantization parameters from data
    float scale_b;
    int32_t zp_b;
    compute_asym_quant_params_i8(B_fp32.data(), k * n, scale_b, zp_b);

    std::vector<float> scale_a(m);
    for (size_t ch = 0; ch < m; ++ch) {
        compute_sym_quant_params_i8(&A_fp32[ch * k], k, scale_a[ch]);
    }

    // Quantize inputs
    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(k * n);
    std::vector<int32_t> bias_i32(m);
    std::vector<int32_t> bias_fused(m);

    // Quantize B (global asymmetric)
    quantize_matrix_i8(B_fp32.data(), B_i8.data(), k * n, scale_b, zp_b);

    // Quantize A (per-channel symmetric, each row is a channel)
    for (size_t ch = 0; ch < m; ++ch) {
        quantize_matrix_i8(&A_fp32[ch * k], &A_i8[ch * k], k, scale_a[ch], 0);
    }

    // Quantize bias (Mx1, scale = scale_a[ch] * scale_b, zp = 0)
    for (size_t ch = 0; ch < m; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_b into bias: bias_fused[m] = bias[m] - zp_b * sum(A_i8[m, 0..K-1])
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), A_i8.data(), zp_b, m, k);

    // Compute fp32 reference: D_ref = A[M,K] * B[K,N] + bias[M,1]
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), n, BLayout::kRowMajor,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::kMx1,
                                 -FLT_MAX, FLT_MAX);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qsi8dx_qai8_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.quant_channel_a = static_cast<int32_t>(m);
    qparams.scale_b = scale_b;
    qparams.zero_point_b = zp_b;
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Pack A and B (B is [K,N] row-major -> use b_pack)
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, k, k, 0, static_cast<const void *>(A_i8.data()),
                     static_cast<void *>(A_packed.data()));
    funcs.run_b_pack(n, k, n, k, 0, static_cast<const void *>(B_i8.data()),
                     static_cast<void *>(B_packed.data()));

    // Execute kernel with packed matrices
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_packed_offset = funcs.get_a_packed_offset(m_idx, 0, k, m);
            const size_t b_packed_offset = funcs.get_b_packed_offset(n_idx, 0, k, n);
            const size_t bias_offset = funcs.get_bias_offset(m_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = A_packed.data() + a_packed_offset;
            const void *b_ptr = B_packed.data() + b_packed_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_fused.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_i8.data()) + d_offset;

            // Note: ldb = k for packed B format
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Function pointer types for B-only-packed requantized variants
// (A is in original layout, only B is packed)
// ============================================================================

template <typename QParams>
struct BOnlyPackedRequantizedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunBPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                  const void *B, void *B_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max, const QParams *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunBPackFunc run_b_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: B-only-packed 1xN bias requantized GEMM
// A: asymmetric int8 [M,K] original layout, B: per-channel symmetric int8 [N,K] -> packed
// D: asymmetric int8
// ============================================================================

/// Test runner for B-only-packed 1xN bias requantized GEMM
inline void run_b_only_packed_1xnbias_reqi32toi8_gemm_test(
    const GemmTestParams &params,
    const BOnlyPackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    // Generate fp32 data
    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);  // B is [N, K] (transposed)
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(n);
    bias_gen.fill_matrix(bias_fp32.data(), 1, n);

    // Compute quantization parameters from data
    float scale_a;
    int32_t zp_a;
    compute_asym_quant_params_i8(A_fp32.data(), m * k, scale_a, zp_a);

    std::vector<float> scale_b(n);
    for (size_t ch = 0; ch < n; ++ch) {
        compute_sym_quant_params_i8(&B_fp32[ch * k], k, scale_b[ch]);
    }

    // Quantize inputs
    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(n * k);  // [N, K]
    std::vector<int32_t> bias_i32(n);
    std::vector<int32_t> bias_fused(n);

    // Quantize A (global asymmetric)
    quantize_matrix_i8(A_fp32.data(), A_i8.data(), m * k, scale_a, zp_a);

    // Quantize B (per-channel symmetric, each row of B[N,K] is a channel)
    for (size_t ch = 0; ch < n; ++ch) {
        quantize_matrix_i8(&B_fp32[ch * k], &B_i8[ch * k], k, scale_b[ch], 0);
    }

    // Quantize bias (1xN, scale = scale_a * scale_b[ch], zp = 0)
    for (size_t ch = 0; ch < n; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        bias_i32[ch] = quantize_bias_i32(bias_fp32[ch], bias_scale);
    }

    // Fuse zero_point_a into bias: bias_fused[n] = bias[n] - zp_a * sum(B_i8[n, 0..K-1])
    fuse_zp_to_bias(bias_fused.data(), bias_i32.data(), B_i8.data(), zp_a, n, k);

    // Compute fp32 reference: D_ref = A[M,K] * B[N,K]^T + bias[1,N]
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), k, BLayout::kTransposed,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::k1xN,
                                 -FLT_MAX, FLT_MAX);

    // Compute D quantization parameters from reference result
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    // Set up quantization params
    struct tqt_qai8_qai8_qsi8cx_params qparams;
    qparams.scale_a = scale_a;
    qparams.zero_point_a = zp_a;
    qparams.scale_b = scale_b.data();
    qparams.quant_channel_b = static_cast<int32_t>(n);
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    // Pack B only (B is [N,K] transposed -> use bt_pack)
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_bt_pack(n, k, k, k, 0, static_cast<const void *>(B_i8.data()),
                      static_cast<void *>(B_packed.data()));

    // Execute kernel with original A and packed B
    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_offset(m_idx, 0, k);
            const size_t b_packed_offset = funcs.get_b_packed_offset(n_idx, 0, k, n);
            const size_t bias_offset = funcs.get_bias_offset(n_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A_i8.data()) + a_offset;
            const void *b_ptr = B_packed.data() + b_packed_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_fused.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_i8.data()) + d_offset;

            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    // Verify with relaxed tolerance for quantization error
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Function pointer types for quantized-weight GEMM (e.g., int4 weight + fp16
// activation). Templated on activation type TA and output/clamp type TD so
// that both same-precision (TA=TD, e.g. fp16->fp16) and mixed-precision
// (TA=fp16 -> TD=fp32) kernels share one struct.
// ============================================================================

template <typename TA, typename TD>
struct QuantizedWeightGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda, size_t bl);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb, size_t bl);
    using GetScaleBOffsetFunc = size_t (*)(size_t n_idx, size_t b_idx, size_t ldsb);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t n_idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *scale_b, size_t ldsb, const void *C, size_t ldc,
                                 void *D, size_t ldd, const void *bias, TD clamp_min, TD clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetScaleBOffsetFunc get_scale_b_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: quantized-weight GEMM
// A / scale_b: TA, B: int4 packed [N, K], D / C / bias: TD
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode>
void run_quantized_weight_gemm_test(const GemmTestParams &params,
                                    const QuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t bl = params.bl;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsb = k / bl;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<TA> a_gen(-1.0f, 1.0f, 42);
    UniformRandomGenerator<TD> d_gen(-5.0f, 5.0f, 42);

    // A [m, k] TA
    std::vector<TA> A(m * k);
    a_gen.fill_matrix(A.data(), m, k);

    // B: generate random TA, then quantize to int4 + scale (TA)
    std::vector<TA> B_src(n * k);
    UniformRandomGenerator<TA> b_gen(-1.0f, 1.0f, 123);
    b_gen.fill_matrix(B_src.data(), n, k);

    std::vector<uint8_t> B(n * (k / 2));
    std::vector<TA> scale_b(n * ldsb);
    quantize_fp16_to_int4(n, k, bl, B_src.data(), B.data(), scale_b.data(), ldsb);

    // C [m, n] TD (optional)
    std::vector<TD> C_data;
    const TD *c_ptr_ref = nullptr;
    if (params.has_c) {
        C_data.resize(m * n);
        d_gen.fill_matrix(C_data.data(), m, n);
        c_ptr_ref = C_data.data();
    }

    // Bias [n] TD (optional)
    std::vector<TD> bias_data;
    const TD *bias_ptr_ref = nullptr;
    BiasMode ref_bias_mode = BiasMode::kNone;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<TD> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
        bias_ptr_ref = bias_data.data();
        ref_bias_mode = kBiasMode;
    }

    const TD clamp_min = to_target_type<TD>(params.clamp_min_f);
    const TD clamp_max = to_target_type<TD>(params.clamp_max_f);

    // Reference
    std::vector<TD> ref_D(m * n);
    reference_gemm_quantized_weight_mixed<TA, TD>(
        m, n, k, bl, A.data(), lda, B.data(), ldb, scale_b.data(), ldsb, c_ptr_ref, ldc,
        ref_D.data(), ldd, bias_ptr_ref, ref_bias_mode, clamp_min, clamp_max);

    // Actual via micro-kernel
    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));

    funcs.run_gemm(m, n, k, bl, A.data(), lda, 0, B.data(), ldb, 0, scale_b.data(), ldsb,
                   params.has_c ? C_data.data() : nullptr, ldc, act_D.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min, clamp_max);

    // Verify (use TA's tolerance because internal accumulation is in TA precision).
    auto result =
        verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, DefaultThreshold<TA>::abs_error,
                               DefaultThreshold<TA>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << " BL=" << bl << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Function pointer types for packed quantized-weight GEMM
// (pre-packed activation + pre-packed int4 weight). Same TA/TD convention.
// ============================================================================

template <typename TA, typename TD>
struct PackedQuantizedWeightGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda,
                                            size_t actual_m);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetScaleBOffsetFunc = size_t (*)(size_t n_idx, size_t b_idx, size_t ldsb);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t n_idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
                                 size_t ldc, void *D, size_t ldd, const void *bias, TD clamp_min,
                                 TD clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetScaleBOffsetFunc get_scale_b_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: packed quantized-weight GEMM
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode>
void run_packed_quantized_weight_gemm_test(const GemmTestParams &params,
                                           const PackedQuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t bl = params.bl;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsb = k / bl;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<TA> a_gen(-1.0f, 1.0f, 42);
    UniformRandomGenerator<TD> d_gen(-5.0f, 5.0f, 42);

    // A [m, k] TA
    std::vector<TA> A(m * k);
    a_gen.fill_matrix(A.data(), m, k);

    // B: generate random TA, then quantize to int4 + scale (TA)
    std::vector<TA> B_src(n * k);
    UniformRandomGenerator<TA> b_gen(-1.0f, 1.0f, 123);
    b_gen.fill_matrix(B_src.data(), n, k);

    std::vector<uint8_t> B(n * (k / 2));
    std::vector<TA> scale_b(n * ldsb);
    quantize_fp16_to_int4(n, k, bl, B_src.data(), B.data(), scale_b.data(), ldsb);

    // C [m, n] TD (optional)
    std::vector<TD> C_data;
    if (params.has_c) {
        C_data.resize(m * n);
        d_gen.fill_matrix(C_data.data(), m, n);
    }

    // Bias [n] TD (optional)
    std::vector<TD> bias_data;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<TD> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
    }

    const TD clamp_min = to_target_type<TD>(params.clamp_min_f);
    const TD clamp_max = to_target_type<TD>(params.clamp_max_f);

    // Reference (using non-packed path)
    std::vector<TD> ref_D(m * n);
    reference_gemm_quantized_weight_mixed<TA, TD>(
        m, n, k, bl, A.data(), lda, B.data(), ldb, scale_b.data(), ldsb,
        params.has_c ? C_data.data() : nullptr, ldc, ref_D.data(), ldd,
        params.has_bias ? bias_data.data() : nullptr, kBiasMode, clamp_min, clamp_max);

    // Pack A
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    funcs.run_a_pack(m, k, lda, k, 0, A.data(), A_packed.data());

    // Pack B (transposed int4)
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_bt_pack(n, k, ldb, k, 0, B.data(), B_packed.data());

    // Run packed GEMM
    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));
    funcs.run_gemm(m, n, k, bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0, scale_b.data(),
                   ldsb, params.has_c ? C_data.data() : nullptr, ldc, act_D.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min, clamp_max);

    // Verify (use TA's tolerance — internal accumulation is in TA precision).
    auto result =
        verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, DefaultThreshold<TA>::abs_error,
                               DefaultThreshold<TA>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << " BL=" << bl << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Function pointer types for fp-output non-packed / packed variants
//   gemm_1xnbias_clamp_<T_D>_qai8dx_qsi8cxt   (NonPacked, a_bt)
//   gemm_1xnbias_clamp_<T_D>_qai8dxp_qsi8cxp  (Packed, ap_bp)
// where T_D is float or float16_t. Bias is the same type as T_D.
// ============================================================================

template <typename T_D>
struct NonPackedClampFpGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 T_D clamp_min, T_D clamp_max,
                                 const tqt_qai8dxp_qsi8cxp_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    RunGemmFunc run_gemm;
};

template <typename T_D>
struct PackedClampFpGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda,
                                            size_t actual_m);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBTPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 T_D clamp_min, T_D clamp_max,
                                 const tqt_qai8dxp_qsi8cxp_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBTPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Test runner: gemm_1xnbias_clamp_<T_D>_qai8dx_qsi8cxt (NonPacked a_bt)
//   A: per-row asymmetric int8 [M, K]
//   B: per-channel symmetric int8 [N, K] (transposed view)
//   D: T_D output
// ============================================================================

template <typename T_D>
inline void run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test(
    const GemmTestParams &params, const NonPackedClampFpGemmFuncs<T_D> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);  // [N, K] transposed
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(n);
    bias_gen.fill_matrix(bias_fp32.data(), 1, n);

    // Per-row asymmetric int8 quantization for A
    std::vector<float> scale_a_vec(m);
    std::vector<int32_t> zp_a_vec(m);
    for (size_t row = 0; row < m; ++row) {
        compute_asym_quant_params_i8(&A_fp32[row * k], k, scale_a_vec[row], zp_a_vec[row]);
    }

    // Per-channel symmetric int8 quantization for B
    std::vector<float> scale_b_vec(n);
    for (size_t ch = 0; ch < n; ++ch) {
        compute_sym_quant_params_i8(&B_fp32[ch * k], k, scale_b_vec[ch]);
    }

    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(n * k);
    for (size_t row = 0; row < m; ++row) {
        quantize_matrix_i8(&A_fp32[row * k], &A_i8[row * k], k, scale_a_vec[row], zp_a_vec[row]);
    }
    for (size_t ch = 0; ch < n; ++ch) {
        quantize_matrix_i8(&B_fp32[ch * k], &B_i8[ch * k], k, scale_b_vec[ch], 0);
    }

    // fp32 reference
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), k, BLayout::kTransposed,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::k1xN,
                                 -FLT_MAX, FLT_MAX);

    // Convert bias to target type T_D
    std::vector<T_D> bias_TD(n);
    for (size_t j = 0; j < n; ++j) {
        bias_TD[j] = to_target_type<T_D>(bias_fp32[j]);
    }

    tqt_qai8dxp_qsi8cxp_params qparams;
    qparams.scale_a = scale_a_vec.data();
    qparams.zero_point_a = zp_a_vec.data();
    qparams.scale_b = scale_b_vec.data();

    const T_D clamp_min = to_target_type<T_D>(-FLT_MAX);
    const T_D clamp_max = to_target_type<T_D>(FLT_MAX);

    std::vector<T_D> D_act(m * n, to_target_type<T_D>(0.0f));

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_offset(m_idx, 0, k);
            const size_t b_offset = funcs.get_b_offset(n_idx, 0, k);
            const size_t bias_offset = funcs.get_bias_offset(n_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A_i8.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const char *>(B_i8.data()) + b_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_TD.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Convert kernel output to fp32 for comparison
    std::vector<float> D_act_fp32(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_act_fp32[i] = to_float<T_D>(D_act[i]);
    }

    // Per-row asymmetric quant + per-channel symmetric quant + K-accumulation noise:
    // typical absolute error scales roughly with K and the dynamic range of D.
    // Use a generous absolute tolerance and rely on cosine similarity for sanity.
    const float tolerance = std::is_same<T_D, float>::value ? 1.5f : 2.5f;
    const float cosine_threshold = std::is_same<T_D, float>::value ? 0.998f : 0.997f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Test runner: gemm_1xnbias_clamp_<T_D>_qai8dxp_qsi8cxp (Packed ap_bp)
// ============================================================================

template <typename T_D>
inline void run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test(
    const GemmTestParams &params, const PackedClampFpGemmFuncs<T_D> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    UniformRandomGenerator<float> bias_gen(-10.0f, 10.0f, 123);
    std::vector<float> bias_fp32(n);
    bias_gen.fill_matrix(bias_fp32.data(), 1, n);

    std::vector<float> scale_a_vec(m);
    std::vector<int32_t> zp_a_vec(m);
    for (size_t row = 0; row < m; ++row) {
        compute_asym_quant_params_i8(&A_fp32[row * k], k, scale_a_vec[row], zp_a_vec[row]);
    }

    std::vector<float> scale_b_vec(n);
    for (size_t ch = 0; ch < n; ++ch) {
        compute_sym_quant_params_i8(&B_fp32[ch * k], k, scale_b_vec[ch]);
    }

    std::vector<int8_t> A_i8(m * k);
    std::vector<int8_t> B_i8(n * k);
    for (size_t row = 0; row < m; ++row) {
        quantize_matrix_i8(&A_fp32[row * k], &A_i8[row * k], k, scale_a_vec[row], zp_a_vec[row]);
    }
    for (size_t ch = 0; ch < n; ++ch) {
        quantize_matrix_i8(&B_fp32[ch * k], &B_i8[ch * k], k, scale_b_vec[ch], 0);
    }

    // Pack A and B
    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));
    funcs.run_a_pack(m, k, k, k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(n, k, k, k, 0, B_i8.data(), B_packed.data());

    // fp32 reference
    std::vector<float> D_ref_fp32(m * n);
    reference_gemm<float, float>(m, n, k, A_fp32.data(), k, B_fp32.data(), k, BLayout::kTransposed,
                                 nullptr, n, D_ref_fp32.data(), n, bias_fp32.data(), BiasMode::k1xN,
                                 -FLT_MAX, FLT_MAX);

    std::vector<T_D> bias_TD(n);
    for (size_t j = 0; j < n; ++j) {
        bias_TD[j] = to_target_type<T_D>(bias_fp32[j]);
    }

    tqt_qai8dxp_qsi8cxp_params qparams;
    qparams.scale_a = scale_a_vec.data();
    qparams.zero_point_a = zp_a_vec.data();
    qparams.scale_b = scale_b_vec.data();

    const T_D clamp_min = to_target_type<T_D>(-FLT_MAX);
    const T_D clamp_max = to_target_type<T_D>(FLT_MAX);

    std::vector<T_D> D_act(m * n, to_target_type<T_D>(0.0f));

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_offset = funcs.get_a_packed_offset(m_idx, 0, k, actual_m);
            const size_t b_offset = funcs.get_b_packed_offset(n_idx, 0, k, actual_n);
            const size_t bias_offset = funcs.get_bias_offset(n_idx);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, n);

            const void *a_ptr = reinterpret_cast<const char *>(A_packed.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const char *>(B_packed.data()) + b_offset;
            const void *bias_ptr = reinterpret_cast<const char *>(bias_TD.data()) + bias_offset;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    std::vector<float> D_act_fp32(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_act_fp32[i] = to_float<T_D>(D_act[i]);
    }

    const float tolerance = std::is_same<T_D, float>::value ? 1.5f : 2.5f;
    const float cosine_threshold = std::is_same<T_D, float>::value ? 0.998f : 0.997f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

}  // namespace test
}  // namespace torq_tile
