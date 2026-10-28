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
#include <limits>
#include <type_traits>
#include <vector>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
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
            const void *bias_ptr = reinterpret_cast<const char *>(bias_i32.data()) + bias_offset;
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
            const void *bias_ptr = reinterpret_cast<const char *>(bias_i32.data()) + bias_offset;
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

/// Verify packed int8 split-K accumulation with a getter-selected non-zero K origin.
inline void run_packed_clamp_i32_i8_split_k_test(size_t m, size_t n, size_t k, size_t split,
                                                 const PackedGemmFuncs<int32_t> &funcs)
{
    ASSERT_GT(split, 0U);
    ASSERT_LT(split, k);

    std::vector<int8_t> A(m * k), B(k * n);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<int8_t>((i * 5 + 1) % 7 - 3);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = static_cast<int8_t>((i * 3 + 2) % 7 - 3);

    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));
    funcs.run_a_pack(m, k, k, k, 0, A.data(), A_packed.data());
    funcs.run_b_pack(n, k, n, k, 0, B.data(), B_packed.data());

    constexpr int32_t clamp_min = std::numeric_limits<int32_t>::min();
    constexpr int32_t clamp_max = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> whole(m * n), split_result(m * n);
    funcs.run_gemm(m, n, k, A_packed.data(), k, 0, B_packed.data(), k, 0, nullptr, n, whole.data(),
                   n, nullptr, clamp_min, clamp_max);
    funcs.run_gemm(m, n, split, A_packed.data(), k, 0, B_packed.data(), k, 0, nullptr, n,
                   split_result.data(), n, nullptr, clamp_min, clamp_max);

    const size_t a_offset = funcs.get_a_packed_offset(0, split, k, m);
    const size_t b_offset = funcs.get_b_packed_offset(0, split, k, n);
    funcs.run_gemm(m, n, k - split, A_packed.data() + a_offset, k, split,
                   B_packed.data() + b_offset, k, split, split_result.data(), n,
                   split_result.data(), n, nullptr, clamp_min, clamp_max);

    SCOPED_TRACE("packed int8 split-K with nonzero origin");
    auto result =
        verify_gemm_result<int32_t>(whole.data(), split_result.data(), m * n, 0.0f, 0.999999f);
    EXPECT_TRUE(result.passed) << result.error_message;
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
            const void *bias_ptr = reinterpret_cast<const char *>(bias_i32.data()) + bias_offset;
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
            const void *bias_ptr = reinterpret_cast<const char *>(bias_i32.data()) + bias_offset;
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
            const void *bias_ptr = reinterpret_cast<const char *>(bias_i32.data()) + bias_offset;
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

/// Compare compact and getter-selected packed suffix windows for the three requantized int8
/// layouts. Requantized output cannot be accumulated across split-K calls, so this differential
/// form isolates K-origin addressing without changing the mathematical operation.
inline void run_packed_requantized_i8_nonzero_origin_test(
    size_t m, size_t n, size_t k, size_t k_origin,
    const PackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params> &one_by_n_funcs,
    const BOnlyPackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params> &b_only_funcs,
    const PackedRequantizedGemmFuncs<tqt_qai8_qsi8dx_qai8_params> &m_by_one_funcs)
{
    ASSERT_GT(k_origin, 0U);
    ASSERT_LT(k_origin, k);
    const size_t window = k - k_origin;

    std::vector<int8_t> A(m * k), B_transposed(n * k), B(k * n);
    for (size_t row = 0; row < m; ++row) {
        for (size_t depth = 0; depth < k; ++depth)
            A[row * k + depth] = static_cast<int8_t>((row + depth * 2) % 3 - 1);
    }
    for (size_t column = 0; column < n; ++column) {
        for (size_t depth = 0; depth < k; ++depth) {
            const int8_t value = static_cast<int8_t>((column * 2 + depth) % 3 - 1);
            B_transposed[column * k + depth] = value;
            B[depth * n + column] = value;
        }
    }

    std::vector<int8_t> A_window(m * window), B_transposed_window(n * window), B_window(window * n);
    for (size_t row = 0; row < m; ++row) {
        std::copy_n(A.data() + row * k + k_origin, window, A_window.data() + row * window);
    }
    for (size_t column = 0; column < n; ++column) {
        std::copy_n(B_transposed.data() + column * k + k_origin, window,
                    B_transposed_window.data() + column * window);
    }
    std::copy(B.begin() + k_origin * n, B.end(), B_window.begin());

    std::vector<float> scale_a(m, 1.0f), scale_b(n, 1.0f);
    std::vector<int32_t> bias_one_by_n(n, 0), bias_m_by_one(m, 0);
    const tqt_qai8_qai8_qsi8cx_params one_by_n_params = {
        1.0f, 0, scale_b.data(), static_cast<int32_t>(n), 1.0f, 0};
    const tqt_qai8_qsi8dx_qai8_params m_by_one_params = {
        scale_a.data(), static_cast<int32_t>(m), 1.0f, 0, 1.0f, 0};
    constexpr int8_t clamp_min = -128;
    constexpr int8_t clamp_max = 127;

    auto verify_equal = [&](const std::vector<int8_t> &compact, const std::vector<int8_t> &selected,
                            const char *layout) {
        SCOPED_TRACE(layout);
        EXPECT_EQ(compact, selected);
    };

    // Fully packed A and transposed B, 1xN bias.
    {
        std::vector<uint8_t> A_full(one_by_n_funcs.get_a_packed_size(m, k));
        std::vector<uint8_t> B_full(one_by_n_funcs.get_b_packed_size(n, k));
        std::vector<uint8_t> A_compact(one_by_n_funcs.get_a_packed_size(m, window));
        std::vector<uint8_t> B_compact(one_by_n_funcs.get_b_packed_size(n, window));
        one_by_n_funcs.run_a_pack(m, k, k, k, 0, A.data(), A_full.data());
        one_by_n_funcs.run_bt_pack(n, k, k, k, 0, B_transposed.data(), B_full.data());
        one_by_n_funcs.run_a_pack(m, window, window, window, 0, A_window.data(), A_compact.data());
        one_by_n_funcs.run_bt_pack(n, window, window, window, 0, B_transposed_window.data(),
                                   B_compact.data());
        const size_t a_offset = one_by_n_funcs.get_a_packed_offset(0, k_origin, k, m);
        const size_t b_offset = one_by_n_funcs.get_b_packed_offset(0, k_origin, k, n);
        std::vector<int8_t> compact(m * n), selected(m * n);
        one_by_n_funcs.run_gemm(m, n, window, A_compact.data(), window, 0, B_compact.data(), window,
                                0, nullptr, n, compact.data(), n, bias_one_by_n.data(), clamp_min,
                                clamp_max, &one_by_n_params);
        one_by_n_funcs.run_gemm(m, n, window, A_full.data() + a_offset, k, k_origin,
                                B_full.data() + b_offset, k, k_origin, nullptr, n, selected.data(),
                                n, bias_one_by_n.data(), clamp_min, clamp_max, &one_by_n_params);
        verify_equal(compact, selected, "fully packed 1xN requantized int8");
    }

    // Raw A and packed transposed B, 1xN bias.
    {
        std::vector<uint8_t> B_full(b_only_funcs.get_b_packed_size(n, k));
        std::vector<uint8_t> B_compact(b_only_funcs.get_b_packed_size(n, window));
        b_only_funcs.run_bt_pack(n, k, k, k, 0, B_transposed.data(), B_full.data());
        b_only_funcs.run_bt_pack(n, window, window, window, 0, B_transposed_window.data(),
                                 B_compact.data());
        const size_t b_offset = b_only_funcs.get_b_packed_offset(0, k_origin, k, n);
        std::vector<int8_t> compact(m * n), selected(m * n);
        b_only_funcs.run_gemm(m, n, window, A_window.data(), window, 0, B_compact.data(), window, 0,
                              nullptr, n, compact.data(), n, bias_one_by_n.data(), clamp_min,
                              clamp_max, &one_by_n_params);
        b_only_funcs.run_gemm(m, n, window, A.data() + k_origin, k, k_origin,
                              B_full.data() + b_offset, k, k_origin, nullptr, n, selected.data(), n,
                              bias_one_by_n.data(), clamp_min, clamp_max, &one_by_n_params);
        verify_equal(compact, selected, "B-only packed 1xN requantized int8");
    }

    // Fully packed A and row-major B, Mx1 bias.
    {
        std::vector<uint8_t> A_full(m_by_one_funcs.get_a_packed_size(m, k));
        std::vector<uint8_t> B_full(m_by_one_funcs.get_b_packed_size(n, k));
        std::vector<uint8_t> A_compact(m_by_one_funcs.get_a_packed_size(m, window));
        std::vector<uint8_t> B_compact(m_by_one_funcs.get_b_packed_size(n, window));
        m_by_one_funcs.run_a_pack(m, k, k, k, 0, A.data(), A_full.data());
        m_by_one_funcs.run_b_pack(n, k, n, k, 0, B.data(), B_full.data());
        m_by_one_funcs.run_a_pack(m, window, window, window, 0, A_window.data(), A_compact.data());
        m_by_one_funcs.run_b_pack(n, window, n, window, 0, B_window.data(), B_compact.data());
        const size_t a_offset = m_by_one_funcs.get_a_packed_offset(0, k_origin, k, m);
        const size_t b_offset = m_by_one_funcs.get_b_packed_offset(0, k_origin, k, n);
        std::vector<int8_t> compact(m * n), selected(m * n);
        m_by_one_funcs.run_gemm(m, n, window, A_compact.data(), window, 0, B_compact.data(), window,
                                0, nullptr, n, compact.data(), n, bias_m_by_one.data(), clamp_min,
                                clamp_max, &m_by_one_params);
        m_by_one_funcs.run_gemm(m, n, window, A_full.data() + a_offset, k, k_origin,
                                B_full.data() + b_offset, k, k_origin, nullptr, n, selected.data(),
                                n, bias_m_by_one.data(), clamp_min, clamp_max, &m_by_one_params);
        verify_equal(compact, selected, "fully packed Mx1 requantized int8");
    }
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

    if (k > bl) {
        const size_t k_origin = bl;
        const size_t window = k - k_origin;
        std::vector<TA> A_window(m * window);
        std::vector<uint8_t> B_window(n * (window / 2));
        for (size_t row = 0; row < m; ++row) {
            std::copy_n(A.data() + row * k + k_origin, window, A_window.data() + row * window);
        }
        for (size_t row = 0; row < n; ++row) {
            std::copy_n(B.data() + row * (k / 2) + k_origin / 2, window / 2,
                        B_window.data() + row * (window / 2));
        }

        std::vector<TD> zero_origin(m * n, to_target_type<TD>(0.0f));
        std::vector<TD> nonzero_origin(m * n, to_target_type<TD>(0.0f));
        const size_t scale_b_offset = funcs.get_scale_b_offset(0, k_origin / bl, ldsb);
        const auto *scale_b_window = reinterpret_cast<const TA *>(
            reinterpret_cast<const uint8_t *>(scale_b.data()) + scale_b_offset);
        funcs.run_gemm(m, n, window, bl, A_window.data(), window, 0, B_window.data(), window, 0,
                       scale_b_window, ldsb, nullptr, n, zero_origin.data(), n, nullptr,
                       to_target_type<TD>(-FLT_MAX), to_target_type<TD>(FLT_MAX));
        funcs.run_gemm(m, n, window, bl, A.data() + k_origin, k, k_origin, B.data() + k_origin / 2,
                       k, k_origin, scale_b_window, ldsb, nullptr, n, nonzero_origin.data(), n,
                       nullptr, to_target_type<TD>(-FLT_MAX), to_target_type<TD>(FLT_MAX));

        SCOPED_TRACE("nonzero quantized-weight K origin");
        auto origin_result = verify_gemm_result<TD>(zero_origin.data(), nonzero_origin.data(),
                                                    m * n, DefaultThreshold<TA>::abs_error,
                                                    DefaultThreshold<TA>::cosine_sim);
        EXPECT_TRUE(origin_result.passed) << origin_result.error_message;
        if (gemm_clamp_probe_active) {
            SCOPED_TRACE("special-value quantized-weight split-K propagation");
            std::vector<TD> partial(m * n), combined(m * n);
            funcs.run_gemm(m, n, k_origin, bl, A.data(), k, 0, B.data(), k, 0, scale_b.data(), ldsb,
                           nullptr, n, partial.data(), n, nullptr, to_target_type<TD>(-INFINITY),
                           to_target_type<TD>(INFINITY));
            funcs.run_gemm(m, n, window, bl, A.data() + k_origin, k, k_origin,
                           B.data() + k_origin / 2, k, k_origin, scale_b_window, ldsb,
                           partial.data(), n, combined.data(), n, nullptr,
                           to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY));
        }
    }
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

    if (k > bl) {
        const size_t k_origin = bl;
        const size_t window = k - k_origin;
        std::vector<TA> A_window(m * window);
        std::vector<uint8_t> B_window(n * (window / 2));
        for (size_t row = 0; row < m; ++row) {
            std::copy_n(A.data() + row * k + k_origin, window, A_window.data() + row * window);
        }
        for (size_t row = 0; row < n; ++row) {
            std::copy_n(B.data() + row * (k / 2) + k_origin / 2, window / 2,
                        B_window.data() + row * (window / 2));
        }

        const size_t a_offset = funcs.get_a_packed_offset(0, k_origin, k, m);
        const size_t b_offset = funcs.get_b_packed_offset(0, k_origin, k, n);
        const size_t scale_b_offset = funcs.get_scale_b_offset(0, k_origin / bl, ldsb);
        const auto *scale_b_window = reinterpret_cast<const TA *>(
            reinterpret_cast<const uint8_t *>(scale_b.data()) + scale_b_offset);

        // A getter-selected window in a full packed buffer must have the same tile-local
        // layout as packing that window into a compact buffer. Check both A tiles and the
        // one- or two-subgroup B layout independently from the numerical GEMM result.
        std::vector<uint8_t> A_window_packed(funcs.get_a_packed_size(m, window));
        std::vector<uint8_t> B_window_packed(funcs.get_b_packed_size(n, window));
        funcs.run_a_pack(m, window, window, window, 0, A_window.data(), A_window_packed.data());
        funcs.run_bt_pack(n, window, window, window, 0, B_window.data(), B_window_packed.data());

        const size_t m_step = funcs.get_m_step();
        for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const uint8_t *full_tile =
                A_packed.data() + funcs.get_a_packed_offset(m_idx, k_origin, k, actual_m);
            const uint8_t *window_tile =
                A_window_packed.data() + funcs.get_a_packed_offset(m_idx, 0, window, actual_m);
            EXPECT_TRUE(
                std::equal(full_tile, full_tile + actual_m * window * sizeof(TA), window_tile))
                << "Packed A window mismatch at m_idx=" << m_idx;
        }

        const size_t n_step = funcs.get_n_step();
        const size_t vlmax = n_step / 2;
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_n = std::min(n - n_idx, n_step);
            const size_t first_n =
                actual_n >= n_step ? n_step : (actual_n >= vlmax ? vlmax : actual_n);
            const uint8_t *full_tile =
                B_packed.data() + funcs.get_b_packed_offset(n_idx, k_origin, k, actual_n);
            const uint8_t *window_tile =
                B_window_packed.data() + funcs.get_b_packed_offset(n_idx, 0, window, actual_n);
            EXPECT_TRUE(std::equal(full_tile, full_tile + first_n * (window / 2), window_tile))
                << "Packed B first subgroup mismatch at n_idx=" << n_idx;

            const size_t remain_n = actual_n - first_n;
            if (remain_n > 0) {
                const uint8_t *full_remain =
                    full_tile + first_n * (k / 2) - (k_origin / 2) * (first_n - remain_n);
                const uint8_t *window_remain = window_tile + first_n * (window / 2);
                EXPECT_TRUE(
                    std::equal(full_remain, full_remain + remain_n * (window / 2), window_remain))
                    << "Packed B second subgroup mismatch at n_idx=" << n_idx;
            }
        }

        std::vector<TD> zero_origin(m * n, to_target_type<TD>(0.0f));
        std::vector<TD> zero_origin_packed(m * n, to_target_type<TD>(0.0f));
        std::vector<TD> nonzero_origin(m * n, to_target_type<TD>(0.0f));
        reference_gemm_quantized_weight_mixed<TA, TD>(
            m, n, window, bl, A_window.data(), window, B_window.data(), window, scale_b.data() + 1,
            ldsb, nullptr, n, zero_origin.data(), n, nullptr, BiasMode::kNone,
            to_target_type<TD>(-FLT_MAX), to_target_type<TD>(FLT_MAX));
        funcs.run_gemm(m, n, window, bl, A_window_packed.data(), window, 0, B_window_packed.data(),
                       window, 0, scale_b_window, ldsb, nullptr, n, zero_origin_packed.data(), n,
                       nullptr, to_target_type<TD>(-FLT_MAX), to_target_type<TD>(FLT_MAX));
        funcs.run_gemm(m, n, window, bl, A_packed.data() + a_offset, k, k_origin,
                       B_packed.data() + b_offset, k, k_origin, scale_b_window, ldsb, nullptr, n,
                       nonzero_origin.data(), n, nullptr, to_target_type<TD>(-FLT_MAX),
                       to_target_type<TD>(FLT_MAX));

        SCOPED_TRACE("nonzero packed quantized-weight K origin");
        auto origin_result = verify_gemm_result<TD>(zero_origin.data(), nonzero_origin.data(),
                                                    m * n, DefaultThreshold<TA>::abs_error,
                                                    DefaultThreshold<TA>::cosine_sim);
        EXPECT_TRUE(origin_result.passed) << origin_result.error_message;
        auto packed_origin_result = verify_gemm_result<TD>(
            zero_origin_packed.data(), nonzero_origin.data(), m * n,
            DefaultThreshold<TA>::abs_error, DefaultThreshold<TA>::cosine_sim);
        EXPECT_TRUE(packed_origin_result.passed)
            << "Compact and nonzero-origin packed GEMM differ:\n"
            << packed_origin_result.error_message;
        if (gemm_clamp_probe_active) {
            SCOPED_TRACE("special-value quantized-weight split-K propagation");
            std::vector<TD> partial(m * n), combined(m * n);
            funcs.run_gemm(m, n, k_origin, bl, A_packed.data(), k, 0, B_packed.data(), k, 0,
                           scale_b.data(), ldsb, nullptr, n, partial.data(), n, nullptr,
                           to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY));
            funcs.run_gemm(m, n, window, bl, A_packed.data() + a_offset, k, k_origin,
                           B_packed.data() + b_offset, k, k_origin, scale_b_window, ldsb,
                           partial.data(), n, combined.data(), n, nullptr,
                           to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY));
        }
    }
}

// ============================================================================
// Test runner: packed block-quantized byte-weight GEMM
// ============================================================================

template <typename InputType, typename OutputType, typename QParamType, typename QParamsPack,
          typename RunGemm>
void run_packed_block_quantized_byte_gemm_test(
    size_t block_size, bool asymmetric, size_t (*get_m_step)(void), size_t (*get_n_step)(void),
    size_t (*get_a_packed_size)(size_t, size_t), size_t (*get_b_packed_size)(size_t, size_t),
    size_t (*get_qparams_packed_size)(size_t, size_t),
    size_t (*get_a_packed_offset)(size_t, size_t, size_t, size_t),
    size_t (*get_b_packed_offset)(size_t, size_t, size_t, size_t),
    size_t (*get_qparams_packed_offset)(size_t, size_t, size_t, size_t),
    void (*run_a_pack)(size_t, size_t, size_t, size_t, size_t, const void *, void *),
    void (*run_bt_pack)(size_t, size_t, size_t, size_t, size_t, const void *, void *),
    QParamsPack run_qparams_pack, RunGemm run_gemm, float tolerance)
{
    const size_t m_step = get_m_step();
    const size_t n_step = get_n_step();
    ASSERT_GT(m_step, 1U);
    ASSERT_GT(n_step, 1U);

    const size_t shapes[][2] = {
        {3, 5},
        {m_step - 1, n_step - 1},
        {m_step + 1, n_step + 1},
    };
    const size_t k = 2 * block_size + 1;
    const size_t blocks = (k + block_size - 1) / block_size;

    for (const auto &shape : shapes) {
        const size_t m = shape[0];
        const size_t n = shape[1];
        std::vector<InputType> A(m * k);
        std::vector<OutputType> bias(n);
        std::vector<int8_t> B(n * k);
        std::vector<QParamType> scale(n * blocks), real_zero(n * blocks);
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = (InputType)(((int)(i % 9) - 4) / 32.0f);
        for (size_t i = 0; i < n; ++i)
            bias[i] = (OutputType)(((int)(i % 5) - 2) / 64.0f);
        for (size_t i = 0; i < B.size(); ++i) {
            if (asymmetric) {
                B[i] = (int8_t)((i * 5 + 3) & 31);
            } else if (block_size == 16) {
                B[i] = (int8_t)((int)(i * 5 + 3) % 64 - 32);
            } else {
                B[i] = (int8_t)((int)(i * 7 + 5) % 256 - 128);
            }
        }
        for (size_t i = 0; i < scale.size(); ++i) {
            scale[i] = (QParamType)(1.0f / 64.0f + (i % 3) / 512.0f);
            real_zero[i] = (QParamType)((float)scale[i] * (12.0f + (float)(i & 1)));
        }

        std::vector<float> reference(m * n);
        for (size_t row = 0; row < m; ++row) {
            for (size_t column = 0; column < n; ++column) {
                if constexpr (std::is_same<OutputType, float>::value) {
                    float accumulator = bias[column];
                    for (size_t depth = 0; depth < k; ++depth) {
                        const size_t block = depth / block_size;
                        const QParamType weight =
                            asymmetric ? (QParamType)((QParamType)(uint8_t)B[column * k + depth] *
                                                          scale[column * blocks + block] -
                                                      real_zero[column * blocks + block])
                                       : (QParamType)((QParamType)B[column * k + depth] *
                                                      scale[column * blocks + block]);
                        accumulator += (float)(float16_t)A[row * k + depth] * (float)weight;
                    }
                    reference[row * n + column] = accumulator;
                } else {
                    OutputType accumulator = bias[column];
                    for (size_t depth = 0; depth < k; ++depth) {
                        const size_t block = depth / block_size;
                        const OutputType weight =
                            asymmetric
                                ? (OutputType)((OutputType)(uint8_t)B[column * k + depth] *
                                                   (OutputType)scale[column * blocks + block] -
                                               (OutputType)real_zero[column * blocks + block])
                                : (OutputType)((OutputType)B[column * k + depth] *
                                               (OutputType)scale[column * blocks + block]);
                        accumulator =
                            (OutputType)(accumulator +
                                         (OutputType)((OutputType)A[row * k + depth] * weight));
                    }
                    reference[row * n + column] = (float)accumulator;
                }
            }
        }

        std::vector<uint8_t> A_packed(get_a_packed_size(m, k));
        std::vector<uint8_t> B_packed(get_b_packed_size(n, k));
        std::vector<uint8_t> qparams_packed(get_qparams_packed_size(n, k));
        run_a_pack(m, k, k, k, 0, A.data(), A_packed.data());
        run_bt_pack(n, k, k, k, 0, B.data(), B_packed.data());
        run_qparams_pack(n, k, blocks, k, 0, scale.data(), asymmetric ? real_zero.data() : nullptr,
                         qparams_packed.data());

        std::vector<OutputType> D(m * n);
        run_gemm(m, n, k, A_packed.data(), k, 0, B_packed.data(), k, 0, nullptr, n, D.data(), n,
                 bias.data(), qparams_packed.data());
        for (size_t i = 0; i < D.size(); ++i) {
            EXPECT_NEAR((float)D[i], reference[i], tolerance)
                << "index=" << i << " m=" << m << " n=" << n;
        }

        std::vector<OutputType> partial(m * n);
        run_gemm(m, n, block_size, A_packed.data(), k, 0, B_packed.data(), k, 0, nullptr, n,
                 partial.data(), n, nullptr, qparams_packed.data());
        const size_t a_offset = get_a_packed_offset(0, block_size, k, m);
        const size_t b_offset = get_b_packed_offset(0, block_size, k, n);
        const size_t qparams_offset = get_qparams_packed_offset(0, block_size, k, n);
        run_gemm(m, n, k - block_size, A_packed.data() + a_offset, k, block_size,
                 B_packed.data() + b_offset, k, block_size, partial.data(), n, D.data(), n,
                 bias.data(), qparams_packed.data() + qparams_offset);
        for (size_t i = 0; i < D.size(); ++i) {
            EXPECT_NEAR((float)D[i], reference[i], tolerance)
                << "split index=" << i << " m=" << m << " n=" << n;
        }
    }
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

// Check the quantized matrix equation separately from the original-F32 reference.
// Its tight tolerance catches incorrect metadata windows hidden by quantization noise.
template <typename T_D, typename BiasT>
inline void check_clamp_fp_i8_quantized_output(size_t m, size_t n, size_t k, const int8_t *a,
                                               const int8_t *bt, const BiasT *bias,
                                               const tqt_qai8dxp_qsi8cxp_params &params,
                                               const T_D *output)
{
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            int64_t dot = 0;
            for (size_t q = 0; q < k; ++q)
                dot += (int64_t(a[row * k + q]) - params.zero_point_a[row]) * bt[col * k + q];
            double product = double(dot) * params.scale_a[row] * params.scale_b[col];
            // Existing narrow-bias variants narrow the product before adding bias.
            if constexpr (!std::is_same<BiasT, float>::value)
                product = to_float<T_D>(to_target_type<T_D>(product));
            const double real = product + double(to_float<BiasT>(bias[col]));
            const float expected = to_float<T_D>(to_target_type<T_D>(real));
            const float relative_tolerance = std::is_same<T_D, float>::value ? 1e-5f : 1e-3f;
            EXPECT_NEAR(to_float<T_D>(output[row * n + col]), expected,
                        std::max(1e-5f, std::abs(expected) * relative_tolerance))
                << "row=" << row << " col=" << col;
        }
    }
}

template <typename T_D, typename BiasT = float>
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
    std::vector<BiasT> bias_data(n);
    for (size_t j = 0; j < n; ++j) {
        bias_data[j] = static_cast<BiasT>(bias_fp32[j]);
        bias_fp32[j] = static_cast<float>(bias_data[j]);
    }

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
            const BiasT *bias_ptr = bias_data.data() + bias_offset / sizeof(BiasT);
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            auto tile_params = qparams;
            tile_params.scale_a += m_idx;
            tile_params.zero_point_a += m_idx;
            tile_params.scale_b += n_idx;
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &tile_params);
        }
    }

    check_clamp_fp_i8_quantized_output(m, n, k, A_i8.data(), B_i8.data(), bias_data.data(), qparams,
                                       D_act.data());

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

template <typename T_D, typename BiasT = float>
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
    std::vector<BiasT> bias_data(n);
    for (size_t j = 0; j < n; ++j) {
        bias_data[j] = static_cast<BiasT>(bias_fp32[j]);
        bias_fp32[j] = static_cast<float>(bias_data[j]);
    }

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
            const BiasT *bias_ptr = bias_data.data() + bias_offset / sizeof(BiasT);
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_offset;

            auto tile_params = qparams;
            tile_params.scale_a += m_idx;
            tile_params.zero_point_a += m_idx;
            tile_params.scale_b += n_idx;
            funcs.run_gemm(actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, nullptr, n, d_ptr, n,
                           bias_ptr, clamp_min, clamp_max, &tile_params);
        }
    }

    check_clamp_fp_i8_quantized_output(m, n, k, A_i8.data(), B_i8.data(), bias_data.data(), qparams,
                                       D_act.data());

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

/// Compare compact and getter-selected packed suffix windows for floating-output int8 GEMM.
template <typename T_D>
inline void run_packed_clamp_fp_i8_nonzero_origin_test(size_t m, size_t n, size_t k,
                                                       size_t k_origin,
                                                       const PackedClampFpGemmFuncs<T_D> &funcs)
{
    ASSERT_GT(k_origin, 0U);
    ASSERT_LT(k_origin, k);
    const size_t window = k - k_origin;

    std::vector<int8_t> A(m * k), B(n * k);
    for (size_t row = 0; row < m; ++row) {
        for (size_t depth = 0; depth < k; ++depth)
            A[row * k + depth] = static_cast<int8_t>((row + depth * 2) % 3 - 1);
    }
    for (size_t row = 0; row < n; ++row) {
        for (size_t depth = 0; depth < k; ++depth)
            B[row * k + depth] = static_cast<int8_t>((row * 2 + depth) % 3 - 1);
    }

    std::vector<int8_t> A_window(m * window), B_window(n * window);
    for (size_t row = 0; row < m; ++row) {
        std::copy_n(A.data() + row * k + k_origin, window, A_window.data() + row * window);
    }
    for (size_t row = 0; row < n; ++row) {
        std::copy_n(B.data() + row * k + k_origin, window, B_window.data() + row * window);
    }

    std::vector<uint8_t> A_full(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_full(funcs.get_b_packed_size(n, k));
    std::vector<uint8_t> A_compact(funcs.get_a_packed_size(m, window));
    std::vector<uint8_t> B_compact(funcs.get_b_packed_size(n, window));
    funcs.run_a_pack(m, k, k, k, 0, A.data(), A_full.data());
    funcs.run_bt_pack(n, k, k, k, 0, B.data(), B_full.data());
    funcs.run_a_pack(m, window, window, window, 0, A_window.data(), A_compact.data());
    funcs.run_bt_pack(n, window, window, window, 0, B_window.data(), B_compact.data());

    const size_t a_offset = funcs.get_a_packed_offset(0, k_origin, k, m);
    const size_t b_offset = funcs.get_b_packed_offset(0, k_origin, k, n);
    std::vector<float> scale_a(m, 1.0f), scale_b(n, 1.0f);
    std::vector<int32_t> zero_point_a(m, 0);
    const tqt_qai8dxp_qsi8cxp_params qparams = {scale_a.data(), zero_point_a.data(),
                                                scale_b.data()};
    std::vector<T_D> compact(m * n), selected(m * n);
    const T_D clamp_min = to_target_type<T_D>(-FLT_MAX);
    const T_D clamp_max = to_target_type<T_D>(FLT_MAX);
    funcs.run_gemm(m, n, window, A_compact.data(), window, 0, B_compact.data(), window, 0, nullptr,
                   n, compact.data(), n, nullptr, clamp_min, clamp_max, &qparams);
    funcs.run_gemm(m, n, window, A_full.data() + a_offset, k, k_origin, B_full.data() + b_offset, k,
                   k_origin, nullptr, n, selected.data(), n, nullptr, clamp_min, clamp_max,
                   &qparams);

    SCOPED_TRACE("floating-output packed int8 nonzero K origin");
    EXPECT_EQ(compact, selected);
}

// ============================================================================
// W4A8 quantization helpers (shared by per-channel and block-quantized tests)
// ----------------------------------------------------------------------------
// quantize_a_qai8dx  : per-row asymmetric int8 (no block dimension)
// quantize_a_qsi8d2  : per-row block-quantized symmetric int8 (block size bl)
// quantize_b_qsi4c2  : per-block symmetric int4   (signed [-8, 7], stored ZP=8)
// quantize_b_qai4c2  : per-block asymmetric int4  (unsigned [0, 15] + fp zp)
//
// Setting bl == k reduces qsi8d2 to per-row symmetric (degenerate single block)
// and reduces qsi4c2 to per-channel symmetric (legacy W4A8 per-channel path).
// ============================================================================

/// Per-row asymmetric int8 quantization for matrix A [m, k] (qai8dx).
/// scale_a is [m] fp32, zp_a is [m] int32.
inline void quantize_a_qai8dx(const float *A_fp32, size_t m, size_t k, int8_t *A_i8, float *scale_a,
                              int32_t *zp_a)
{
    for (size_t row = 0; row < m; ++row) {
        const float *row_ptr = A_fp32 + row * k;
        float fmin = FLT_MAX;
        float fmax = -FLT_MAX;
        for (size_t i = 0; i < k; ++i) {
            if (row_ptr[i] < fmin)
                fmin = row_ptr[i];
            if (row_ptr[i] > fmax)
                fmax = row_ptr[i];
        }
        fmax = fmax > 0.0f ? fmax : 0.0f;
        fmin = fmin < 0.0f ? fmin : 0.0f;

        float s = (fmax - fmin) / 255.0f;
        if (s == 0.0f) {
            scale_a[row] = 1.0f;
            zp_a[row] = 0;
            for (size_t i = 0; i < k; ++i) {
                A_i8[row * k + i] = 0;
            }
            continue;
        }
        float zp_tmp = -128.0f - fmin / s;
        int32_t zp = static_cast<int32_t>(std::nearbyintf(zp_tmp));
        if (zp < -128)
            zp = -128;
        if (zp > 127)
            zp = 127;
        scale_a[row] = s;
        zp_a[row] = zp;
        for (size_t i = 0; i < k; ++i) {
            A_i8[row * k + i] = quantize_i8(row_ptr[i], s, zp);
        }
    }
}

/// Per-row block-quantized symmetric int8 for matrix A [m, k] (qsi8d2).
/// scale_a is [m, k/bl] fp32 row-major (ldsa = k/bl). No zero point.
inline void quantize_a_qsi8d2(const float *A_fp32, size_t m, size_t k, size_t bl, int8_t *A_i8,
                              float *scale_a, size_t ldsa)
{
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < m; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = A_fp32 + row * k + blk * bl;
            float abs_max = 0.0f;
            for (size_t i = 0; i < bl; ++i) {
                float v = std::fabsf(blk_ptr[i]);
                if (v > abs_max)
                    abs_max = v;
            }
            // Symmetric int8 in [-127, 127]; scale = abs_max / 127.
            float s = abs_max / 127.0f;
            if (s == 0.0f)
                s = 1.0f;
            scale_a[row * ldsa + blk] = s;
            for (size_t i = 0; i < bl; ++i) {
                float q = std::nearbyintf(blk_ptr[i] / s);
                if (q > 127.0f)
                    q = 127.0f;
                if (q < -127.0f)
                    q = -127.0f;
                A_i8[row * k + blk * bl + i] = static_cast<int8_t>(q);
            }
        }
    }
}

/// Per-block symmetric int4 for transposed B [n, k] (qsi4c2t).
/// scale_b is [n, k/bl] fp32 row-major (ldsb = k/bl).
/// Nibbles stored ZP=8 unsigned [0, 15] in [n, k/2] uint8 buffer.
/// Caller must zero-initialize B_packed (the routine writes nibbles in-place).
inline void quantize_b_qsi4c2(const float *B_fp32, size_t n, size_t k, size_t bl, uint8_t *B_packed,
                              float *scale_b, size_t ldsb)
{
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < n; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = B_fp32 + row * k + blk * bl;
            float abs_max = 0.0f;
            for (size_t i = 0; i < bl; ++i) {
                float v = std::fabsf(blk_ptr[i]);
                if (v > abs_max)
                    abs_max = v;
            }
            // Symmetric int4 in [-8, 7]; scale = abs_max / 8 (W4A16 convention).
            float s = abs_max / 8.0f;
            if (s == 0.0f)
                s = 1.0f;
            scale_b[row * ldsb + blk] = s;
            for (size_t i = 0; i < bl; ++i) {
                float q = std::nearbyintf(blk_ptr[i] / s);
                if (q > 7.0f)
                    q = 7.0f;
                if (q < -8.0f)
                    q = -8.0f;
                int32_t qi = static_cast<int32_t>(q);
                const size_t kidx = blk * bl + i;
                const size_t byte_idx = row * (k / 2) + kidx / 2;
                uint8_t nib = static_cast<uint8_t>(qi + 8);
                if ((kidx & 1) == 0) {
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0xf0) | (nib & 0x0f);
                } else {
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0x0f) | ((nib & 0x0f) << 4);
                }
            }
        }
    }
}

/// Per-block ASYMMETRIC int4 for transposed B [n, k] (qai4c2t).
/// scale_b is [n, k/bl] fp32 (ldsb), offset_b is [n, k/bl] fp32 (ldob).
/// Nibbles stored ZP=0 unsigned [0, 15]; dequant: q * scale_b[n, blk] + offset_b[n, blk].
/// Caller must zero-initialize B_packed (the routine writes nibbles in-place).
inline void quantize_b_qai4c2(const float *B_fp32, size_t n, size_t k, size_t bl, uint8_t *B_packed,
                              float *scale_b, float *offset_b, size_t ldsb, size_t ldob)
{
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < n; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = B_fp32 + row * k + blk * bl;
            float fmin = FLT_MAX;
            float fmax = -FLT_MAX;
            for (size_t i = 0; i < bl; ++i) {
                if (blk_ptr[i] < fmin)
                    fmin = blk_ptr[i];
                if (blk_ptr[i] > fmax)
                    fmax = blk_ptr[i];
            }
            // qmax = 15, qmin = 0. Asym: s = (fmax - fmin) / (qmax - qmin); zp = -fmin / s.
            float s = (fmax - fmin) / 15.0f;
            if (s == 0.0f) {
                scale_b[row * ldsb + blk] = 0.0f;
                offset_b[row * ldob + blk] = fmin;
                for (size_t i = 0; i < bl; ++i) {
                    const size_t kidx = blk * bl + i;
                    const size_t byte_idx = row * (k / 2) + kidx / 2;
                    if ((kidx & 1) == 0) {
                        B_packed[byte_idx] = (B_packed[byte_idx] & 0xf0);
                    } else {
                        B_packed[byte_idx] = (B_packed[byte_idx] & 0x0f);
                    }
                }
                continue;
            }
            float zp_f = -fmin / s;
            // Clamp zp to [0, 15] in fp domain; the kernel uses it verbatim as fp.
            if (zp_f < 0.0f)
                zp_f = 0.0f;
            if (zp_f > 15.0f)
                zp_f = 15.0f;
            scale_b[row * ldsb + blk] = s;
            offset_b[row * ldob + blk] = -zp_f * s;
            for (size_t i = 0; i < bl; ++i) {
                float q = std::nearbyintf(blk_ptr[i] / s + zp_f);
                if (q > 15.0f)
                    q = 15.0f;
                if (q < 0.0f)
                    q = 0.0f;
                uint8_t qb = static_cast<uint8_t>(q);
                const size_t kidx = blk * bl + i;
                const size_t byte_idx = row * (k / 2) + kidx / 2;
                if ((kidx & 1) == 0) {
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0xf0) | (qb & 0x0f);
                } else {
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0x0f) | ((qb & 0x0f) << 4);
                }
            }
        }
    }
}

// ============================================================================
// W4A8 reference fp32 implementations (block-quantized; bl=k recovers per-channel)
// ----------------------------------------------------------------------------
// ref_qai8dx_qsi4c2 : Class 1 — A per-row asym, B per-block sym
// ref_qsi8d2_qsi4c2 : Class 2 — A per-block sym,  B per-block sym
// ref_qsi8d2_qai4c2 : Class 3 — A per-block sym,  B per-block asym
// ============================================================================

/// Class 1: qai8dx × qsi4c2t (scale_a per-row, scale_b per-block).
/// acc = scale_a[m] * sum_blk(scale_b[n, blk]
///         * sum_{k in blk}((qa[m, k] - zp_a[m]) * qb_signed[n, k]))
inline void ref_qai8dx_qsi4c2(size_t m, size_t n, size_t k, size_t bl, const int8_t *A_i8,
                              const uint8_t *B_packed, const float *scale_a, const int32_t *zp_a,
                              const float *scale_b, size_t ldsb, const float *C, const float *bias,
                              float *D, float clamp_min, float clamp_max)
{
    const size_t num_blocks = k / bl;
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (size_t blk = 0; blk < num_blocks; ++blk) {
                int32_t blk_acc = 0;
                for (size_t kk = 0; kk < bl; ++kk) {
                    size_t kidx = blk * bl + kk;
                    uint8_t byte = B_packed[ni * (k / 2) + kidx / 2];
                    int32_t qb =
                        (kidx & 1) ? ((int32_t)(byte >> 4) - 8) : ((int32_t)(byte & 0xf) - 8);
                    int32_t qa = (int32_t)A_i8[mi * k + kidx];
                    blk_acc += (qa - zp_a[mi]) * qb;
                }
                acc += scale_b[ni * ldsb + blk] * (float)blk_acc;
            }
            acc *= scale_a[mi];
            float result = acc;
            if (C)
                result += C[mi * n + ni];
            if (bias)
                result += bias[ni];
            if (result < clamp_min)
                result = clamp_min;
            if (result > clamp_max)
                result = clamp_max;
            D[mi * n + ni] = result;
        }
    }
}

/// Class 2: qsi8d2 × qsi4c2t (per-block on both sides).
/// acc = sum_blk(scale_a[m, blk]
///         * sum_{k in blk}(qa[m, k] * qb_signed[n, k]))
inline void ref_qsi8d2_qsi4c2(size_t m, size_t n, size_t k, size_t bl, const int8_t *A_i8,
                              const uint8_t *B_packed, const float *scale_a, size_t ldsa,
                              const float *scale_b, size_t ldsb, const float *C, const float *bias,
                              float *D, float clamp_min, float clamp_max)
{
    const size_t num_blocks = k / bl;
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (size_t blk = 0; blk < num_blocks; ++blk) {
                int32_t blk_acc = 0;
                for (size_t kk = 0; kk < bl; ++kk) {
                    size_t kidx = blk * bl + kk;
                    uint8_t byte = B_packed[ni * (k / 2) + kidx / 2];
                    int32_t qb =
                        (kidx & 1) ? ((int32_t)(byte >> 4) - 8) : ((int32_t)(byte & 0xf) - 8);
                    int32_t qa = (int32_t)A_i8[mi * k + kidx];
                    blk_acc += qa * qb;
                }
                acc += scale_a[mi * ldsa + blk] * scale_b[ni * ldsb + blk] * (float)blk_acc;
            }
            float result = acc;
            if (C)
                result += C[mi * n + ni];
            if (bias)
                result += bias[ni];
            if (result < clamp_min)
                result = clamp_min;
            if (result > clamp_max)
                result = clamp_max;
            D[mi * n + ni] = result;
        }
    }
}

/// Class 3: qsi8d2 × qai4c2t (per-block on both sides + ASYM B with independent FP32 offset).
/// acc = sum_blk(scale_a[m, blk]
///         * sum_{k in blk}(qa[m, k] * (qb_unsigned[n, k] * scale_b[n, blk] + offset_b[n, blk])))
inline void ref_qsi8d2_qai4c2(size_t m, size_t n, size_t k, size_t bl, const int8_t *A_i8,
                              const uint8_t *B_packed, const float *scale_a, size_t ldsa,
                              const float *scale_b, size_t ldsb, const float *offset_b, size_t ldob,
                              const float *C, const float *bias, float *D, float clamp_min,
                              float clamp_max)
{
    const size_t num_blocks = k / bl;
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (size_t blk = 0; blk < num_blocks; ++blk) {
                float ob = offset_b[ni * ldob + blk];
                float blk_acc = 0.0f;
                for (size_t kk = 0; kk < bl; ++kk) {
                    size_t kidx = blk * bl + kk;
                    uint8_t byte = B_packed[ni * (k / 2) + kidx / 2];
                    float qb = (kidx & 1) ? (float)(byte >> 4) : (float)(byte & 0xf);
                    float qa = (float)A_i8[mi * k + kidx];
                    blk_acc += qa * (qb * scale_b[ni * ldsb + blk] + ob);
                }
                acc += scale_a[mi * ldsa + blk] * blk_acc;
            }
            float result = acc;
            if (C)
                result += C[mi * n + ni];
            if (bias)
                result += bias[ni];
            if (result < clamp_min)
                result = clamp_min;
            if (result > clamp_max)
                result = clamp_max;
            D[mi * n + ni] = result;
        }
    }
}

// ============================================================================
// W4A8 per-channel (qai8dx × qsi4cxt): Funcs templates + runners
// ----------------------------------------------------------------------------
// Reuses the BQ helpers with bl == k and ldsb == 1 (single block per row).
// Output type TD ∈ {float, float16_t, bfloat16_t}; the qparams struct is
// tqt_qai8dx_qsi4cx_params (scale_a[m], zp_a[m], scale_b[n]).
// ============================================================================

template <typename TD>
struct W4A8GemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8PackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetBPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

inline std::vector<int8_t> copy_i8_k_window(const std::vector<int8_t> &src, size_t rows,
                                            size_t stride, size_t k_origin, size_t window)
{
    std::vector<int8_t> dst(rows * window);
    for (size_t row = 0; row < rows; ++row) {
        std::copy_n(src.data() + row * stride + k_origin, window, dst.data() + row * window);
    }
    return dst;
}

inline std::vector<uint8_t> copy_i4_k_window(const std::vector<uint8_t> &src, size_t rows,
                                             size_t stride, size_t k_origin, size_t window)
{
    TQT_ASSUME(k_origin % 2 == 0);
    TQT_ASSUME(window % 2 == 0);
    std::vector<uint8_t> dst(rows * (window / 2));
    for (size_t row = 0; row < rows; ++row) {
        std::copy_n(src.data() + row * (stride / 2) + k_origin / 2, window / 2,
                    dst.data() + row * (window / 2));
    }
    return dst;
}

template <typename TD>
inline void check_w4a8_nonzero_origin(const GemmTestParams &params, const W4A8GemmFuncs<TD> &funcs,
                                      const std::vector<int8_t> &A_i8,
                                      const std::vector<uint8_t> &B_i4,
                                      const struct tqt_qai8dx_qsi4cx_params &qparams)
{
    const size_t k_origin = (params.k / 4) * 2;
    const size_t window = params.k - k_origin;
    if (k_origin == 0 || window == 0 || window % 2 != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<TD> expected(params.m * params.n, to_target_type<TD>(0.0f));
    std::vector<TD> actual(params.m * params.n, to_target_type<TD>(0.0f));
    const TD clamp_min = to_target_type<TD>(-FLT_MAX);
    const TD clamp_max = to_target_type<TD>(FLT_MAX);

    funcs.run_gemm(params.m, params.n, window, A_window.data(), window, 0, B_window.data(), window,
                   0, nullptr, params.n, expected.data(), params.n, nullptr, clamp_min, clamp_max,
                   &qparams);
    funcs.run_gemm(params.m, params.n, window, A_i8.data() + k_origin, params.k, k_origin,
                   B_i4.data() + k_origin / 2, params.k, k_origin, nullptr, params.n, actual.data(),
                   params.n, nullptr, clamp_min, clamp_max, &qparams);

    SCOPED_TRACE("nonzero K origin");
    auto result =
        verify_gemm_result<TD>(expected.data(), actual.data(), actual.size(),
                               DefaultThreshold<TD>::abs_error, DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message;
    if (gemm_clamp_probe_active) {
        SCOPED_TRACE("special-value split-K propagation");
        std::vector<TD> partial(params.m * params.n), combined(params.m * params.n);
        funcs.run_gemm(params.m, params.n, k_origin, A_i8.data(), params.k, 0, B_i4.data(),
                       params.k, 0, nullptr, params.n, partial.data(), params.n, nullptr,
                       to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY), &qparams);
        funcs.run_gemm(params.m, params.n, window, A_i8.data() + k_origin, params.k, k_origin,
                       B_i4.data() + k_origin / 2, params.k, k_origin, partial.data(), params.n,
                       combined.data(), params.n, nullptr, to_target_type<TD>(-INFINITY),
                       to_target_type<TD>(INFINITY), &qparams);
    }
}

template <typename TD>
inline void check_w4a8_nonzero_origin(const GemmTestParams &params,
                                      const W4A8PackedGemmFuncs<TD> &funcs,
                                      const std::vector<int8_t> &A_i8,
                                      const std::vector<uint8_t> &B_i4,
                                      const std::vector<uint8_t> &A_packed,
                                      const std::vector<uint8_t> &B_packed,
                                      const struct tqt_qai8dx_qsi4cx_params &qparams)
{
    const size_t k_origin = (params.k / 4) * 2;
    const size_t window = params.k - k_origin;
    if (k_origin == 0 || window == 0 || window % 2 != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<uint8_t> A_window_packed(funcs.get_a_packed_size(params.m, window));
    std::vector<uint8_t> B_window_packed(funcs.get_b_packed_size(params.n, window));
    funcs.run_a_pack(params.m, window, window, window, 0, A_window.data(), A_window_packed.data());
    funcs.run_bt_pack(params.n, window, window, window, 0, B_window.data(), B_window_packed.data());

    const size_t a_offset = funcs.get_a_packed_offset(0, k_origin, params.k, params.m);
    const size_t b_offset = funcs.get_b_packed_offset(0, k_origin, params.k, params.n);
    std::vector<TD> expected(params.m * params.n, to_target_type<TD>(0.0f));
    std::vector<TD> actual(params.m * params.n, to_target_type<TD>(0.0f));
    const TD clamp_min = to_target_type<TD>(-FLT_MAX);
    const TD clamp_max = to_target_type<TD>(FLT_MAX);

    funcs.run_gemm(params.m, params.n, window, A_window_packed.data(), window, 0,
                   B_window_packed.data(), window, 0, nullptr, params.n, expected.data(), params.n,
                   nullptr, clamp_min, clamp_max, &qparams);
    funcs.run_gemm(params.m, params.n, window, A_packed.data() + a_offset, params.k, k_origin,
                   B_packed.data() + b_offset, params.k, k_origin, nullptr, params.n, actual.data(),
                   params.n, nullptr, clamp_min, clamp_max, &qparams);

    SCOPED_TRACE("nonzero packed K origin");
    auto result =
        verify_gemm_result<TD>(expected.data(), actual.data(), actual.size(),
                               DefaultThreshold<TD>::abs_error, DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message;
    if (gemm_clamp_probe_active) {
        SCOPED_TRACE("special-value split-K propagation");
        std::vector<TD> partial(params.m * params.n), combined(params.m * params.n);
        funcs.run_gemm(params.m, params.n, k_origin, A_packed.data(), params.k, 0, B_packed.data(),
                       params.k, 0, nullptr, params.n, partial.data(), params.n, nullptr,
                       to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY), &qparams);
        funcs.run_gemm(params.m, params.n, window, A_packed.data() + a_offset, params.k, k_origin,
                       B_packed.data() + b_offset, params.k, k_origin, partial.data(), params.n,
                       combined.data(), params.n, nullptr, to_target_type<TD>(-INFINITY),
                       to_target_type<TD>(INFINITY), &qparams);
    }
}

template <typename TD, typename BiasT = float>
inline void run_w4a8_gemm_test(const GemmTestParams &params, const W4A8GemmFuncs<TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_qai8dx(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    // Per-channel int4 is the bl == k, ldsb == 1 special case of qsi4c2.
    std::vector<uint8_t> B_packed(n * (k / 2), 0);
    std::vector<float> scale_b(n);
    quantize_b_qsi4c2(B_fp32.data(), n, k, k, B_packed.data(), scale_b.data(), 1);

    std::vector<TD> C_data;
    if (params.has_c) {
        C_data.resize(m * n);
        UniformRandomGenerator<TD> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_data.data(), m, n);
    }
    std::vector<BiasT> bias_data;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<BiasT> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
    }

    const float clamp_min_f = params.clamp_min_f;
    const float clamp_max_f = params.clamp_max_f;
    const TD clamp_min_td = to_target_type<TD>(clamp_min_f);
    const TD clamp_max_td = to_target_type<TD>(clamp_max_f);

    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        for (size_t i = 0; i < m * n; ++i) {
            C_fp32[i] = to_float(C_data[i]);
        }
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        for (size_t i = 0; i < n; ++i) {
            bias_fp32[i] = to_float(bias_data[i]);
        }
    }

    std::vector<float> D_ref_fp32(m * n);
    ref_qai8dx_qsi4c2(m, n, k, k, A_i8.data(), B_packed.data(), scale_a.data(), zp_a.data(),
                      scale_b.data(), 1, params.has_c ? C_fp32.data() : nullptr,
                      params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(), clamp_min_f,
                      clamp_max_f);
    std::vector<TD> D_ref(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_ref[i] = to_target_type<TD>(D_ref_fp32[i]);
    }

    struct tqt_qai8dx_qsi4cx_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();

    std::vector<TD> D_act(m * n, to_target_type<TD>(0.0f));
    funcs.run_gemm(m, n, k, A_i8.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min_td, clamp_max_td,
                   &qparams);

    auto result =
        verify_gemm_result<TD>(D_ref.data(), D_act.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;

    check_w4a8_nonzero_origin(params, funcs, A_i8, B_packed, qparams);
}

template <typename TD, typename BiasT = float>
inline void run_w4a8_packed_gemm_test(const GemmTestParams &params,
                                      const W4A8PackedGemmFuncs<TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_qai8dx(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_int4(n * (k / 2), 0);
    std::vector<float> scale_b(n);
    quantize_b_qsi4c2(B_fp32.data(), n, k, k, B_int4.data(), scale_b.data(), 1);

    std::vector<TD> C_data;
    if (params.has_c) {
        C_data.resize(m * n);
        UniformRandomGenerator<TD> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_data.data(), m, n);
    }
    std::vector<BiasT> bias_data;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<BiasT> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
    }

    const float clamp_min_f = params.clamp_min_f;
    const float clamp_max_f = params.clamp_max_f;
    const TD clamp_min_td = to_target_type<TD>(clamp_min_f);
    const TD clamp_max_td = to_target_type<TD>(clamp_max_f);

    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        for (size_t i = 0; i < m * n; ++i) {
            C_fp32[i] = to_float(C_data[i]);
        }
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        for (size_t i = 0; i < n; ++i) {
            bias_fp32[i] = to_float(bias_data[i]);
        }
    }

    std::vector<float> D_ref_fp32(m * n);
    ref_qai8dx_qsi4c2(m, n, k, k, A_i8.data(), B_int4.data(), scale_a.data(), zp_a.data(),
                      scale_b.data(), 1, params.has_c ? C_fp32.data() : nullptr,
                      params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(), clamp_min_f,
                      clamp_max_f);
    std::vector<TD> D_ref(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_ref[i] = to_target_type<TD>(D_ref_fp32[i]);
    }

    struct tqt_qai8dx_qsi4cx_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();

    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(n, k, ldb, k, 0, B_int4.data(), B_packed.data());

    std::vector<TD> D_act(m * n, to_target_type<TD>(0.0f));
    funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min_td, clamp_max_td,
                   &qparams);

    auto result =
        verify_gemm_result<TD>(D_ref.data(), D_act.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;

    check_w4a8_nonzero_origin(params, funcs, A_i8, B_int4, A_packed, B_packed, qparams);
}

// ============================================================================
// W4A8 per-channel requantized int8 output (qai8dx × qsi4cxt -> qai8):
// Funcs structs + runners
// ----------------------------------------------------------------------------
// A per-row asym int8, B per-channel sym int4, optional f32 C / 1xN f32 bias,
// requantized int8 output via tqt_qai8_qai8dx_qsi4cx_params (scale_a[m],
// zp_a[m], scale_b[n], scale_d, zero_point_d). The real-domain reference reuses
// ref_qai8dx_qsi4c2 (bl == k); the int8 kernel output is dequantized back to
// fp32 for a cosine + abs-error comparison (same style as the i8_i8 reqi32toi8
// runner), which tolerates the inherent ±1 requantization rounding.
// ============================================================================

struct W4A8ReqI8GemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max,
                                 const struct tqt_qai8_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

struct W4A8ReqI8PackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetBPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max,
                                 const struct tqt_qai8_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

inline void check_w4a8_reqi8_nonzero_origin(const GemmTestParams &params,
                                            const W4A8ReqI8GemmFuncs &funcs,
                                            const std::vector<int8_t> &A_i8,
                                            const std::vector<uint8_t> &B_i4,
                                            const struct tqt_qai8_qai8dx_qsi4cx_params &qparams)
{
    const size_t k_origin = (params.k / 4) * 2;
    const size_t window = params.k - k_origin;
    if (k_origin == 0 || window == 0 || window % 2 != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<int8_t> expected(params.m * params.n);
    std::vector<int8_t> actual(params.m * params.n);
    funcs.run_gemm(params.m, params.n, window, A_window.data(), window, 0, B_window.data(), window,
                   0, nullptr, params.n, expected.data(), params.n, nullptr, INT8_MIN, INT8_MAX,
                   &qparams);
    funcs.run_gemm(params.m, params.n, window, A_i8.data() + k_origin, params.k, k_origin,
                   B_i4.data() + k_origin / 2, params.k, k_origin, nullptr, params.n, actual.data(),
                   params.n, nullptr, INT8_MIN, INT8_MAX, &qparams);

    SCOPED_TRACE("nonzero K origin");
    EXPECT_EQ(actual, expected);
}

inline void check_w4a8_reqi8_nonzero_origin(const GemmTestParams &params,
                                            const W4A8ReqI8PackedGemmFuncs &funcs,
                                            const std::vector<int8_t> &A_i8,
                                            const std::vector<uint8_t> &B_i4,
                                            const std::vector<uint8_t> &A_packed,
                                            const std::vector<uint8_t> &B_packed,
                                            const struct tqt_qai8_qai8dx_qsi4cx_params &qparams)
{
    const size_t k_origin = (params.k / 4) * 2;
    const size_t window = params.k - k_origin;
    if (k_origin == 0 || window == 0 || window % 2 != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<uint8_t> A_window_packed(funcs.get_a_packed_size(params.m, window));
    std::vector<uint8_t> B_window_packed(funcs.get_b_packed_size(params.n, window));
    funcs.run_a_pack(params.m, window, window, window, 0, A_window.data(), A_window_packed.data());
    funcs.run_bt_pack(params.n, window, window, window, 0, B_window.data(), B_window_packed.data());

    const size_t a_offset = funcs.get_a_packed_offset(0, k_origin, params.k, params.m);
    const size_t b_offset = funcs.get_b_packed_offset(0, k_origin, params.k, params.n);
    std::vector<int8_t> expected(params.m * params.n);
    std::vector<int8_t> actual(params.m * params.n);
    funcs.run_gemm(params.m, params.n, window, A_window_packed.data(), window, 0,
                   B_window_packed.data(), window, 0, nullptr, params.n, expected.data(), params.n,
                   nullptr, INT8_MIN, INT8_MAX, &qparams);
    funcs.run_gemm(params.m, params.n, window, A_packed.data() + a_offset, params.k, k_origin,
                   B_packed.data() + b_offset, params.k, k_origin, nullptr, params.n, actual.data(),
                   params.n, nullptr, INT8_MIN, INT8_MAX, &qparams);

    SCOPED_TRACE("nonzero packed K origin");
    EXPECT_EQ(actual, expected);
}

inline void run_w4a8_reqi8_gemm_test(const GemmTestParams &params, const W4A8ReqI8GemmFuncs &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_qai8dx(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_packed(n * (k / 2), 0);
    std::vector<float> scale_b(n);
    quantize_b_qsi4c2(B_fp32.data(), n, k, k, B_packed.data(), scale_b.data(), 1);

    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        UniformRandomGenerator<float> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_fp32.data(), m, n);
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        UniformRandomGenerator<float> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_fp32.data(), 1, n);
    }

    // Real-domain reference (no clamp in real domain; int8 clamp applied later).
    std::vector<float> D_ref_fp32(m * n);
    ref_qai8dx_qsi4c2(m, n, k, k, A_i8.data(), B_packed.data(), scale_a.data(), zp_a.data(),
                      scale_b.data(), 1, params.has_c ? C_fp32.data() : nullptr,
                      params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(), -FLT_MAX,
                      FLT_MAX);

    // Derive output quantization params (asymmetric int8) from the real reference.
    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    struct tqt_qai8_qai8dx_qsi4cx_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);
    funcs.run_gemm(m, n, k, A_i8.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_fp32.data() : nullptr, ldc, D_i8.data(), ldd,
                   params.has_bias ? bias_fp32.data() : nullptr, clamp_min, clamp_max, &qparams);

    // Dequantize the int8 kernel output and compare with the fp32 reference.
    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;

    check_w4a8_reqi8_nonzero_origin(params, funcs, A_i8, B_packed, qparams);
}

inline void run_w4a8_reqi8_packed_gemm_test(const GemmTestParams &params,
                                            const W4A8ReqI8PackedGemmFuncs &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_qai8dx(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_int4(n * (k / 2), 0);
    std::vector<float> scale_b(n);
    quantize_b_qsi4c2(B_fp32.data(), n, k, k, B_int4.data(), scale_b.data(), 1);

    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        UniformRandomGenerator<float> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_fp32.data(), m, n);
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        UniformRandomGenerator<float> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_fp32.data(), 1, n);
    }

    std::vector<float> D_ref_fp32(m * n);
    ref_qai8dx_qsi4c2(m, n, k, k, A_i8.data(), B_int4.data(), scale_a.data(), zp_a.data(),
                      scale_b.data(), 1, params.has_c ? C_fp32.data() : nullptr,
                      params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(), -FLT_MAX,
                      FLT_MAX);

    float scale_d;
    int32_t zp_d;
    compute_asym_quant_params_i8(D_ref_fp32.data(), m * n, scale_d, zp_d);

    struct tqt_qai8_qai8dx_qsi4cx_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(n, k, ldb, k, 0, B_int4.data(), B_packed.data());

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;
    std::vector<int8_t> D_i8(m * n, 0);
    funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_fp32.data() : nullptr, ldc, D_i8.data(), ldd,
                   params.has_bias ? bias_fp32.data() : nullptr, clamp_min, clamp_max, &qparams);

    std::vector<float> D_act_fp32(m * n);
    dequantize_matrix_i8(D_i8.data(), D_act_fp32.data(), m * n, scale_d, zp_d);

    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result = verify_gemm_result<float>(D_ref_fp32.data(), D_act_fp32.data(), m * n, tolerance,
                                            cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;

    check_w4a8_reqi8_nonzero_origin(params, funcs, A_i8, B_int4, A_packed, B_packed, qparams);
}

// ============================================================================
// W4A8 block-quantized: Funcs templates
// ----------------------------------------------------------------------------
// 3 input categories × {unpacked, packed} = 6 Funcs structs.
// Class 1 (qai8dx × qsi4c2): tqt_qai8dx_qsi4c2_params
// Class 2 (qsi8d2 × qsi4c2): tqt_qsi8d2_qsi4c2_params
// Class 3 (qsi8d2 × qai4c2): tqt_qsi8d2_qai4c2_params
// ============================================================================

// ---------- Class 1: qai8dx × qsi4c2t (unpacked) ----------
template <typename TD>
struct W4A8BQ_Qai8dx_Qsi4c2_Funcs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

// ---------- Class 1: qai8dxp × qsi4c2p (packed) ----------
template <typename TD>
struct W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetBPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ---------- Class 2: qsi8d2 × qsi4c2t (unpacked) ----------
template <typename TD>
struct W4A8BQ_Qsi8d2_Qsi4c2_Funcs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

// ---------- Class 2: qsi8d2p × qsi4c2p (packed) ----------
template <typename TD>
struct W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetBPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ---------- Class 3: qsi8d2 × qai4c2t (unpacked) ----------
template <typename TD>
struct W4A8BQ_Qsi8d2_Qai4c2_Funcs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qai4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

// ---------- Class 3: qsi8d2p × qai4c2p (packed) ----------
template <typename TD>
struct W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetBPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qai4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// W4A8 block-quantized: shared setup helpers
// ----------------------------------------------------------------------------
// W4A8BQInputs<TD>          : random fp32 inputs + optional TD C / bias
// make_w4a8_bq_inputs<TD>   : populate the above from GemmTestParams
// verify_w4a8_bq_result<TD> : verify_gemm_result wrapper with rich EXPECT message
// round_through_td<TD>      : convert std::vector<float> to std::vector<TD>
// ============================================================================

template <typename TD, typename BiasT = float>
struct W4A8BQInputs
{
    size_t m;
    size_t n;
    size_t k;
    size_t bl;
    size_t ldsa;  // = k/bl
    size_t ldsb;  // = k/bl
    std::vector<float> A_fp32;
    std::vector<float> B_fp32;
    std::vector<TD> C_data;
    std::vector<BiasT> bias_data;
    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    bool has_c;
    bool has_bias;
    float clamp_min_f;
    float clamp_max_f;
    TD clamp_min_td;
    TD clamp_max_td;
};

template <typename TD, typename BiasT = float>
inline W4A8BQInputs<TD, BiasT> make_w4a8_bq_inputs(const GemmTestParams &params)
{
    W4A8BQInputs<TD, BiasT> in;
    in.m = params.m;
    in.n = params.n;
    in.k = params.k;
    in.bl = params.bl;
    in.ldsa = params.k / params.bl;
    in.ldsb = params.k / params.bl;
    in.has_c = params.has_c;
    in.has_bias = params.has_bias;
    in.clamp_min_f = params.clamp_min_f;
    in.clamp_max_f = params.clamp_max_f;
    in.clamp_min_td = to_target_type<TD>(params.clamp_min_f);
    in.clamp_max_td = to_target_type<TD>(params.clamp_max_f);

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    in.A_fp32.resize(in.m * in.k);
    in.B_fp32.resize(in.n * in.k);
    gen.fill_matrix(in.A_fp32.data(), in.m, in.k);
    gen.fill_matrix(in.B_fp32.data(), in.n, in.k);

    if (in.has_c) {
        in.C_data.resize(in.m * in.n);
        UniformRandomGenerator<TD> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(in.C_data.data(), in.m, in.n);
        in.C_fp32.resize(in.m * in.n);
        for (size_t i = 0; i < in.m * in.n; ++i) {
            in.C_fp32[i] = to_float(in.C_data[i]);
        }
    }
    if (in.has_bias) {
        in.bias_data.resize(in.n);
        UniformRandomGenerator<BiasT> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(in.bias_data.data(), 1, in.n);
        in.bias_fp32.resize(in.n);
        for (size_t i = 0; i < in.n; ++i) {
            in.bias_fp32[i] = to_float(in.bias_data[i]);
        }
    }
    return in;
}

template <typename TD>
inline void verify_w4a8_bq_result(const std::vector<TD> &D_ref, const std::vector<TD> &D_act,
                                  const GemmTestParams &params)
{
    auto result =
        verify_gemm_result<TD>(D_ref.data(), D_act.data(), D_ref.size(),
                               DefaultThreshold<TD>::abs_error, DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << params.m
                               << " N=" << params.n << " K=" << params.k << " BL=" << params.bl
                               << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

template <typename TD>
inline std::vector<TD> round_through_td(const std::vector<float> &src)
{
    std::vector<TD> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = to_target_type<TD>(src[i]);
    }
    return dst;
}

template <typename TD, typename Funcs, typename QParams>
inline void check_w4a8_bq_nonzero_origin(const GemmTestParams &params, const Funcs &funcs,
                                         const std::vector<int8_t> &A_i8,
                                         const std::vector<uint8_t> &B_i4,
                                         const QParams &window_params, const QParams &full_params)
{
    const size_t k_origin = params.bl;
    if (params.bl == 0 || params.k <= params.bl) {
        return;
    }
    const size_t window = params.k - k_origin;
    if (window % params.bl != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<TD> expected(params.m * params.n, to_target_type<TD>(0.0f));
    std::vector<TD> actual(params.m * params.n, to_target_type<TD>(0.0f));
    const TD clamp_min = to_target_type<TD>(-FLT_MAX);
    const TD clamp_max = to_target_type<TD>(FLT_MAX);

    funcs.run_gemm(params.m, params.n, window, params.bl, A_window.data(), window, 0,
                   B_window.data(), window, 0, nullptr, params.n, expected.data(), params.n,
                   nullptr, clamp_min, clamp_max, &window_params);
    funcs.run_gemm(params.m, params.n, window, params.bl, A_i8.data() + k_origin, params.k,
                   k_origin, B_i4.data() + k_origin / 2, params.k, k_origin, nullptr, params.n,
                   actual.data(), params.n, nullptr, clamp_min, clamp_max, &full_params);

    SCOPED_TRACE("nonzero block-quantized K origin");
    auto result =
        verify_gemm_result<TD>(expected.data(), actual.data(), actual.size(),
                               DefaultThreshold<TD>::abs_error, DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message;
    if (gemm_clamp_probe_active) {
        SCOPED_TRACE("special-value block-quantized split-K propagation");
        std::vector<TD> partial(params.m * params.n), combined(params.m * params.n);
        funcs.run_gemm(params.m, params.n, k_origin, params.bl, A_i8.data(), params.k, 0,
                       B_i4.data(), params.k, 0, nullptr, params.n, partial.data(), params.n,
                       nullptr, to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY),
                       &full_params);
        funcs.run_gemm(params.m, params.n, window, params.bl, A_i8.data() + k_origin, params.k,
                       k_origin, B_i4.data() + k_origin / 2, params.k, k_origin, partial.data(),
                       params.n, combined.data(), params.n, nullptr, to_target_type<TD>(-INFINITY),
                       to_target_type<TD>(INFINITY), &full_params);
    }
}

template <typename TD, typename Funcs, typename QParams>
inline void check_w4a8_bq_nonzero_packed_origin(
    const GemmTestParams &params, const Funcs &funcs, const std::vector<int8_t> &A_i8,
    const std::vector<uint8_t> &B_i4, const std::vector<uint8_t> &A_packed,
    const std::vector<uint8_t> &B_packed, const QParams &window_params, const QParams &full_params)
{
    const size_t k_origin = params.bl;
    if (params.bl == 0 || params.k <= params.bl) {
        return;
    }
    const size_t window = params.k - k_origin;
    if (window % params.bl != 0) {
        return;
    }

    const auto A_window = copy_i8_k_window(A_i8, params.m, params.k, k_origin, window);
    const auto B_window = copy_i4_k_window(B_i4, params.n, params.k, k_origin, window);
    std::vector<uint8_t> A_window_packed(funcs.get_a_packed_size(params.m, window));
    std::vector<uint8_t> B_window_packed(funcs.get_b_packed_size(params.n, window));
    funcs.run_a_pack(params.m, window, window, window, 0, A_window.data(), A_window_packed.data());
    funcs.run_bt_pack(params.n, window, window, window, 0, B_window.data(), B_window_packed.data());

    const size_t a_offset = funcs.get_a_packed_offset(0, k_origin, params.k, params.m);
    const size_t b_offset = funcs.get_b_packed_offset(0, k_origin, params.k, params.n);
    std::vector<TD> expected(params.m * params.n, to_target_type<TD>(0.0f));
    std::vector<TD> actual(params.m * params.n, to_target_type<TD>(0.0f));
    const TD clamp_min = to_target_type<TD>(-FLT_MAX);
    const TD clamp_max = to_target_type<TD>(FLT_MAX);

    funcs.run_gemm(params.m, params.n, window, params.bl, A_window_packed.data(), window, 0,
                   B_window_packed.data(), window, 0, nullptr, params.n, expected.data(), params.n,
                   nullptr, clamp_min, clamp_max, &window_params);
    funcs.run_gemm(params.m, params.n, window, params.bl, A_packed.data() + a_offset, params.k,
                   k_origin, B_packed.data() + b_offset, params.k, k_origin, nullptr, params.n,
                   actual.data(), params.n, nullptr, clamp_min, clamp_max, &full_params);

    SCOPED_TRACE("nonzero packed block-quantized K origin");
    auto result =
        verify_gemm_result<TD>(expected.data(), actual.data(), actual.size(),
                               DefaultThreshold<TD>::abs_error, DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message;
    if (gemm_clamp_probe_active) {
        SCOPED_TRACE("special-value block-quantized split-K propagation");
        std::vector<TD> partial(params.m * params.n), combined(params.m * params.n);
        funcs.run_gemm(params.m, params.n, k_origin, params.bl, A_packed.data(), params.k, 0,
                       B_packed.data(), params.k, 0, nullptr, params.n, partial.data(), params.n,
                       nullptr, to_target_type<TD>(-INFINITY), to_target_type<TD>(INFINITY),
                       &full_params);
        funcs.run_gemm(params.m, params.n, window, params.bl, A_packed.data() + a_offset, params.k,
                       k_origin, B_packed.data() + b_offset, params.k, k_origin, partial.data(),
                       params.n, combined.data(), params.n, nullptr, to_target_type<TD>(-INFINITY),
                       to_target_type<TD>(INFINITY), &full_params);
    }
}

// ============================================================================
// W4A8 block-quantized: runners (6 — 3 categories × {unpacked, packed})
// ============================================================================

// ---------- Class 1 unpacked: qai8dx × qsi4c2t ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qai8dx_qsi4c2t_test(const GemmTestParams &params,
                                            const W4A8BQ_Qai8dx_Qsi4c2_Funcs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m);
    std::vector<int32_t> zp_a(in.m);
    quantize_a_qai8dx(in.A_fp32.data(), in.m, in.k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    quantize_b_qsi4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(), in.ldsb);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qai8dx_qsi4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(),
                      zp_a.data(), scale_b.data(), in.ldsb, in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qai8dx_qsi4c2_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_i8.data(), lda, 0, B_int4.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_b += 1;
    check_w4a8_bq_nonzero_origin<TD>(params, funcs, A_i8, B_int4, window_params, qparams);
}

// ---------- Class 1 packed: qai8dxp × qsi4c2p ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qai8dxp_qsi4c2p_test(const GemmTestParams &params,
                                             const W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m);
    std::vector<int32_t> zp_a(in.m);
    quantize_a_qai8dx(in.A_fp32.data(), in.m, in.k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    quantize_b_qsi4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(), in.ldsb);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qai8dx_qsi4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(),
                      zp_a.data(), scale_b.data(), in.ldsb, in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qai8dx_qsi4c2_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, B_int4.data(), B_packed.data());

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_b += 1;
    check_w4a8_bq_nonzero_packed_origin<TD>(params, funcs, A_i8, B_int4, A_packed, B_packed,
                                            window_params, qparams);
}

// ---------- Class 2 unpacked: qsi8d2 × qsi4c2t ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qsi8d2_qsi4c2t_test(const GemmTestParams &params,
                                            const W4A8BQ_Qsi8d2_Qsi4c2_Funcs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m * in.ldsa);
    quantize_a_qsi8d2(in.A_fp32.data(), in.m, in.k, in.bl, A_i8.data(), scale_a.data(), in.ldsa);

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    quantize_b_qsi4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(), in.ldsb);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qsi8d2_qsi4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(), in.ldsa,
                      scale_b.data(), in.ldsb, in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qsi8d2_qsi4c2_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.ldsa = in.ldsa;
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_i8.data(), lda, 0, B_int4.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_a += 1;
    window_params.scale_b += 1;
    check_w4a8_bq_nonzero_origin<TD>(params, funcs, A_i8, B_int4, window_params, qparams);
}

// ---------- Class 2 packed: qsi8d2p × qsi4c2p ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qsi8d2p_qsi4c2p_test(const GemmTestParams &params,
                                             const W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m * in.ldsa);
    quantize_a_qsi8d2(in.A_fp32.data(), in.m, in.k, in.bl, A_i8.data(), scale_a.data(), in.ldsa);

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    quantize_b_qsi4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(), in.ldsb);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qsi8d2_qsi4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(), in.ldsa,
                      scale_b.data(), in.ldsb, in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qsi8d2_qsi4c2_params qparams
    {
    };
    qparams.scale_a = scale_a.data();
    qparams.ldsa = in.ldsa;
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, B_int4.data(), B_packed.data());

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_a += 1;
    window_params.scale_b += 1;
    check_w4a8_bq_nonzero_packed_origin<TD>(params, funcs, A_i8, B_int4, A_packed, B_packed,
                                            window_params, qparams);
}

// ---------- Class 3 unpacked: qsi8d2 × qai4c2t ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qsi8d2_qai4c2t_test(const GemmTestParams &params,
                                            const W4A8BQ_Qsi8d2_Qai4c2_Funcs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;
    const size_t ldob = in.ldsb;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m * in.ldsa);
    quantize_a_qsi8d2(in.A_fp32.data(), in.m, in.k, in.bl, A_i8.data(), scale_a.data(), in.ldsa);

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    std::vector<float> offset_b(in.n * ldob);
    quantize_b_qai4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(),
                      offset_b.data(), in.ldsb, ldob);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qsi8d2_qai4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(), in.ldsa,
                      scale_b.data(), in.ldsb, offset_b.data(), ldob,
                      in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qsi8d2_qai4c2_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.ldsa = in.ldsa;
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;
    qparams.offset_b = offset_b.data();
    qparams.ldob = ldob;

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_i8.data(), lda, 0, B_int4.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_a += 1;
    window_params.scale_b += 1;
    window_params.offset_b += 1;
    check_w4a8_bq_nonzero_origin<TD>(params, funcs, A_i8, B_int4, window_params, qparams);
}

// ---------- Class 3 packed: qsi8d2p × qai4c2p ----------
template <typename TD, typename BiasT = float>
inline void run_w4a8_bq_qsi8d2p_qai4c2p_test(const GemmTestParams &params,
                                             const W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs<TD, BiasT>(params);
    const size_t lda = in.k;
    const size_t ldb = in.k;
    const size_t ldc = in.n;
    const size_t ldd = in.n;
    const size_t ldob = in.ldsb;

    std::vector<int8_t> A_i8(in.m * in.k);
    std::vector<float> scale_a(in.m * in.ldsa);
    quantize_a_qsi8d2(in.A_fp32.data(), in.m, in.k, in.bl, A_i8.data(), scale_a.data(), in.ldsa);

    std::vector<uint8_t> B_int4(in.n * (in.k / 2), 0);
    std::vector<float> scale_b(in.n * in.ldsb);
    std::vector<float> offset_b(in.n * ldob);
    quantize_b_qai4c2(in.B_fp32.data(), in.n, in.k, in.bl, B_int4.data(), scale_b.data(),
                      offset_b.data(), in.ldsb, ldob);

    std::vector<float> D_ref_fp32(in.m * in.n);
    ref_qsi8d2_qai4c2(in.m, in.n, in.k, in.bl, A_i8.data(), B_int4.data(), scale_a.data(), in.ldsa,
                      scale_b.data(), in.ldsb, offset_b.data(), ldob,
                      in.has_c ? in.C_fp32.data() : nullptr,
                      in.has_bias ? in.bias_fp32.data() : nullptr, D_ref_fp32.data(),
                      in.clamp_min_f, in.clamp_max_f);
    auto D_ref = round_through_td<TD>(D_ref_fp32);

    struct tqt_qsi8d2_qai4c2_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.ldsa = in.ldsa;
    qparams.scale_b = scale_b.data();
    qparams.ldsb = in.ldsb;
    qparams.offset_b = offset_b.data();
    qparams.ldob = ldob;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, B_int4.data(), B_packed.data());

    std::vector<TD> D_act(in.m * in.n, to_target_type<TD>(0.0f));
    funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   in.has_c ? in.C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   in.has_bias ? in.bias_data.data() : nullptr, in.clamp_min_td, in.clamp_max_td,
                   &qparams);

    verify_w4a8_bq_result(D_ref, D_act, params);
    auto window_params = qparams;
    window_params.scale_a += 1;
    window_params.scale_b += 1;
    window_params.offset_b += 1;
    check_w4a8_bq_nonzero_packed_origin<TD>(params, funcs, A_i8, B_int4, A_packed, B_packed,
                                            window_params, qparams);
}

// ============================================================================
// Packed QSI8 block-quantized GEMM test runner
// ============================================================================

enum class Qsi8BlockWeightEncoding
{
    Qau4,
    Qau5,
    Qau8,
    Qsi6,
    Qsi8,
};

enum class Qsi8BlockPackedLayout
{
    NInterleaved,
    ZttSquare,
};

struct Qsi8BlockQuantizedGemmTestData
{
    size_t m;
    size_t n;
    size_t k;
    size_t activation_block_size;
    size_t weight_block_size;
    bool asymmetric;
    std::vector<int8_t> a;
    std::vector<int8_t> b;
    std::vector<float> scale_a;
    std::vector<int16_t> bsums_a;
    std::vector<float> scale_b;
    std::vector<float> real_zero_b;
    std::vector<float> c;
    std::vector<float> bias;
    std::vector<float> reference;
};

inline Qsi8BlockQuantizedGemmTestData make_qsi8_block_quantized_gemm_test_data(
    size_t n, size_t k, size_t activation_block_size, size_t weight_block_size, bool asymmetric,
    int32_t max_weight, size_t m = 5)
{
    Qsi8BlockQuantizedGemmTestData data = {};
    data.m = m;
    data.n = n;
    data.k = k;
    data.activation_block_size = activation_block_size;
    data.weight_block_size = weight_block_size;
    data.asymmetric = asymmetric;
    data.a.resize(m * k);
    data.b.resize(n * k);
    data.scale_a.resize(m * (k / activation_block_size));
    data.bsums_a.resize(m * (k / 16));
    data.scale_b.resize(n * (k / weight_block_size));
    data.real_zero_b.resize(n * (k / weight_block_size));
    data.c.resize(m * n);
    data.bias.resize(n);
    data.reference.resize(m * n);

    for (size_t row = 0; row < m; ++row) {
        for (size_t depth = 0; depth < k; ++depth)
            data.a[row * k + depth] = (int8_t)(((row * 13 + depth * 7) % 31) - 15);
        for (size_t block = 0; block < k / activation_block_size; ++block) {
            data.scale_a[row * (k / activation_block_size) + block] =
                0.006f + 0.001f * (float)((row + block) % 3);
        }
        for (size_t block = 0; block < k / 16; ++block) {
            int32_t sum = 0;
            for (size_t depth = 0; depth < 16; ++depth)
                sum += data.a[row * k + block * 16 + depth];
            data.bsums_a[row * (k / 16) + block] = (int16_t)sum;
        }
    }

    for (size_t column = 0; column < n; ++column) {
        for (size_t depth = 0; depth < k; ++depth) {
            const int32_t value =
                asymmetric ? (int32_t)((column * 5 + depth * 3) % (size_t)(max_weight + 1))
                           : (int32_t)((column * 5 + depth * 3) % (size_t)(2 * max_weight + 1)) -
                                 max_weight;
            data.b[column * k + depth] = (int8_t)value;
        }
        for (size_t block = 0; block < k / weight_block_size; ++block) {
            data.scale_b[column * (k / weight_block_size) + block] =
                0.004f + 0.0005f * (float)((column + block) % 5);
            data.real_zero_b[column * (k / weight_block_size) + block] =
                0.015f + 0.002f * (float)((column + 2 * block) % 4);
        }
        data.bias[column] = 0.02f * (float)((int)(column % 5) - 2);
    }

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column)
            data.c[row * n + column] = 0.01f * (float)((int)((row + column) % 7) - 3);
    }

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            float value = data.c[row * n + column] + data.bias[column];
            for (size_t group = 0; group < k / weight_block_size; ++group) {
                int32_t dot = 0;
                int32_t sum_a = 0;
                for (size_t depth = 0; depth < weight_block_size; ++depth) {
                    const size_t k_idx = group * weight_block_size + depth;
                    dot += (int32_t)data.a[row * k + k_idx] * (int32_t)data.b[column * k + k_idx];
                    sum_a += data.a[row * k + k_idx];
                }
                const float scale_a =
                    data.scale_a[row * (k / activation_block_size) +
                                 (group * weight_block_size) / activation_block_size];
                const float scale_b = data.scale_b[column * (k / weight_block_size) + group];
                value += scale_a * scale_b * (float)dot;
                if (asymmetric) {
                    value -= scale_a * data.real_zero_b[column * (k / weight_block_size) + group] *
                             (float)sum_a;
                }
            }
            data.reference[row * n + column] = value;
        }
    }
    return data;
}

inline void check_qsi8_block_quantized_gemm_result(const std::vector<float> &actual,
                                                   const std::vector<float> &reference)
{
    ASSERT_EQ(actual.size(), reference.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 2.0e-4f * std::max(1.0f, std::abs(reference[i]));
        EXPECT_NEAR(actual[i], reference[i], tolerance) << "index=" << i;
    }
}

inline std::vector<uint8_t> pack_qau5c32_source(const Qsi8BlockQuantizedGemmTestData &data)
{
    const size_t row_bytes = (data.k / 32) * 20;
    std::vector<uint8_t> packed(data.n * row_bytes);
    for (size_t column = 0; column < data.n; ++column) {
        for (size_t group = 0; group < data.k / 32; ++group) {
            uint8_t *dst = packed.data() + column * row_bytes + group * 20;
            const int8_t *src = data.b.data() + column * data.k + group * 32;
            for (size_t pair = 0; pair < 16; ++pair) {
                dst[pair] =
                    ((uint8_t)src[2 * pair] & 0xf) | (((uint8_t)src[2 * pair + 1] & 0xf) << 4);
            }
            for (size_t high = 0; high < 4; ++high) {
                uint8_t bits = 0;
                for (size_t bit = 0; bit < 8; ++bit)
                    bits |= ((uint8_t)src[high * 8 + bit] >> 4) << bit;
                dst[16 + high] = bits;
            }
        }
    }
    return packed;
}

inline std::vector<uint8_t> pack_qsi6c16_source(const Qsi8BlockQuantizedGemmTestData &data)
{
    const size_t row_bytes = (data.k / 16) * 12;
    std::vector<uint8_t> packed(data.n * row_bytes);
    for (size_t column = 0; column < data.n; ++column) {
        for (size_t group = 0; group < data.k / 16; ++group) {
            uint8_t *dst = packed.data() + column * row_bytes + group * 12;
            const int8_t *src = data.b.data() + column * data.k + group * 16;
            for (size_t pair = 0; pair < 8; ++pair) {
                const uint8_t lo = (uint8_t)((int32_t)src[2 * pair] + 32);
                const uint8_t hi = (uint8_t)((int32_t)src[2 * pair + 1] + 32);
                dst[pair] = (lo & 0xf) | ((hi & 0xf) << 4);
            }
            for (size_t high = 0; high < 4; ++high) {
                uint8_t bits = 0;
                for (size_t code = 0; code < 4; ++code) {
                    const uint8_t value = (uint8_t)((int32_t)src[high * 4 + code] + 32);
                    bits |= ((value >> 4) & 3) << (2 * code);
                }
                dst[8 + high] = bits;
            }
        }
    }
    return packed;
}

inline int32_t unpack_qau5c32_source(const uint8_t *row, size_t k_idx)
{
    const uint8_t *group = row + (k_idx / 32) * 20;
    const size_t in_group = k_idx % 32;
    const uint8_t low_byte = group[in_group / 2];
    const uint8_t low = (low_byte >> (4 * (in_group & 1))) & 0xf;
    const uint8_t high = (group[16 + in_group / 8] >> (in_group % 8)) & 1;
    return (int32_t)(low | (high << 4));
}

inline int32_t unpack_qsi6c16_source(const uint8_t *row, size_t k_idx)
{
    const uint8_t *group = row + (k_idx / 16) * 12;
    const size_t in_group = k_idx % 16;
    const uint8_t low_byte = group[in_group / 2];
    const uint8_t low = (low_byte >> (4 * (in_group & 1))) & 0xf;
    const uint8_t high = (group[8 + in_group / 4] >> (2 * (in_group % 4))) & 3;
    return (int32_t)(low | (high << 4)) - 32;
}

inline size_t get_qsi8_block_weight_row_bytes(Qsi8BlockWeightEncoding encoding, size_t k)
{
    switch (encoding) {
        case Qsi8BlockWeightEncoding::Qau4:
            return k / 2;
        case Qsi8BlockWeightEncoding::Qau5:
            return (k / 32) * 20;
        case Qsi8BlockWeightEncoding::Qsi6:
            return (k / 16) * 12;
        case Qsi8BlockWeightEncoding::Qau8:
        case Qsi8BlockWeightEncoding::Qsi8:
            return k;
    }
    return 0;
}

inline std::vector<uint8_t> make_qsi8_block_weight_source(
    const Qsi8BlockQuantizedGemmTestData &data, Qsi8BlockWeightEncoding encoding)
{
    if (encoding == Qsi8BlockWeightEncoding::Qau5)
        return pack_qau5c32_source(data);
    if (encoding == Qsi8BlockWeightEncoding::Qsi6)
        return pack_qsi6c16_source(data);

    const size_t row_bytes = get_qsi8_block_weight_row_bytes(encoding, data.k);
    std::vector<uint8_t> source(data.n * row_bytes);
    if (encoding == Qsi8BlockWeightEncoding::Qau4) {
        for (size_t column = 0; column < data.n; ++column) {
            for (size_t depth = 0; depth < data.k; depth += 2) {
                source[column * row_bytes + depth / 2] =
                    ((uint8_t)data.b[column * data.k + depth] & 0xf) |
                    (((uint8_t)data.b[column * data.k + depth + 1] & 0xf) << 4);
            }
        }
    } else {
        for (size_t i = 0; i < data.b.size(); ++i)
            source[i] = (uint8_t)data.b[i];
    }
    return source;
}

inline void check_qsi8_block_weight_n_interleaved_pack(const std::vector<uint8_t> &source,
                                                       const std::vector<uint8_t> &packed, size_t n,
                                                       size_t row_bytes, size_t n_step)
{
    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t nr = std::min(n - n_idx, n_step);
        const size_t tile_offset = n_idx * row_bytes;
        for (size_t byte = 0; byte < row_bytes; ++byte) {
            for (size_t lane = 0; lane < nr; ++lane) {
                EXPECT_EQ(packed[tile_offset + byte * nr + lane],
                          source[(n_idx + lane) * row_bytes + byte])
                    << "n_idx=" << n_idx << " byte=" << byte << " lane=" << lane;
            }
        }
    }
}

template <typename QParams>
struct PackedQsi8BlockQuantizedGemmTestFuncs
{
    using GetNStepFunc = size_t (*)(void);
    using GetPackedSizeFunc = size_t (*)(size_t, size_t);
    using GetPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using RunPackFunc = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);
    using MakeParamsFunc = QParams (*)(const float *, size_t, const int16_t *, size_t,
                                       const float *, const float *, size_t);
    using RunGemmFunc = void (*)(size_t, size_t, size_t, const void *, size_t, size_t, const void *,
                                 size_t, size_t, const void *, size_t, void *, size_t, const void *,
                                 float, float, const QParams *);

    size_t activation_block_size;
    size_t weight_block_size;
    bool asymmetric;
    int32_t max_weight;
    Qsi8BlockWeightEncoding weight_encoding;
    GetNStepFunc get_n_step;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    GetPackedOffsetFunc get_a_packed_offset;
    GetPackedOffsetFunc get_b_packed_offset;
    RunPackFunc run_a_pack;
    RunPackFunc run_bt_pack;
    MakeParamsFunc make_params;
    RunGemmFunc run_gemm;
    Qsi8BlockPackedLayout packed_layout = Qsi8BlockPackedLayout::NInterleaved;
    bool test_clamp = false;
};

// Ztt keeps full, zero-padded NLEN squares and recenters normalized U4 to I4.
template <typename QParams>
void check_qsi8_block_ztt_pack(const Qsi8BlockQuantizedGemmTestData &data,
                               const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                               const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs)
{
    const size_t s = funcs.get_n_step();
    const size_t pm = (data.m + s - 1) / s * s;
    const size_t pn = (data.n + s - 1) / s * s;
    const size_t pk = a.size() / pm;
    const bool compact = funcs.weight_encoding == Qsi8BlockWeightEncoding::Qau4;
    ASSERT_GE(pk, data.k);
    ASSERT_EQ(b.size(), pn * pk / (compact ? 2 : 1));
    for (bool weight : {false, true}) {
        const size_t rows = weight ? pn : pm;
        const auto &packed = weight ? b : a;
        size_t index = 0;
        for (size_t tile = 0; tile < rows; tile += s) {
            for (size_t depth = 0; depth < pk; depth += s) {
                for (size_t row = 0; row < s; ++row) {
                    for (size_t lane = 0; lane < s; ++lane, ++index) {
                        int expected = 0;
                        if (tile + row < (weight ? data.n : data.m) && depth + lane < data.k) {
                            expected =
                                (weight ? data.b : data.a)[(tile + row) * data.k + depth + lane];
                            if (weight && compact)
                                expected -= 8;
                        }
                        const int actual =
                            weight && compact ? (int)((packed[index / 2] >> (4 * (index % 2))) & 15)
                                              : (int)packed[index];
                        EXPECT_EQ(actual, expected & (weight && compact ? 15 : 255))
                            << "weight=" << weight << " element=" << index;
                    }
                }
            }
        }
    }
}

template <typename QParams>
void check_qsi8_block_two_pass_pack(size_t m, size_t n, size_t k,
                                    const Qsi8BlockQuantizedGemmTestData &data,
                                    const std::vector<uint8_t> &b_source,
                                    const std::vector<uint8_t> &a_packed,
                                    const std::vector<uint8_t> &b_packed,
                                    const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs)
{
    const size_t split = funcs.activation_block_size;
    const size_t a_offset = funcs.get_a_packed_offset(0, split, k, m);
    const size_t b_offset = funcs.get_b_packed_offset(0, split, k, n);
    const size_t b_source_offset = get_qsi8_block_weight_row_bytes(funcs.weight_encoding, split);
    std::vector<uint8_t> a_two_pass(a_packed.size(), 0xa5);
    std::vector<uint8_t> b_two_pass(b_packed.size(), 0xa5);

    funcs.run_a_pack(m, split, k, k, 0, data.a.data(), a_two_pass.data());
    funcs.run_a_pack(m, k - split, k, k, split, data.a.data() + split,
                     a_two_pass.data() + a_offset);
    funcs.run_bt_pack(n, split, k, k, 0, b_source.data(), b_two_pass.data());
    funcs.run_bt_pack(n, k - split, k, k, split, b_source.data() + b_source_offset,
                      b_two_pass.data() + b_offset);

    ASSERT_EQ(a_two_pass.size(), a_packed.size());
    for (size_t byte = 0; byte < a_packed.size(); ++byte)
        EXPECT_EQ(a_two_pass[byte], a_packed[byte]) << "A byte=" << byte;
    ASSERT_EQ(b_two_pass.size(), b_packed.size());
    for (size_t byte = 0; byte < b_packed.size(); ++byte)
        EXPECT_EQ(b_two_pass[byte], b_packed[byte]) << "B byte=" << byte;
}

template <typename QParams, typename ExpandedQParams>
void run_packed_qsi8_block_quantized_gemm_test(
    size_t m, size_t n, size_t k, bool compare_expanded,
    const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs,
    const PackedQsi8BlockQuantizedGemmTestFuncs<ExpandedQParams> &expanded_funcs,
    bool test_split_k = true)
{
    SCOPED_TRACE(::testing::Message()
                 << "m=" << m << " n=" << n << " k=" << k << " split_k=" << test_split_k);
    auto data = make_qsi8_block_quantized_gemm_test_data(n, k, funcs.activation_block_size,
                                                         funcs.weight_block_size, funcs.asymmetric,
                                                         funcs.max_weight, m);
    const std::vector<uint8_t> b_source =
        make_qsi8_block_weight_source(data, funcs.weight_encoding);
    std::vector<uint8_t> a_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> b_packed(funcs.get_b_packed_size(n, k));
    std::vector<float> output(m * n);

    if (funcs.packed_layout != Qsi8BlockPackedLayout::ZttSquare)
        ASSERT_EQ(b_packed.size(), b_source.size());
    funcs.run_a_pack(m, k, k, k, 0, data.a.data(), a_packed.data());
    funcs.run_bt_pack(n, k, k, k, 0, b_source.data(), b_packed.data());
    const size_t n_step = funcs.get_n_step();
    const size_t row_bytes = b_source.size() / n;
    if (funcs.packed_layout == Qsi8BlockPackedLayout::ZttSquare) {
        check_qsi8_block_ztt_pack(data, a_packed, b_packed, funcs);
    } else {
        check_qsi8_block_weight_n_interleaved_pack(b_source, b_packed, n, row_bytes, n_step);
    }
    if (funcs.packed_layout == Qsi8BlockPackedLayout::ZttSquare &&
        k >= 2 * funcs.activation_block_size) {
        check_qsi8_block_two_pass_pack(m, n, k, data, b_source, a_packed, b_packed, funcs);
    }

    const QParams params = funcs.make_params(data.scale_a.data(), k / funcs.activation_block_size,
                                             data.bsums_a.data(), k / 16, data.scale_b.data(),
                                             data.real_zero_b.data(), k / funcs.weight_block_size);
    funcs.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, data.c.data(), n,
                   output.data(), n, data.bias.data(), -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity(), &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);

    if (funcs.test_clamp) {
        constexpr float clamp_min = -0.04f;
        constexpr float clamp_max = 0.05f;
        funcs.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, data.c.data(), n,
                       output.data(), n, data.bias.data(), clamp_min, clamp_max, &params);
        std::vector<float> clamped_reference = data.reference;
        for (float &value : clamped_reference)
            value = std::min(std::max(value, clamp_min), clamp_max);
        check_qsi8_block_quantized_gemm_result(output, clamped_reference);
    }

    if (test_split_k && k >= 2 * funcs.activation_block_size) {
        SCOPED_TRACE("split-K continuation");
        std::vector<float> partial(m * n);
        const size_t split = funcs.activation_block_size;
        funcs.run_gemm(m, n, split, a_packed.data(), k, 0, b_packed.data(), k, 0, data.c.data(), n,
                       partial.data(), n, nullptr, -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), &params);
        const size_t a_offset = funcs.get_a_packed_offset(0, split, k, m);
        const size_t b_offset = funcs.get_b_packed_offset(0, split, k, n);
        const size_t packed_lanes =
            funcs.packed_layout == Qsi8BlockPackedLayout::ZttSquare ? n_step : std::min(n, n_step);
        EXPECT_EQ(b_offset,
                  packed_lanes * get_qsi8_block_weight_row_bytes(funcs.weight_encoding, split));
        funcs.run_gemm(m, n, k - split, a_packed.data() + a_offset, k, split,
                       b_packed.data() + b_offset, k, split, partial.data(), n, output.data(), n,
                       data.bias.data(), -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), &params);
        check_qsi8_block_quantized_gemm_result(output, data.reference);
    }

    if (compare_expanded) {
        const std::vector<uint8_t> expanded_b_source =
            make_qsi8_block_weight_source(data, expanded_funcs.weight_encoding);
        std::vector<uint8_t> expanded_a_packed(expanded_funcs.get_a_packed_size(m, k));
        std::vector<uint8_t> expanded_b_packed(expanded_funcs.get_b_packed_size(n, k));
        std::vector<float> expanded_output(m * n);
        expanded_funcs.run_a_pack(m, k, k, k, 0, data.a.data(), expanded_a_packed.data());
        expanded_funcs.run_bt_pack(n, k, k, k, 0, expanded_b_source.data(),
                                   expanded_b_packed.data());
        const ExpandedQParams expanded_params = expanded_funcs.make_params(
            data.scale_a.data(), k / expanded_funcs.activation_block_size, data.bsums_a.data(),
            k / 16, data.scale_b.data(), data.real_zero_b.data(),
            k / expanded_funcs.weight_block_size);
        expanded_funcs.run_gemm(m, n, k, expanded_a_packed.data(), k, 0, expanded_b_packed.data(),
                                k, 0, data.c.data(), n, expanded_output.data(), n, data.bias.data(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(), &expanded_params);
        check_qsi8_block_quantized_gemm_result(output, expanded_output);
    }
}

template <typename QParams>
void run_packed_qsi8_block_quantized_gemm_test(
    size_t m, size_t n, size_t k, const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs,
    bool test_split_k = true)
{
    run_packed_qsi8_block_quantized_gemm_test(m, n, k, false, funcs, funcs, test_split_k);
}

// Exercise caller contracts with independent source, packed, metadata and output
// strides. The reference consumes unpacked codes and computes each integer group
// independently; it never uses the implementation's packing or mask helpers.
template <typename QParams>
void run_qsi8_block_ztt_contract_test(const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs)
{
    const size_t s = funcs.get_n_step(), m = 2 * s + 1, n = 2 * s + 3, k = 768;
    const size_t group = funcs.weight_block_size;
    const bool compact = funcs.weight_encoding == Qsi8BlockWeightEncoding::Qau4;
    const size_t lda = k + 32, ldb = k + 64, packed_lda = k, packed_ldb = k + 256;
    const size_t ldsa = k / 256 + 2, ldsums = k / 16 + 3, ldsb = k / group + 4;
    std::vector<int8_t> a(m * lda);
    std::vector<uint8_t> b(n * ldb / (compact ? 2 : 1));
    std::vector<float> sa(m * ldsa, 0), sb(n * ldsb, 0), zb(n * ldsb, 0);
    std::vector<int16_t> sums(m * ldsums, 0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t q = 0; q < k; ++q) {
            const int8_t value = q / 256 == 2 ? 0 : ((q + i) % 3 == 0 ? -128 : 127);
            a[i * lda + q] = value;
            sums[i * ldsums + q / 16] += value;
        }
        for (size_t q = 0; q < k / 256; ++q)
            sa[i * ldsa + q] = (q + i) % 2 ? -0.125f : 0.0625f;
    }
    const auto weight = [compact](size_t j, size_t q) {
        return compact ? ((j + q) % 3 == 0 ? 0 : 15) : ((j + q) % 3 == 0 ? -32 : 31);
    };
    for (size_t j = 0; j < n; ++j) {
        for (size_t q = 0; q < k; ++q) {
            const int value = weight(j, q);
            if (compact)
                b[(j * ldb + q) / 2] |= (value & 15) << (4 * (q % 2));
            else
                b[j * ldb + q] = static_cast<uint8_t>(value);
        }
        for (size_t q = 0; q < k / group; ++q) {
            sb[j * ldsb + q] = (q + j) % 3 == 0 ? -0.25f : 0.125f;
            zb[j * ldsb + q] = (q + j) % 2 ? -0.5f : 0.75f;
        }
    }
    constexpr size_t guard = 32;
    const size_t asize = funcs.get_a_packed_size(m, packed_lda);
    const size_t bsize = funcs.get_b_packed_size(n, packed_ldb);
    std::vector<uint8_t> ap(asize + 2 * guard, 0xa5), bp(bsize + 2 * guard, 0xa5);
    funcs.run_a_pack(m, k, lda, packed_lda, 0, a.data(), ap.data() + guard);
    funcs.run_bt_pack(n, k, ldb, packed_ldb, 0, b.data(), bp.data() + guard);
    for (size_t r = 0; r < guard; ++r) {
        ASSERT_EQ(ap[r], 0xa5);
        ASSERT_EQ(ap[guard + asize + r], 0xa5);
        ASSERT_EQ(bp[r], 0xa5);
        ASSERT_EQ(bp[guard + bsize + r], 0xa5);
    }
    const size_t ldc = n + 3, ldd = n + 7;
    std::vector<float> c(m * ldc, 0.25f), bias(n, -0.125f);
    const float inf = std::numeric_limits<float>::infinity();
    for (bool window : {false, true}) {
        const size_t mi = window ? s : 0, ni = window ? s : 0, ki = window ? 256 : 0;
        const size_t km = window ? 256 : k;
        QParams params =
            funcs.make_params(sa.data() + mi * ldsa, ldsa, sums.data() + mi * ldsums, ldsums,
                              sb.data() + ni * ldsb, zb.data() + ni * ldsb, ldsb);
        for (bool with_c : {false, true}) {
            for (bool with_bias : {false, true}) {
                for (bool inplace : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "window=" << window << " C=" << with_c << " bias=" << with_bias
                                 << " inplace=" << inplace);
                    std::vector<float> d(2 * guard + m * ldd, 123456.0f);
                    if (inplace && with_c)
                        for (size_t i = mi; i < m; ++i)
                            for (size_t j = ni; j < n; ++j)
                                d[guard + i * ldd + j] = 0.25f;
                    float *dest = d.data() + guard + mi * ldd + ni;
                    funcs.run_gemm(
                        m - mi, n - ni, km,
                        ap.data() + guard + funcs.get_a_packed_offset(mi, ki, packed_lda, m - mi),
                        packed_lda, ki,
                        bp.data() + guard + funcs.get_b_packed_offset(ni, ki, packed_ldb, n - ni),
                        packed_ldb, ki,
                        with_c ? (inplace ? dest : c.data() + mi * ldc + ni) : nullptr,
                        inplace ? ldd : ldc, dest, ldd, with_bias ? bias.data() + ni : nullptr,
                        -inf, inf, &params);
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < ldd; ++j) {
                            if (i < mi || j < ni || j >= n) {
                                EXPECT_EQ(d[guard + i * ldd + j], 123456.0f);
                                continue;
                            }
                            double expected = (with_c ? 0.25 : 0.0) + (with_bias ? -0.125 : 0.0);
                            for (size_t g = ki; g < ki + km; g += group) {
                                int32_t dot = 0, sum = 0;
                                for (size_t q = g; q < g + group; ++q) {
                                    dot += a[i * lda + q] * weight(j, q);
                                    sum += a[i * lda + q];
                                }
                                expected +=
                                    (double)sa[i * ldsa + g / 256] *
                                    ((double)sb[j * ldsb + g / group] * dot -
                                     (compact ? (double)zb[j * ldsb + g / group] * sum : 0.0));
                            }
                            EXPECT_NEAR(d[guard + i * ldd + j], expected,
                                        2.0e-4 * std::max(1.0, std::abs(expected)));
                        }
                    }
                    for (size_t r = 0; r < guard; ++r) {
                        EXPECT_EQ(d[r], 123456.0f);
                        EXPECT_EQ(d[guard + m * ldd + r], 123456.0f);
                    }
                }
            }
        }
    }
    // Empty dimensions are no-ops; K=0 needs neither code nor metadata buffers.
    funcs.run_gemm(0, n, k, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, nullptr, 0, nullptr, -inf,
                   inf, nullptr);
    funcs.run_gemm(m, 0, k, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, nullptr, 0, nullptr, -inf,
                   inf, nullptr);
    std::vector<float> d(m * ldd, 0.25f);
    funcs.run_gemm(m, n, 0, nullptr, 0, 0, nullptr, 0, 0, d.data(), ldd, d.data(), ldd, bias.data(),
                   -0.1f, 0.1f, nullptr);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_FLOAT_EQ(d[i * ldd + j], 0.1f);
}

template <typename QParams>
void run_qsi8_block_ztt_integer_test(const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs)
{
    constexpr size_t k = 512;
    const size_t group = funcs.weight_block_size;
    const float inf = std::numeric_limits<float>::infinity();
    auto data = make_qsi8_block_quantized_gemm_test_data(1, k, 256, group, funcs.asymmetric,
                                                         funcs.max_weight, 1);
    for (int pattern : {0, 1, 2}) {
        for (size_t q = 0; q < k; ++q) {
            data.a[q] = pattern == 0 ? -128 : (q % 2 ? -128 : 127);
            data.b[q] = funcs.asymmetric
                            ? (pattern == 1 ? 0 : 15)
                            : (pattern == 2 ? (q % 2 ? -128 : 127) : (q % 2 ? -32 : 31));
        }
        for (size_t q = 0; q < k / 16; ++q) {
            int32_t sum = 0;
            for (size_t l = 0; l < 16; ++l)
                sum += data.a[q * 16 + l];
            data.bsums_a[q] = static_cast<int16_t>(sum);
        }
        std::fill(data.scale_a.begin(), data.scale_a.end(), 1.0f);
        std::fill(data.real_zero_b.begin(), data.real_zero_b.end(), 0.0f);
        auto bsource = make_qsi8_block_weight_source(data, funcs.weight_encoding);
        std::vector<uint8_t> ap(funcs.get_a_packed_size(1, k)), bp(funcs.get_b_packed_size(1, k));
        funcs.run_a_pack(1, k, k, k, 0, data.a.data(), ap.data());
        funcs.run_bt_pack(1, k, k, k, 0, bsource.data(), bp.data());
        for (size_t selected :
             {size_t(0), size_t(1), 256 / group - 1, 256 / group, k / group - 1}) {
            std::fill(data.scale_b.begin(), data.scale_b.end(), 0.0f);
            data.scale_b[selected] = 1.0f;
            const auto params =
                funcs.make_params(data.scale_a.data(), 2, data.bsums_a.data(), k / 16,
                                  data.scale_b.data(), data.real_zero_b.data(), k / group);
            float output;
            funcs.run_gemm(1, 1, k, ap.data(), k, 0, bp.data(), k, 0, nullptr, 1, &output, 1,
                           nullptr, -inf, inf, &params);
            int32_t dot = 0;
            for (size_t q = selected * group; q < (selected + 1) * group; ++q)
                dot += (int32_t)data.a[q] * data.b[q];
            EXPECT_EQ(output, static_cast<float>(dot))
                << "pattern=" << pattern << " group=" << selected;
        }
    }
}

// Pack the complete model-shaped B, then calculate only its first and last
// fragments. This is packing/slice evidence, not a full layer or model run.
template <typename QParams>
void run_qsi8_block_ztt_model_slices(const PackedQsi8BlockQuantizedGemmTestFuncs<QParams> &funcs)
{
    const size_t s = funcs.get_n_step();
    const float inf = std::numeric_limits<float>::infinity();
    for (const auto shape : {std::pair<size_t, size_t>{1024, 1024},
                             {1024, 2048},
                             {1024, 3072},
                             {2048, 1024},
                             {3072, 1024},
                             {151936, 1024}}) {
        const size_t n = shape.first, k = shape.second, group = funcs.weight_block_size;
        const bool compact = funcs.weight_encoding == Qsi8BlockWeightEncoding::Qau4;
        std::vector<int8_t> a(k, 1);
        std::vector<uint8_t> b(n * k / (compact ? 2 : 1), compact ? 0x11 : 1);
        std::vector<float> sa(k / 256, 1.0f), sb(n * k / group, 1.0f), zb(n * k / group, 0.0f);
        std::vector<int16_t> sums(k / 16, 16);
        std::vector<uint8_t> ap(funcs.get_a_packed_size(1, k)), bp(funcs.get_b_packed_size(n, k));
        funcs.run_a_pack(1, k, k, k, 0, a.data(), ap.data());
        funcs.run_bt_pack(n, k, k, k, 0, b.data(), bp.data());
        for (size_t ni : {size_t(0), (n - 1) / s * s}) {
            const size_t nr = std::min(s, n - ni);
            std::vector<float> output(nr);
            auto params = funcs.make_params(sa.data(), k / 256, sums.data(), k / 16,
                                            sb.data() + ni * (k / group),
                                            zb.data() + ni * (k / group), k / group);
            funcs.run_gemm(1, nr, k, ap.data(), k, 0,
                           bp.data() + funcs.get_b_packed_offset(ni, 0, k, nr), k, 0, nullptr, nr,
                           output.data(), nr, nullptr, -inf, inf, &params);
            for (float x : output)
                EXPECT_EQ(x, static_cast<float>(k));
        }
    }
}

// ============================================================================
// Q2_K/Q3_K-derived factorized-metadata GEMM test runner
// ============================================================================

enum class Qsi8KBlockWeightEncoding
{
    Qau2,
    Qsi3,
    Qau4,
    Qsi4,
};

struct Qsi8KBlockGemmTestData
{
    size_t m;
    size_t n;
    size_t k;
    bool asymmetric;
    std::vector<int8_t> a;
    std::vector<int8_t> b;
    std::vector<float> scale_a;
    std::vector<int16_t> bsums_a;
    std::vector<int8_t> qscale_b;
    std::vector<uint8_t> qmin_b;
    std::vector<float> super_scale_b;
    std::vector<float> super_min_b;
    std::vector<float> c;
    std::vector<float> bias;
    std::vector<float> reference;
};

inline size_t get_qsi8_kblock_weight_row_bytes(Qsi8KBlockWeightEncoding encoding, size_t k)
{
    switch (encoding) {
        case Qsi8KBlockWeightEncoding::Qau2:
            return k / 4;
        case Qsi8KBlockWeightEncoding::Qsi3:
            return (k / 16) * 6;
        case Qsi8KBlockWeightEncoding::Qau4:
        case Qsi8KBlockWeightEncoding::Qsi4:
            return k / 2;
    }
    return 0;
}

inline Qsi8KBlockGemmTestData make_qsi8_kblock_gemm_test_data(size_t m, size_t n, size_t k,
                                                              bool asymmetric)
{
    const size_t groups = k / 16;
    const size_t super_blocks = k / 256;
    Qsi8KBlockGemmTestData data = {};
    data.m = m;
    data.n = n;
    data.k = k;
    data.asymmetric = asymmetric;
    data.a.resize(m * k);
    data.b.resize(n * k);
    data.scale_a.resize(m * super_blocks);
    data.bsums_a.resize(m * groups);
    data.qscale_b.resize(n * groups);
    data.qmin_b.resize(n * groups);
    data.super_scale_b.resize(n * super_blocks);
    data.super_min_b.resize(n * super_blocks);
    data.c.resize(m * n);
    data.bias.resize(n);
    data.reference.resize(m * n);

    for (size_t row = 0; row < m; ++row) {
        for (size_t depth = 0; depth < k; ++depth)
            data.a[row * k + depth] = (int8_t)(((row * 11 + depth * 7) % 31) - 15);
        for (size_t block = 0; block < super_blocks; ++block)
            data.scale_a[row * super_blocks + block] = 0.006f + 0.001f * (float)((row + block) % 3);
        for (size_t group = 0; group < groups; ++group) {
            int32_t sum = 0;
            for (size_t depth = 0; depth < 16; ++depth)
                sum += data.a[row * k + group * 16 + depth];
            data.bsums_a[row * groups + group] = (int16_t)sum;
        }
    }
    for (size_t column = 0; column < n; ++column) {
        for (size_t depth = 0; depth < k; ++depth) {
            data.b[column * k + depth] = asymmetric
                                             ? (int8_t)((column * 5 + depth * 3) % 4)
                                             : (int8_t)((int)((column * 5 + depth * 3) % 8) - 4);
        }
        for (size_t group = 0; group < groups; ++group) {
            data.qscale_b[column * groups + group] =
                asymmetric ? (int8_t)(1 + (column + group) % 7)
                           : (int8_t)((int)((column + 2 * group) % 7) - 3);
            data.qmin_b[column * groups + group] = (uint8_t)((column + 3 * group) % 8);
        }
        for (size_t block = 0; block < super_blocks; ++block) {
            data.super_scale_b[column * super_blocks + block] =
                0.004f + 0.0005f * (float)((column + block) % 5);
            data.super_min_b[column * super_blocks + block] =
                0.003f + 0.00025f * (float)((column + 2 * block) % 4);
        }
        data.bias[column] = 0.02f * (float)((int)(column % 5) - 2);
    }
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            data.c[row * n + column] = 0.01f * (float)((int)((row + column) % 7) - 3);
            float value = data.c[row * n + column] + data.bias[column];
            for (size_t block = 0; block < super_blocks; ++block) {
                int32_t scaled_dot = 0;
                int32_t min_correction = 0;
                for (size_t local_group = 0; local_group < 16; ++local_group) {
                    const size_t group = block * 16 + local_group;
                    int32_t dot = 0;
                    for (size_t depth = 0; depth < 16; ++depth) {
                        const size_t k_idx = group * 16 + depth;
                        dot +=
                            (int32_t)data.a[row * k + k_idx] * (int32_t)data.b[column * k + k_idx];
                    }
                    scaled_dot += dot * data.qscale_b[column * groups + group];
                    if (asymmetric) {
                        min_correction += data.bsums_a[row * groups + group] *
                                          data.qmin_b[column * groups + group];
                    }
                }
                value +=
                    data.scale_a[row * super_blocks + block] *
                    (data.super_scale_b[column * super_blocks + block] * scaled_dot -
                     (asymmetric ? data.super_min_b[column * super_blocks + block] * min_correction
                                 : 0.0f));
            }
            data.reference[row * n + column] = value;
        }
    }
    return data;
}

inline std::vector<uint8_t> make_qsi8_kblock_weight_source(const Qsi8KBlockGemmTestData &data,
                                                           Qsi8KBlockWeightEncoding encoding)
{
    const size_t row_bytes = get_qsi8_kblock_weight_row_bytes(encoding, data.k);
    std::vector<uint8_t> source(data.n * row_bytes, 0);
    for (size_t column = 0; column < data.n; ++column) {
        for (size_t group = 0; group < data.k / 16; ++group) {
            const int8_t *values = data.b.data() + column * data.k + group * 16;
            uint8_t *packed = source.data() + column * row_bytes;
            if (encoding == Qsi8KBlockWeightEncoding::Qau2) {
                packed += group * 4;
                for (size_t byte = 0; byte < 4; ++byte) {
                    for (size_t index = 0; index < 4; ++index)
                        packed[byte] |= ((uint8_t)values[byte * 4 + index] & 3) << (2 * index);
                }
            } else if (encoding == Qsi8KBlockWeightEncoding::Qsi3) {
                packed += group * 6;
                for (size_t index = 0; index < 16; ++index) {
                    const uint8_t low =
                        values[index] < 0 ? (uint8_t)(values[index] + 4) : (uint8_t)values[index];
                    packed[index / 4] |= (low & 3) << (2 * (index % 4));
                    if (values[index] >= 0)
                        packed[4 + index / 8] |= (uint8_t)1 << (index % 8);
                }
            } else {
                packed += group * 8;
                for (size_t index = 0; index < 16; ++index)
                    packed[index / 2] |= ((uint8_t)values[index] & 0xf) << (4 * (index % 2));
            }
        }
    }
    return source;
}

template <typename QParams>
struct PackedQsi8KBlockGemmTestFuncs
{
    using GetNStepFunc = size_t (*)(void);
    using GetPackedSizeFunc = size_t (*)(size_t, size_t);
    using GetPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using RunPackFunc = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);
    using MakeParamsFunc = QParams (*)(const Qsi8KBlockGemmTestData &);
    using RunGemmFunc = void (*)(size_t, size_t, size_t, const void *, size_t, size_t, const void *,
                                 size_t, size_t, const void *, size_t, void *, size_t, const void *,
                                 float, float, const QParams *);

    bool asymmetric;
    Qsi8KBlockWeightEncoding weight_encoding;
    GetNStepFunc get_n_step;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    GetPackedOffsetFunc get_a_packed_offset;
    GetPackedOffsetFunc get_b_packed_offset;
    RunPackFunc run_a_pack;
    RunPackFunc run_bt_pack;
    MakeParamsFunc make_params;
    RunGemmFunc run_gemm;
};

template <typename QParams>
std::vector<float> run_packed_qsi8_kblock_gemm_once(
    const Qsi8KBlockGemmTestData &data, const PackedQsi8KBlockGemmTestFuncs<QParams> &funcs,
    bool test_split_k)
{
    SCOPED_TRACE(::testing::Message() << "m=" << data.m << " n=" << data.n << " k=" << data.k
                                      << " split_k=" << test_split_k);
    const std::vector<uint8_t> b_source =
        make_qsi8_kblock_weight_source(data, funcs.weight_encoding);
    std::vector<uint8_t> a_packed(funcs.get_a_packed_size(data.m, data.k));
    std::vector<uint8_t> b_packed(funcs.get_b_packed_size(data.n, data.k));
    EXPECT_EQ(b_packed.size(), b_source.size());
    funcs.run_a_pack(data.m, data.k, data.k, data.k, 0, data.a.data(), a_packed.data());
    funcs.run_bt_pack(data.n, data.k, data.k, data.k, 0, b_source.data(), b_packed.data());
    const size_t n_step = funcs.get_n_step();
    const size_t row_bytes = b_source.size() / data.n;
    check_qsi8_block_weight_n_interleaved_pack(b_source, b_packed, data.n, row_bytes, n_step);

    const QParams params = funcs.make_params(data);
    std::vector<float> output(data.m * data.n);
    funcs.run_gemm(data.m, data.n, data.k, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
                   data.c.data(), data.n, output.data(), data.n, data.bias.data(),
                   -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                   &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);

    constexpr float clamp_min = -0.04f;
    constexpr float clamp_max = 0.05f;
    funcs.run_gemm(data.m, data.n, data.k, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
                   data.c.data(), data.n, output.data(), data.n, data.bias.data(), clamp_min,
                   clamp_max, &params);
    std::vector<float> clamped_reference = data.reference;
    for (float &value : clamped_reference)
        value = std::min(std::max(value, clamp_min), clamp_max);
    check_qsi8_block_quantized_gemm_result(output, clamped_reference);

    if (test_split_k && data.k >= 512) {
        SCOPED_TRACE("split-K continuation");
        constexpr size_t split = 256;
        std::vector<float> partial(data.m * data.n);
        funcs.run_gemm(data.m, data.n, split, a_packed.data(), data.k, 0, b_packed.data(), data.k,
                       0, data.c.data(), data.n, partial.data(), data.n, nullptr,
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), &params);
        const size_t a_offset = funcs.get_a_packed_offset(0, split, data.k, data.m);
        const size_t b_offset = funcs.get_b_packed_offset(0, split, data.k, data.n);
        EXPECT_EQ(b_offset, std::min(data.n, n_step) *
                                get_qsi8_kblock_weight_row_bytes(funcs.weight_encoding, split));
        funcs.run_gemm(data.m, data.n, data.k - split, a_packed.data() + a_offset, data.k, split,
                       b_packed.data() + b_offset, data.k, split, partial.data(), data.n,
                       output.data(), data.n, data.bias.data(),
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), &params);
        check_qsi8_block_quantized_gemm_result(output, data.reference);
    }
    return output;
}

template <typename QParams>
void run_packed_qsi8_kblock_gemm_test(size_t m, size_t n, size_t k,
                                      const PackedQsi8KBlockGemmTestFuncs<QParams> &funcs,
                                      bool test_split_k = true)
{
    const auto data = make_qsi8_kblock_gemm_test_data(m, n, k, funcs.asymmetric);
    (void)run_packed_qsi8_kblock_gemm_once(data, funcs, test_split_k);
}

template <typename CompactQParams, typename ExpandedQParams>
void run_packed_qsi8_kblock_gemm_equivalence_test(
    size_t m, size_t n, size_t k, const PackedQsi8KBlockGemmTestFuncs<CompactQParams> &compact,
    const PackedQsi8KBlockGemmTestFuncs<ExpandedQParams> &expanded, bool test_split_k = true)
{
    const auto data = make_qsi8_kblock_gemm_test_data(m, n, k, compact.asymmetric);
    const auto compact_output = run_packed_qsi8_kblock_gemm_once(data, compact, test_split_k);
    const auto expanded_output = run_packed_qsi8_kblock_gemm_once(data, expanded, test_split_k);
    check_qsi8_block_quantized_gemm_result(compact_output, expanded_output);
}

template <typename QParams>
void run_packed_qsi8_kblock_gemm_zero_k_test(const PackedQsi8KBlockGemmTestFuncs<QParams> &funcs)
{
    const size_t m = 5, n = funcs.get_n_step() + 1, ld = n + 3;
    const auto data = make_qsi8_kblock_gemm_test_data(m, n, 256, funcs.asymmetric);
    const auto params = funcs.make_params(data);
    const int8_t a = 0;
    const uint8_t b = 0;
    std::vector<float> output(m * ld, 0.25f);
    const float inf = std::numeric_limits<float>::infinity();
    funcs.run_gemm(m, n, 0, &a, 0, 0, &b, 0, 0, output.data(), ld, output.data(), ld,
                   data.bias.data(), -inf, inf, &params);
    for (size_t r = 0; r < m; ++r) {
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(output[r * ld + j], 0.25f + data.bias[j]);
        for (size_t j = n; j < ld; ++j)
            EXPECT_EQ(output[r * ld + j], 0.25f);
    }
}

// Compute from scalar, unpacked codes. Do not use either packed decoder or the
// implementation's narrowed accumulator in the oracle.
template <typename QParams>
void run_packed_qsi8_kblock_gemm_extreme_test(size_t m, size_t n, size_t k,
                                              const PackedQsi8KBlockGemmTestFuncs<QParams> &funcs)
{
    const bool q2 = funcs.asymmetric;
    const size_t groups = k / 16;
    const size_t blocks = k / 256;
    const size_t ldq = groups + 3;
    const size_t lds = blocks + 2;
    const size_t ldsum = groups + 5;
    const size_t ldd = n + 7;
    auto data = make_qsi8_kblock_gemm_test_data(m, n, k, q2);
    for (size_t r = 0; r < m; ++r) {
        for (size_t z = 0; z < k; ++z) {
            const int8_t values[] = {-128, 127, -1, 0, 1};
            data.a[r * k + z] = values[(r + z / 16) % 5];
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t z = 0; z < k; ++z)
            data.b[j * k + z] = q2 ? (j + z / 16) % 4 : int((j + z / 16) % 8) - 4;
    }
    std::vector<float> sa(m * lds, 0), ss(n * lds, 0), sm(n * lds, 0);
    std::vector<int16_t> sums(m * ldsum, 0);
    std::vector<int8_t> qs(n * ldq, 0);
    std::vector<uint8_t> qm(n * ldq, 0);
    const uint8_t uq[] = {0, 15, 127, 128, 255};
    const int8_t sq[] = {-128, -127, -1, 0, 1, 127};
    for (size_t r = 0; r < m; ++r) {
        for (size_t b = 0; b < blocks; ++b)
            sa[r * lds + b] = (b % 2 ? -1.0f : 1.0f) / 256.0f;
        for (size_t g = 0; g < groups; ++g) {
            int64_t sum = 0;
            for (size_t z = 0; z < 16; ++z)
                sum += data.a[r * k + g * 16 + z];
            sums[r * ldsum + g] = static_cast<int16_t>(sum);
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t b = 0; b < blocks; ++b) {
            ss[j * lds + b] = 1.0f / 256.0f;
            sm[j * lds + b] = 1.0f / 512.0f;
        }
        for (size_t g = 0; g < groups; ++g) {
            if (q2) {
                const uint8_t value = uq[(j + g) % 5];
                std::memcpy(&qs[j * ldq + g], &value, 1);
            } else {
                qs[j * ldq + g] = sq[(j + g) % 6];
            }
            qm[j * ldq + g] = uq[(j + g + 2) % 5];
        }
    }
    QParams params = funcs.make_params(data);
    params.scale_a = sa.data();
    params.ldsa = lds;
    params.qscale_b = reinterpret_cast<decltype(params.qscale_b)>(qs.data());
    params.ldqb = ldq;
    params.super_scale_b = ss.data();
    params.ldssb = lds;
    if constexpr (std::is_same_v<decltype(params.qscale_b), const uint8_t *>) {
        params.bsums_a = sums.data();
        params.ldbsuma = ldsum;
        params.qmin_b = qm.data();
        params.super_min_b = sm.data();
    }
    const auto source = make_qsi8_kblock_weight_source(data, funcs.weight_encoding);
    std::vector<uint8_t> ap(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> bp(funcs.get_b_packed_size(n, k));
    funcs.run_a_pack(m, k, k, k, 0, data.a.data(), ap.data());
    funcs.run_bt_pack(n, k, k, k, 0, source.data(), bp.data());
    const float inf = std::numeric_limits<float>::infinity();
    for (bool split : {false, true}) {
        std::vector<float> output(m * ldd, 123456.0f);
        std::vector<float> expected(output);
        const size_t step = split ? 256 : k;
        for (size_t start = 0; start < k; start += step) {
            const size_t count = std::min(step, k - start);
            const bool last = start + count == k;
            for (size_t r = 0; r < m; ++r) {
                for (size_t j = 0; j < n; ++j) {
                    float result = 0;
                    double accurate = 0;
                    double magnitude = 0;
                    for (size_t b = start / 256; b < (start + count) / 256; ++b) {
                        int64_t dot = 0, correction = 0;
                        for (size_t g = b * 16; g < (b + 1) * 16; ++g) {
                            int64_t group_dot = 0, group_sum = 0;
                            for (size_t z = 0; z < 16; ++z) {
                                const int av = data.a[r * k + g * 16 + z];
                                group_dot += av * int(data.b[j * k + g * 16 + z]);
                                group_sum += av;
                            }
                            const int scale =
                                q2 ? static_cast<uint8_t>(qs[j * ldq + g]) : qs[j * ldq + g];
                            dot += group_dot * scale;
                            if (q2)
                                correction += group_sum * qm[j * ldq + g];
                        }
                        const float ad = float(dot) * sa[r * lds + b];
                        result = std::fma(ad, ss[j * lds + b], result);
                        if (q2) {
                            const float am = float(correction) * sa[r * lds + b];
                            result = std::fma(-am, sm[j * lds + b], result);
                        }
                        const double d = double(dot) * sa[r * lds + b] * ss[j * lds + b];
                        const double v = double(correction) * sa[r * lds + b] * sm[j * lds + b];
                        accurate += d - v;
                        magnitude += std::abs(d) + std::abs(v);
                    }
                    EXPECT_NEAR(double(result), accurate,
                                8 * std::numeric_limits<float>::epsilon() * (1 + magnitude));
                    if (start)
                        result += expected[r * ldd + j];
                    if (last)
                        result += data.bias[j];
                    expected[r * ldd + j] = result;
                }
            }
            funcs.run_gemm(m, n, count, ap.data() + funcs.get_a_packed_offset(0, start, k, m), k,
                           start, bp.data() + funcs.get_b_packed_offset(0, start, k, n), k, start,
                           start ? output.data() : nullptr, ldd, output.data(), ldd,
                           last ? data.bias.data() : nullptr, -inf, inf, &params);
            for (size_t i = 0; i < output.size(); ++i) {
                ASSERT_EQ(std::memcmp(&output[i], &expected[i], sizeof(float)), 0)
                    << "index=" << i << " split=" << split << " k_offset=" << start;
            }
        }
    }
}

// ============================================================================
// Q8_0 activation x Q1_0/Q2_0/Q4_0/Q5_0 source-aware execution tests
// ============================================================================

enum class Qx0SourceEncoding
{
    Bipolar1,
    U2Z1,
    I4,
    I5,
};

inline size_t qx0_source_block_size(Qx0SourceEncoding encoding)
{
    switch (encoding) {
        case Qx0SourceEncoding::Bipolar1:
            return 128;
        case Qx0SourceEncoding::U2Z1:
            return 64;
        case Qx0SourceEncoding::I4:
        case Qx0SourceEncoding::I5:
            return 32;
    }
    return 0;
}

inline size_t qx0_row_bytes(Qx0SourceEncoding encoding, size_t k)
{
    switch (encoding) {
        case Qx0SourceEncoding::Bipolar1:
            return k / 8;
        case Qx0SourceEncoding::U2Z1:
            return k / 4;
        case Qx0SourceEncoding::I4:
            return k / 2;
        case Qx0SourceEncoding::I5:
            return (k / 32) * 20;
    }
    return 0;
}

inline int8_t qx0_pattern_value(Qx0SourceEncoding encoding, size_t index)
{
    switch (encoding) {
        case Qx0SourceEncoding::Bipolar1: {
            static constexpr int8_t values[] = {-1, 1};
            return values[index % 2];
        }
        case Qx0SourceEncoding::U2Z1: {
            static constexpr int8_t values[] = {-1, 0, 1, 2};
            return values[index % 4];
        }
        case Qx0SourceEncoding::I4: {
            static constexpr int8_t values[] = {-8, -1, 0, 7};
            return values[index % 4];
        }
        case Qx0SourceEncoding::I5: {
            static constexpr int8_t values[] = {-16, -1, 0, 15};
            return values[index % 4];
        }
    }
    return 0;
}

inline void set_qx0_raw_value(std::vector<uint8_t> &raw, Qx0SourceEncoding encoding, size_t k,
                              size_t row, size_t kk, int8_t value)
{
    uint8_t *dst = raw.data() + row * qx0_row_bytes(encoding, k);
    switch (encoding) {
        case Qx0SourceEncoding::Bipolar1:
            dst[kk / 8] |= static_cast<uint8_t>((value > 0) << (kk % 8));
            break;
        case Qx0SourceEncoding::U2Z1:
            dst[kk / 4] |= static_cast<uint8_t>((value + 1) << ((kk % 4) * 2));
            break;
        case Qx0SourceEncoding::I4: {
            const size_t block_base = (kk / 32) * 16;
            const size_t in_block = kk % 32;
            const size_t byte = block_base + in_block % 16;
            const uint8_t code = static_cast<uint8_t>(value + 8);
            dst[byte] |= static_cast<uint8_t>(code << ((in_block / 16) * 4));
            break;
        }
        case Qx0SourceEncoding::I5: {
            const size_t block_base = (kk / 32) * 20;
            const size_t in_block = kk % 32;
            const uint8_t code = static_cast<uint8_t>(value + 16);
            dst[block_base + 4 + in_block % 16] |=
                static_cast<uint8_t>((code & 15) << ((in_block / 16) * 4));
            dst[block_base + in_block / 8] |=
                static_cast<uint8_t>(((code >> 4) & 1) << (in_block % 8));
            break;
        }
    }
}

inline int8_t get_qx0_raw_value(const std::vector<uint8_t> &raw, Qx0SourceEncoding encoding,
                                size_t k, size_t row, size_t kk)
{
    const uint8_t *src = raw.data() + row * qx0_row_bytes(encoding, k);
    switch (encoding) {
        case Qx0SourceEncoding::Bipolar1:
            return ((src[kk / 8] >> (kk % 8)) & 1) ? 1 : -1;
        case Qx0SourceEncoding::U2Z1:
            return static_cast<int8_t>(((src[kk / 4] >> ((kk % 4) * 2)) & 3) - 1);
        case Qx0SourceEncoding::I4: {
            const size_t block_base = (kk / 32) * 16;
            const size_t in_block = kk % 32;
            const uint8_t code = (src[block_base + in_block % 16] >> ((in_block / 16) * 4)) & 15;
            return static_cast<int8_t>(code) - 8;
        }
        case Qx0SourceEncoding::I5: {
            const size_t block_base = (kk / 32) * 20;
            const size_t in_block = kk % 32;
            const uint8_t low = (src[block_base + 4 + in_block % 16] >> ((in_block / 16) * 4)) & 15;
            const uint8_t high = (src[block_base + in_block / 8] >> (in_block % 8)) & 1;
            return static_cast<int8_t>(low | (high << 4)) - 16;
        }
    }
    return 0;
}

inline std::vector<uint8_t> adapt_qx0_raw_to_execution(const std::vector<uint8_t> &raw,
                                                       Qx0SourceEncoding encoding,
                                                       size_t execution_bits, size_t n, size_t k)
{
    const size_t execution_row_bytes = k * execution_bits / 8;
    std::vector<uint8_t> execution(n * execution_row_bytes, 0);
    for (size_t row = 0; row < n; ++row) {
        uint8_t *dst = execution.data() + row * execution_row_bytes;
        for (size_t kk = 0; kk < k; ++kk) {
            const int8_t value = get_qx0_raw_value(raw, encoding, k, row, kk);
            if (execution_bits == 8) {
                reinterpret_cast<int8_t *>(dst)[kk] = value;
                continue;
            }
            if (execution_bits == 4) {
                dst[kk / 2] |= static_cast<uint8_t>((value & 15) << ((kk % 2) * 4));
                continue;
            }
            switch (encoding) {
                case Qx0SourceEncoding::Bipolar1:
                    dst[kk / 8] |= static_cast<uint8_t>((value > 0) << (kk % 8));
                    break;
                case Qx0SourceEncoding::U2Z1:
                    dst[kk / 4] |= static_cast<uint8_t>((value + 1) << ((kk % 4) * 2));
                    break;
                case Qx0SourceEncoding::I4:
                    dst[kk / 2] |= static_cast<uint8_t>((value & 15) << ((kk % 2) * 4));
                    break;
                case Qx0SourceEncoding::I5: {
                    const size_t block_base = (kk / 32) * 20;
                    const size_t in_block = kk % 32;
                    const uint8_t code = static_cast<uint8_t>(value) & 31;
                    dst[block_base + in_block / 2] |=
                        static_cast<uint8_t>((code & 15) << ((in_block % 2) * 4));
                    dst[block_base + 16 + in_block / 8] |=
                        static_cast<uint8_t>(((code >> 4) & 1) << (in_block % 8));
                    break;
                }
            }
        }
    }
    return execution;
}

template <typename QParams>
struct PackedQsi8Qx0GemmTestFuncs
{
    using GetStepFunc = size_t (*)(void);
    using GetPackedOffsetFunc = size_t (*)(size_t, size_t, size_t, size_t);
    using GetPackedSizeFunc = size_t (*)(size_t, size_t);
    using RunPackFunc = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);
    using MakeParamsFunc = QParams (*)(const float *, size_t, const float *, size_t);
    using RunGemmFunc = void (*)(size_t, size_t, size_t, const void *, size_t, size_t, const void *,
                                 size_t, size_t, const void *, size_t, void *, size_t, const void *,
                                 float, float, const QParams *);

    Qx0SourceEncoding encoding;
    size_t execution_bits;
    size_t execution_scale_block;
    GetStepFunc get_m_step;
    GetStepFunc get_n_step;
    GetPackedOffsetFunc get_a_packed_offset;
    GetPackedOffsetFunc get_b_packed_offset;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    RunPackFunc run_a_pack;
    RunPackFunc run_bt_pack;
    MakeParamsFunc make_params;
    RunGemmFunc run_gemm;
};

template <typename QParams>
void run_packed_qsi8_qx0_source_test(size_t m, size_t n, size_t k,
                                     const PackedQsi8Qx0GemmTestFuncs<QParams> &funcs,
                                     bool check_single_tail_tile = true)
{
    const size_t source_block = qx0_source_block_size(funcs.encoding);
    ASSERT_EQ(k % source_block, 0U);
    std::vector<int8_t> a(m * k);
    for (size_t index = 0; index < a.size(); ++index)
        a[index] = static_cast<int8_t>((index * 37 + 13) % 255 - 127);

    std::vector<uint8_t> raw(n * qx0_row_bytes(funcs.encoding, k), 0);
    for (size_t row = 0; row < n; ++row) {
        for (size_t kk = 0; kk < k; ++kk) {
            const int8_t value = qx0_pattern_value(funcs.encoding, kk + row * 3);
            set_qx0_raw_value(raw, funcs.encoding, k, row, kk, value);
        }
    }
    const std::vector<uint8_t> execution =
        adapt_qx0_raw_to_execution(raw, funcs.encoding, funcs.execution_bits, n, k);

    switch (funcs.encoding) {
        case Qx0SourceEncoding::Bipolar1:
            EXPECT_EQ(raw[0], 0xaa);
            break;
        case Qx0SourceEncoding::U2Z1:
            EXPECT_EQ(raw[0], 0xe4);
            break;
        case Qx0SourceEncoding::I4:
            EXPECT_EQ(raw[0], 0x00);
            EXPECT_EQ(raw[1], 0x77);
            break;
        case Qx0SourceEncoding::I5:
            EXPECT_EQ(raw[0], 0xcc);
            EXPECT_EQ(raw[4], 0x00);
            EXPECT_EQ(raw[5], 0xff);
            break;
    }
    if (funcs.execution_bits == 8) {
        EXPECT_EQ(static_cast<int8_t>(execution[0]), qx0_pattern_value(funcs.encoding, 0));
        EXPECT_EQ(static_cast<int8_t>(execution[1]), qx0_pattern_value(funcs.encoding, 1));
    } else if (funcs.execution_bits == 4) {
        const uint8_t expected = static_cast<uint8_t>(qx0_pattern_value(funcs.encoding, 0)) & 15;
        const uint8_t expected_next =
            static_cast<uint8_t>(qx0_pattern_value(funcs.encoding, 1)) & 15;
        EXPECT_EQ(execution[0], expected | static_cast<uint8_t>(expected_next << 4));
    } else {
        switch (funcs.encoding) {
            case Qx0SourceEncoding::Bipolar1:
                EXPECT_EQ(execution[0], 0xaa);
                break;
            case Qx0SourceEncoding::U2Z1:
                EXPECT_EQ(execution[0], 0xe4);
                break;
            case Qx0SourceEncoding::I4:
                EXPECT_EQ(execution[0], 0xf8);
                EXPECT_EQ(execution[1], 0x70);
                break;
            case Qx0SourceEncoding::I5:
                EXPECT_EQ(execution[0], 0xf0);
                EXPECT_EQ(execution[16], 0x33);
                break;
        }
    }

    const size_t ldsa = k / 32;
    const size_t source_ldsb = k / source_block;
    const size_t ldsb = k / funcs.execution_scale_block;
    std::vector<float> scale_a(m * ldsa);
    std::vector<float> source_scale_b(n * source_ldsb);
    std::vector<float> scale_b(n * ldsb);
    for (size_t index = 0; index < scale_a.size(); ++index)
        scale_a[index] = 0.003f * static_cast<float>(1 + index % 7);
    for (size_t index = 0; index < source_scale_b.size(); ++index)
        source_scale_b[index] = 0.007f * static_cast<float>(1 + index % 5);
    for (size_t row = 0; row < n; ++row) {
        for (size_t group = 0; group < ldsb; ++group) {
            scale_b[row * ldsb + group] =
                source_scale_b[row * source_ldsb +
                               (group * funcs.execution_scale_block) / source_block];
        }
    }
    std::vector<float> c(m * n), bias(n), reference(m * n), output(m * n), partial(m * n);
    for (size_t index = 0; index < c.size(); ++index)
        c[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03125f;
    for (size_t column = 0; column < n; ++column)
        bias[column] = static_cast<float>(static_cast<int>(column % 5) - 2) * 0.0625f;

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            float value = c[row * n + column] + bias[column];
            for (size_t group_k = 0; group_k < k; group_k += 32) {
                int32_t dot = 0;
                for (size_t kk = 0; kk < 32; ++kk) {
                    dot += static_cast<int32_t>(a[row * k + group_k + kk]) *
                           get_qx0_raw_value(raw, funcs.encoding, k, column, group_k + kk);
                }
                value += static_cast<float>(dot) * scale_a[row * ldsa + group_k / 32] *
                         source_scale_b[column * source_ldsb + group_k / source_block];
            }
            reference[row * n + column] = value;
        }
    }

    EXPECT_EQ(funcs.get_a_packed_size(m, k), m * k);
    const size_t execution_row_bytes = k * funcs.execution_bits / 8;
    EXPECT_EQ(funcs.get_b_packed_size(n, k), n * execution_row_bytes);
    std::vector<uint8_t> a_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> b_packed(funcs.get_b_packed_size(n, k));
    funcs.run_a_pack(m, k, k, k, 0, a.data(), a_packed.data());
    funcs.run_bt_pack(n, k, k, k, 0, execution.data(), b_packed.data());
    const QParams params = funcs.make_params(scale_a.data(), ldsa, scale_b.data(), ldsb);
    const float inf = std::numeric_limits<float>::infinity();
    {
        SCOPED_TRACE("full K");
        funcs.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, c.data(), n,
                       output.data(), n, bias.data(), -inf, inf, &params);
        check_qsi8_block_quantized_gemm_result(output, reference);
    }

    const size_t split = 32;
    funcs.run_gemm(m, n, split, a_packed.data(), k, 0, b_packed.data(), k, 0, c.data(), n,
                   partial.data(), n, nullptr, -inf, inf, &params);
    const size_t a_offset = funcs.get_a_packed_offset(0, split, k, m);
    const size_t b_offset = funcs.get_b_packed_offset(0, split, k, n);
    funcs.run_gemm(m, n, k - split, a_packed.data() + a_offset, k, split,
                   b_packed.data() + b_offset, k, split, partial.data(), n, output.data(), n,
                   bias.data(), -inf, inf, &params);
    {
        SCOPED_TRACE("split K");
        check_qsi8_block_quantized_gemm_result(output, reference);
    }

    const size_t n_step = funcs.get_n_step();
    if (n > n_step)
        EXPECT_EQ(funcs.get_b_packed_offset(n_step, 0, k, n), n_step * execution_row_bytes);

    if (check_single_tail_tile) {
        SCOPED_TRACE("single M/N tail tile");
        run_packed_qsi8_qx0_source_test(1, 1, k, funcs, false);
    }
}

// Exercise the complete i8 range and long runs of extreme weights.
// Exact power-of-two scales make every expected result exactly representable.
template <typename Params>
void run_packed_qsi8_block_extreme_test(const PackedQsi8BlockQuantizedGemmTestFuncs<Params> &funcs,
                                        int low, int high)
{
    for (size_t m : {1u, 2u, 3u, 5u}) {
        const size_t n = funcs.get_n_step() + 1;
        constexpr size_t k = 512;
        for (int pattern = 0; pattern < 4; ++pattern) {
            auto data = make_qsi8_block_quantized_gemm_test_data(n, k, funcs.activation_block_size,
                                                                 funcs.weight_block_size,
                                                                 funcs.asymmetric, high, m);
            for (size_t i = 0; i < data.a.size(); ++i)
                data.a[i] = pattern == 0   ? -128
                            : pattern == 1 ? 127
                            : pattern == 2 ? -128
                                           : int(i * 29 % 256) - 128;
            for (size_t i = 0; i < data.b.size(); ++i)
                data.b[i] = pattern < 2    ? high
                            : pattern == 2 ? low
                                           : low + int(i * 13 % (high - low + 1));
            std::fill(data.scale_a.begin(), data.scale_a.end(), 0.5f);
            std::fill(data.scale_b.begin(), data.scale_b.end(), 0.125f);
            std::fill(data.real_zero_b.begin(), data.real_zero_b.end(), 0.25f);
            for (size_t row = 0; row < m; ++row) {
                for (size_t group = 0; group < k / 16; ++group) {
                    int sum = 0;
                    for (size_t j = 0; j < 16; ++j)
                        sum += data.a[row * k + group * 16 + j];
                    data.bsums_a[row * (k / 16) + group] = sum;
                }
            }
            const auto src = make_qsi8_block_weight_source(data, funcs.weight_encoding);
            std::vector<uint8_t> ap(funcs.get_a_packed_size(m, k));
            std::vector<uint8_t> bp(funcs.get_b_packed_size(n, k));
            funcs.run_a_pack(m, k, k, k, 0, data.a.data(), ap.data());
            funcs.run_bt_pack(n, k, k, k, 0, src.data(), bp.data());
            const auto params = funcs.make_params(
                data.scale_a.data(), k / funcs.activation_block_size, data.bsums_a.data(), k / 16,
                data.scale_b.data(), data.real_zero_b.data(), k / funcs.weight_block_size);
            const size_t ldd = n + 3;
            for (bool split : {false, true}) {
                std::vector<float> d(m * ldd, 12345.0f);
                if (split) {
                    funcs.run_gemm(m, n, 256, ap.data(), k, 0, bp.data(), k, 0, nullptr, ldd,
                                   d.data(), ldd, nullptr, -INFINITY, INFINITY, &params);
                    funcs.run_gemm(m, n, 256, ap.data() + funcs.get_a_packed_offset(0, 256, k, m),
                                   k, 256, bp.data() + funcs.get_b_packed_offset(0, 256, k, n), k,
                                   256, d.data(), ldd, d.data(), ldd, nullptr, -INFINITY, INFINITY,
                                   &params);
                } else {
                    funcs.run_gemm(m, n, k, ap.data(), k, 0, bp.data(), k, 0, nullptr, ldd,
                                   d.data(), ldd, nullptr, -INFINITY, INFINITY, &params);
                }
                for (size_t row = 0; row < m; ++row) {
                    for (size_t col = 0; col < n; ++col) {
                        int dot = 0, sum = 0;
                        for (size_t j = 0; j < k; ++j) {
                            dot += int(data.a[row * k + j]) * int(data.b[col * k + j]);
                            sum += data.a[row * k + j];
                        }
                        const float expected =
                            dot * 0.0625f - (funcs.asymmetric ? sum * 0.125f : 0.0f);
                        EXPECT_EQ(d[row * ldd + col], expected)
                            << "pattern=" << pattern << " m=" << m << " col=" << col
                            << " split=" << split;
                    }
                    for (size_t col = n; col < ldd; ++col)
                        EXPECT_EQ(d[row * ldd + col], 12345.0f);
                }
            }
        }
    }
}

template <bool PerRow, typename Run>
void check_requantized_i8_rounding_modes(size_t m_step, size_t n_step, Run run)
{
    const size_t m = m_step + 1, n = n_step + 3, ldd = n + 5;
    const int32_t values[] = {1077952705, 1077952706, -1077952705, -1077952706};
    const int8_t expected[] = {0, 1, 0, -1};
    const float scale_a = 16777214.0f * 0x1p-24f;
    const float scale_b = 16711679.0f * 0x1p-55f;
    const float scale_d = 16777215.0f * 0x1p-24f;
    std::vector<float> channel_scale(PerRow ? m : n, PerRow ? scale_a : scale_b);
    std::vector<int32_t> bias(channel_scale.size());
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = values[i % 4];

    for (size_t k : {size_t(0), size_t(3)}) {
        const size_t lda = k + 3, ldb = (PerRow ? n : k) + 5;
        std::vector<int8_t> a(m * lda, 0), b((PerRow ? k + 1 : n) * ldb, 0);
        for (bool per_channel : {false, true}) {
            for (unsigned long mode = 0; mode < 5; ++mode) {
                const unsigned long vector_mode = mode % 4;
                std::vector<int8_t> d(m * ldd + 8, 93);
                unsigned long saved, saved_vector, after, after_vector;
                auto invoke = [&](const auto &params) {
                    asm volatile("csrrw %0, frm, %2\n\tcsrrw %1, vxrm, %3"
                                 : "=&r"(saved), "=&r"(saved_vector)
                                 : "r"(mode), "r"(vector_mode)
                                 : "memory");
                    run(m, n, k, a.data(), lda, b.data(), ldb, nullptr, n, bias.data(), d.data(),
                        ldd, -128, 127, &params);
                    asm volatile(
                        "csrr %0, frm\n\tcsrr %1, vxrm\n\t"
                        "csrw frm, %2\n\tcsrw vxrm, %3"
                        : "=&r"(after), "=&r"(after_vector)
                        : "r"(saved), "r"(saved_vector)
                        : "memory");
                };
                if constexpr (PerRow) {
                    tqt_qai8_qsi8dx_qai8_params params{};
                    params.scale_a = channel_scale.data();
                    params.quant_channel_a = per_channel ? m : 1;
                    params.scale_b = scale_b;
                    params.scale_d = scale_d;
                    invoke(params);
                } else {
                    tqt_qai8_qai8_qsi8cx_params params{};
                    params.scale_a = scale_a;
                    params.scale_b = channel_scale.data();
                    params.quant_channel_b = per_channel ? n : 1;
                    params.scale_d = scale_d;
                    invoke(params);
                }
                ASSERT_EQ(after, mode);
                ASSERT_EQ(after_vector, vector_mode);
                for (size_t row = 0; row < m; ++row) {
                    for (size_t col = 0; col < n; ++col)
                        ASSERT_EQ(d[row * ldd + col], expected[(PerRow ? row : col) % 4])
                            << "frm=" << mode << " k=" << k << " row=" << row << " col=" << col
                            << " per_channel=" << per_channel;
                    for (size_t col = n; col < ldd; ++col)
                        ASSERT_EQ(d[row * ldd + col], 93);
                }
                for (size_t i = m * ldd; i < d.size(); ++i)
                    ASSERT_EQ(d[i], 93);
            }
        }
    }
}

template <bool PerRow, typename Run>
void check_requantized_i8_contract(size_t m_step, size_t n_step, Run run)
{
    check_requantized_i8_rounding_modes<PerRow>(m_step, n_step, run);
    const float scales[] = {0.5f, 1.0f, 2.0f, 0x1p-33f, 0x1p16f};
    for (size_t k : {size_t(0), size_t(1), size_t(35)}) {
        for (int32_t input_zp : {-128, 37, 127}) {
            for (size_t tail : {size_t(0), size_t(1)}) {
                const size_t m = 2 * m_step + tail;
                const size_t n = 2 * n_step + tail;
                const size_t lda = k + 3;
                const size_t ldb = (PerRow ? n : k) + 5;
                const size_t ldc = n + 2;
                const size_t ldd = n + 7;
                std::vector<int8_t> a(m * lda), b((PerRow ? k : n) * ldb);
                std::vector<int32_t> c(m * ldc), bias(PerRow ? m : n);
                std::vector<float> channel_scale(bias.size());
                for (size_t i = 0; i < a.size(); ++i)
                    a[i] = (int8_t)((i * 71 + 13) % 256 - 128);
                for (size_t i = 0; i < b.size(); ++i)
                    b[i] = (int8_t)((i * 43 + 9) % 256 - 128);
                for (size_t i = 0; i < bias.size(); ++i) {
                    bias[i] = (int32_t)(i % 11) - 5;
                    channel_scale[i] = scales[i % 5];
                }
                // Cancellation leaves small, signed accumulators, including exact
                // halfway values. This also detects raw versus pre-corrected bias.
                for (size_t row = 0; row < m; ++row) {
                    for (size_t col = 0; col < n; ++col) {
                        int64_t dot = 0;
                        for (size_t i = 0; i < k; ++i) {
                            const int32_t av = a[row * lda + i] - (PerRow ? 0 : input_zp);
                            const int32_t bv =
                                b[PerRow ? i * ldb + col : col * ldb + i] - (PerRow ? input_zp : 0);
                            dot += (int64_t)av * bv;
                        }
                        c[row * ldc + col] = (int32_t)(-dot + (int)(row + col) % 17 - 8);
                    }
                }
                for (bool with_addends : {false, true}) {
                    const int8_t lo = with_addends ? -53 : -128;
                    const int8_t hi = with_addends ? 79 : 127;
                    const int32_t output_zp = -7;
                    std::vector<int8_t> d(m * ldd + 16, 93);
                    if constexpr (PerRow) {
                        tqt_qai8_qsi8dx_qai8_params params{};
                        params.scale_a = channel_scale.data();
                        params.quant_channel_a = m;
                        params.scale_b = 1.0f;
                        params.zero_point_b = input_zp;
                        params.scale_d = 1.0f;
                        params.zero_point_d = output_zp;
                        run(m, n, k, a.data(), lda, b.data(), ldb,
                            with_addends ? c.data() : nullptr, ldc,
                            with_addends ? bias.data() : nullptr, d.data(), ldd, lo, hi, &params);
                    } else {
                        tqt_qai8_qai8_qsi8cx_params params{};
                        params.scale_a = 1.0f;
                        params.zero_point_a = input_zp;
                        params.scale_b = channel_scale.data();
                        params.quant_channel_b = n;
                        params.scale_d = 1.0f;
                        params.zero_point_d = output_zp;
                        run(m, n, k, a.data(), lda, b.data(), ldb,
                            with_addends ? c.data() : nullptr, ldc,
                            with_addends ? bias.data() : nullptr, d.data(), ldd, lo, hi, &params);
                    }
                    for (size_t row = 0; row < m; ++row) {
                        for (size_t col = 0; col < n; ++col) {
                            int64_t dot =
                                with_addends ? c[row * ldc + col] + bias[PerRow ? row : col] : 0;
                            for (size_t i = 0; i < k; ++i) {
                                const int32_t av = a[row * lda + i] - (PerRow ? 0 : input_zp);
                                const int32_t bv = b[PerRow ? i * ldb + col : col * ldb + i] -
                                                   (PerRow ? input_zp : 0);
                                dot += (int64_t)av * bv;
                            }
                            ASSERT_GE(dot, INT32_MIN);
                            ASSERT_LE(dot, INT32_MAX);
                            ASSERT_EQ(
                                d[row * ldd + col],
                                reference_requantized_i8(dot, channel_scale[PerRow ? row : col],
                                                         output_zp, lo, hi))
                                << "m=" << m << " n=" << n << " k=" << k << " row=" << row
                                << " col=" << col << " zp=" << input_zp;
                        }
                        for (size_t col = n; col < ldd; ++col)
                            ASSERT_EQ(d[row * ldd + col], 93);
                    }
                    for (size_t i = m * ldd; i < d.size(); ++i)
                        ASSERT_EQ(d[i], 93);
                }
            }
        }
    }
}

// Exercise scales whose reciprocal overflows, unity/amplifying/reducing scales,
// and all five legal incoming FP rounding modes without leaking CSR state.
template <typename Run>
void check_lowbit_output_scales(size_t ns, Run run)
{
    const size_t m = 5, n = ns + 1, ldd = n + 3;
    std::vector<int32_t> bias(n);
    const int32_t values[] = {0, 1, -1, 3, -3, 7, -7, 127, -127, 1234567};
    for (size_t j = 0; j < n; ++j)
        bias[j] = values[j % 10];
    for (float output_scale : {1e-40f, 0.5f, 1.0f, 2.0f, 3.0f, 3e38f}) {
        const float input_scale = output_scale < 1e-30f ? output_scale : 1.0f;
        std::vector<int8_t> expected(n);
        for (size_t j = 0; j < n; ++j) {
            volatile float real = float(bias[j]) * input_scale;
            volatile float quotient = real / output_scale;
            const double x = quotient, base = std::floor(x), fraction = x - base;
            const double rounded =
                base +
                (fraction > 0.5 || (fraction == 0.5 && std::fmod(std::abs(base), 2.0) == 1.0));
            expected[j] = (int8_t)std::max(-53.0, std::min(79.0, rounded + 7.0));
        }
        for (unsigned long mode = 0; mode < 5; ++mode) {
            std::vector<int8_t> output(m * ldd + 8, 99);
            unsigned long saved, after;
            asm volatile("csrrw %0, frm, %1" : "=r"(saved) : "r"(mode) : "memory");
            run(m, n, input_scale, output_scale, bias.data(), output.data(), ldd);
            asm volatile("csrr %0, frm\n\tcsrw frm, %1" : "=&r"(after) : "r"(saved) : "memory");
            EXPECT_EQ(after, mode);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j)
                    EXPECT_EQ(output[i * ldd + j], expected[j])
                        << "scale=" << output_scale << " mode=" << mode << " col=" << j;
                for (size_t j = n; j < ldd; ++j)
                    EXPECT_EQ(output[i * ldd + j], 99);
            }
            for (size_t i = m * ldd; i < output.size(); ++i)
                EXPECT_EQ(output[i], 99);
        }
    }
}

// Scalar oracle uses decoded values and independent affine metadata. The runner
// selects a nonzero M/N/K window from full buffers through each public getter.
template <typename TA, typename TD, bool Affine, bool QuantA, bool RowA, typename Run>
void check_w4_contract(size_t ms, size_t ns, float tolerance, Run run)
{
    const float16_t lut16[16] = {3,  -1, (float16_t)0.5f,  -4, 7,  2, -2, 0, 1,
                                 -3, 4,  (float16_t)-0.5f, 6,  -6, 5, -5};
    const int8_t lut8[16] = {-127, 0, 23, -17, 64, -2, 1, 7, -5, 113, 8, -3, 11, -32, 2, -1};
    for (size_t bl : {size_t(2), size_t(32), size_t(64)}) {
        for (size_t extra : {size_t(0), size_t(2)}) {
            const size_t m = ms - 1 + extra, n = ns - 1 + extra;
            const size_t fm = ms + m, fn = ns + n, fk = 3 * bl, ki = bl, k = 2 * bl;
            const size_t ld = fk + 8, lds = 5, ldd = n + 5;
            std::vector<TA> a(fm * ld, (TA)0);
            std::vector<uint8_t> b(fn * ld / 2, 0);
            std::vector<float> sa(fm * lds, 0), sb(fn * lds, 0), ob(fn * lds, 0);
            std::vector<int32_t> za(fm, 0);
            std::vector<float> bias(fn);
            std::vector<TD> c(m * ldd), d(m * ldd + 8);
            for (size_t i = 0; i < fm; ++i) {
                za[i] = int(i % 3) - 1;
                for (size_t q = 0; q < lds; ++q)
                    sa[i * lds + q] = 0.25f * (1 + (i + q) % 2);
                for (size_t q = 0; q < fk; ++q)
                    a[i * ld + q] = (TA)(int((i * 3 + q) % 7) - 3);
            }
            for (size_t j = 0; j < fn; ++j) {
                bias[j] = 0.25f * (int(j % 7) - 3);
                for (size_t q = 0; q < lds; ++q) {
                    // Includes scale=0 with nonzero offset; cannot be encoded as -zp*scale.
                    sb[j * lds + q] = Affine && (j + q) % 3 == 0 ? 0.0f : 0.125f;
                    ob[j * lds + q] = 0.25f * (int((j + q) % 5) - 2);
                }
                for (size_t q = 0; q < fk; ++q) {
                    const uint8_t code = (j * 5 + q * 3) % 16;
                    b[j * ld / 2 + q / 2] |= code << (4 * (q & 1));
                }
            }
            for (size_t i = 0; i < c.size(); ++i)
                c[i] = to_target_type<TD>(0.5f * (int(i % 3) - 1));
            for (int custom = 0; custom < (Affine ? 1 : 2); ++custom) {
                for (int mask = 0; mask < 8; ++mask) {
                    SCOPED_TRACE(::testing::Message()
                                 << "bl=" << bl << " m=" << m << " n=" << n << " custom=" << custom
                                 << " options=" << mask);
                    std::fill(d.begin(), d.end(), to_target_type<TD>(123));
                    const float lo = mask & 4 ? -7.0f : -INFINITY;
                    const float hi = mask & 4 ? 9.0f : INFINITY;
                    run(m, n, k, bl, fm, fn, fk, ms, ns, ki, ld, lds, a, b, sa, za, sb, ob,
                        mask & 1 ? bias.data() : nullptr, mask & 2 ? c.data() : nullptr, d.data(),
                        ldd, lo, hi, custom ? lut16 : nullptr, custom ? lut8 : nullptr);
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            double ref = mask & 2 ? to_float(c[i * ldd + j]) : 0.0;
                            for (size_t q = ki; q < ki + k; ++q) {
                                const size_t bi = q / bl;
                                const uint8_t raw = b[(ns + j) * ld / 2 + q / 2];
                                const unsigned code = (raw >> (4 * (q & 1))) & 15;
                                double av = (double)a[(ms + i) * ld + q];
                                if constexpr (QuantA) {
                                    if constexpr (RowA)
                                        av = (av - za[ms + i]) * sa[(ms + i) * lds];
                                    else
                                        av *= sa[(ms + i) * lds + bi];
                                }
                                double bv = Affine ? double(code)
                                                   : (custom ? (QuantA ? double(lut8[code])
                                                                       : double(lut16[code]))
                                                             : double(int(code) - 8));
                                bv *= sb[(ns + j) * lds + bi];
                                if constexpr (Affine)
                                    bv += ob[(ns + j) * lds + bi];
                                ref += av * bv;
                            }
                            if (mask & 1)
                                ref += bias[ns + j];
                            ref = std::min(std::max(ref, double(lo)), double(hi));
                            const float expected = to_float(to_target_type<TD>((float)ref));
                            EXPECT_NEAR(to_float(d[i * ldd + j]), expected, tolerance)
                                << "row=" << i << " col=" << j;
                        }
                        for (size_t j = n; j < ldd; ++j)
                            EXPECT_EQ(to_float(d[i * ldd + j]), 123);
                    }
                    for (size_t i = m * ldd; i < d.size(); ++i)
                        EXPECT_EQ(to_float(d[i]), 123);
                }
            }
        }
    }
}

}  // namespace test
}  // namespace torq_tile

// ============================================================================
// W4A8 block-quantized TEST_P boilerplate macros
// ----------------------------------------------------------------------------
// Placed OUTSIDE the namespace so the token-pasted kernel function names
// (e.g. tqt_run_gemm_<KBASE>_4x1vl_rvv) resolve at global / extern "C" scope.
// KBASE is the part AFTER "gemm_" (e.g. "1xnbias_clamp_f32_qai8dx_qsi4c2t").
// Consumer TUs must:
//   1. `using namespace torq_tile::test;` (or qualified references) so the
//      Funcs templates and runner names are visible at macro expansion.
//   2. Define a static `common_w4a8_bq_test_values()` returning a
//      `testing::Values(...)` of GemmTestParams in the TU before invoking
//      the macros (the macros reference this name verbatim).
// ============================================================================

// Unpacked test: kernel base + TD + runner + funcs typename.
#define W4A8_BQ_DEFINE_UNPACKED_TEST(KBASE, TD, RUNNER, FUNCS_TYPE, BIAS_T)                   \
    static FUNCS_TYPE<TD> make_gemm_##KBASE##_funcs()                                         \
    {                                                                                         \
        FUNCS_TYPE<TD> funcs;                                                                 \
        funcs.get_m_step = tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv;                           \
        funcs.get_n_step = tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv;                           \
        funcs.run_gemm = clamp_test_run<tqt_run_gemm_##KBASE##_4x1vl_rvv, TD, 14, 12, 13, 15, \
                                        ClampSource::kColumnBias, false, 6, true, BIAS_T>();  \
        return funcs;                                                                         \
    }                                                                                         \
    class Test_gemm_##KBASE##_4x1vl_rvv : public testing::TestWithParam<GemmTestParams>       \
    {                                                                                         \
    };                                                                                        \
    TEST_P(Test_gemm_##KBASE##_4x1vl_rvv, Correctness)                                        \
    {                                                                                         \
        auto funcs = make_gemm_##KBASE##_funcs();                                             \
        RUNNER<TD, BIAS_T>(GetParam(), funcs);                                                \
    }                                                                                         \
    TEST_P(Test_gemm_##KBASE##_4x1vl_rvv, ClampSpecialValues)                                 \
    {                                                                                         \
        ScopedGemmClampProbe probe;                                                           \
        auto funcs = make_gemm_##KBASE##_funcs();                                             \
        RUNNER<TD, BIAS_T>(GetParam(), funcs);                                                \
    }                                                                                         \
    INSTANTIATE_TEST_SUITE_P(                                                                 \
        GemmTests, Test_gemm_##KBASE##_4x1vl_rvv,                                             \
        with_clamp_dimensions(common_w4a8_bq_test_values(),                                   \
                              tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv(),                      \
                              tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv(), 1, true),            \
        GemmTestParamNameGenerator())

// Packed test: kernel base + TD + runner + funcs typename.
#define W4A8_BQ_DEFINE_PACKED_TEST(KBASE, TD, RUNNER, FUNCS_TYPE, BIAS_T)                     \
    static FUNCS_TYPE<TD> make_gemm_##KBASE##_funcs()                                         \
    {                                                                                         \
        FUNCS_TYPE<TD> funcs;                                                                 \
        funcs.get_m_step = tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv;                           \
        funcs.get_n_step = tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv;                           \
        funcs.get_a_packed_offset = tqt_get_a_packed_offset_gemm_##KBASE##_4x1vl_rvv;         \
        funcs.get_b_packed_offset = tqt_get_b_packed_offset_gemm_##KBASE##_4x1vl_rvv;         \
        funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_##KBASE##_4x1vl_rvv;             \
        funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_##KBASE##_4x1vl_rvv;             \
        funcs.run_a_pack = tqt_run_a_pack_gemm_##KBASE##_4x1vl_rvv;                           \
        funcs.run_bt_pack = tqt_run_bt_pack_gemm_##KBASE##_4x1vl_rvv;                         \
        funcs.run_gemm = clamp_test_run<tqt_run_gemm_##KBASE##_4x1vl_rvv, TD, 14, 12, 13, 15, \
                                        ClampSource::kColumnBias, false, 6, true, BIAS_T>();  \
        return funcs;                                                                         \
    }                                                                                         \
    class Test_gemm_##KBASE##_4x1vl_rvv : public testing::TestWithParam<GemmTestParams>       \
    {                                                                                         \
    };                                                                                        \
    TEST_P(Test_gemm_##KBASE##_4x1vl_rvv, Correctness)                                        \
    {                                                                                         \
        auto funcs = make_gemm_##KBASE##_funcs();                                             \
        RUNNER<TD, BIAS_T>(GetParam(), funcs);                                                \
    }                                                                                         \
    TEST_P(Test_gemm_##KBASE##_4x1vl_rvv, ClampSpecialValues)                                 \
    {                                                                                         \
        ScopedGemmClampProbe probe;                                                           \
        auto funcs = make_gemm_##KBASE##_funcs();                                             \
        RUNNER<TD, BIAS_T>(GetParam(), funcs);                                                \
    }                                                                                         \
    INSTANTIATE_TEST_SUITE_P(                                                                 \
        GemmTests, Test_gemm_##KBASE##_4x1vl_rvv,                                             \
        with_clamp_dimensions(common_w4a8_bq_test_values(),                                   \
                              tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv(),                      \
                              tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv(), 1, true),            \
        GemmTestParamNameGenerator())
