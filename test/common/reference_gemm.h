//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "src/common/tqt_common.h"
#include "test/common/test_common.h"

namespace torq_tile
{
namespace test
{

/// Reference GEMM implementation with FP32 accumulation.
///
/// Computes: D = clamp(A * B + C + bias, clamp_min, clamp_max)
/// where A, B are TA and C, D, bias are TD. For same-type GEMM use TA == TD.
///
/// @param m       Number of rows in A and D
/// @param n       Number of columns in D
/// @param k       Common dimension (columns of A, rows/columns of B depending on layout)
/// @param A       Input matrix A [m, k] TA, row-major, lda >= k
/// @param lda     Leading dimension of A
/// @param B       Input matrix B TA, layout depends on b_layout parameter
/// @param ldb     Leading dimension of B
/// @param b_layout  BLayout::kRowMajor: B is [K, N]; BLayout::kTransposed: B is [N, K]
/// @param C       Residual matrix C [m, n] TD (can be nullptr)
/// @param ldc     Leading dimension of C
/// @param D       Output matrix D [m, n] TD
/// @param ldd     Leading dimension of D
/// @param bias    Bias vector TD (can be nullptr)
/// @param bias_mode  BiasMode::kNone, k1xN (bias[1,N]), or kMx1 (bias[M,1])
/// @param clamp_min  Minimum clamp value (as TD)
/// @param clamp_max  Maximum clamp value (as TD)
template <typename TA, typename TD>
void reference_gemm(size_t m, size_t n, size_t k, const TA *A, size_t lda, const TA *B, size_t ldb,
                    BLayout b_layout, const TD *C, size_t ldc, TD *D, size_t ldd, const TD *bias,
                    BiasMode bias_mode, TD clamp_min, TD clamp_max);

// Explicit instantiation declarations
extern template void reference_gemm<float, float>(size_t, size_t, size_t, const float *, size_t,
                                                  const float *, size_t, BLayout, const float *,
                                                  size_t, float *, size_t, const float *, BiasMode,
                                                  float, float);
extern template void reference_gemm<float16_t, float16_t>(size_t, size_t, size_t, const float16_t *,
                                                          size_t, const float16_t *, size_t,
                                                          BLayout, const float16_t *, size_t,
                                                          float16_t *, size_t, const float16_t *,
                                                          BiasMode, float16_t, float16_t);
extern template void reference_gemm<bfloat16_t, bfloat16_t>(
    size_t, size_t, size_t, const bfloat16_t *, size_t, const bfloat16_t *, size_t, BLayout,
    const bfloat16_t *, size_t, bfloat16_t *, size_t, const bfloat16_t *, BiasMode, bfloat16_t,
    bfloat16_t);
extern template void reference_gemm<bfloat16_t, float>(size_t, size_t, size_t, const bfloat16_t *,
                                                       size_t, const bfloat16_t *, size_t, BLayout,
                                                       const float *, size_t, float *, size_t,
                                                       const float *, BiasMode, float, float);
extern template void reference_gemm<bfloat16_t, float16_t>(size_t, size_t, size_t,
                                                           const bfloat16_t *, size_t,
                                                           const bfloat16_t *, size_t, BLayout,
                                                           const float16_t *, size_t, float16_t *,
                                                           size_t, const float16_t *, BiasMode,
                                                           float16_t, float16_t);

/// Reference GEMM: int8 inputs, int32 output with clamp
///
/// Computes: D = clamp(A * B + C, clamp_min, clamp_max)
///
/// @param m         Number of rows in A and D
/// @param n         Number of columns in D
/// @param k         Common dimension (columns of A, rows/columns of B)
/// @param A         Input matrix A [m, k], row-major, int8_t
/// @param lda       Leading dimension of A
/// @param B         Input matrix B, layout depends on b_layout parameter, int8_t
/// @param ldb       Leading dimension of B
/// @param b_layout  BLayout::kRowMajor: B is [K, N]; BLayout::kTransposed: B is [N, K]
/// @param C         Residual matrix C [m, n], int32_t (can be nullptr)
/// @param ldc       Leading dimension of C
/// @param D         Output matrix D [m, n], int32_t
/// @param ldd       Leading dimension of D
/// @param clamp_min Minimum clamp value
/// @param clamp_max Maximum clamp value
void reference_gemm_i32_i8(size_t m, size_t n, size_t k, const int8_t *A, size_t lda,
                           const int8_t *B, size_t ldb, BLayout b_layout, const int32_t *C,
                           size_t ldc, int32_t *D, size_t ldd, int32_t clamp_min,
                           int32_t clamp_max);

/// Reference GEMM with quantized weight (int4) and mixed input/output precision.
///
/// Computes: D_TD = clamp(A_TA * dequant(B_i4, scale_b_TA) + C_TD + bias_TD, min, max)
/// where activations / scales are TA (e.g. f16) and the residual / bias / output are TD
/// (e.g. f32). Internally accumulates in float and casts to TD at the store boundary.
///
/// @param m         Number of rows in A and D
/// @param n         Number of columns in D
/// @param k         Common dimension, must be a multiple of bl
/// @param bl        Block length for quantization scale, >= 2 and multiple of 2
/// @param A         Input matrix A [m, k] TA, row-major
/// @param lda       Leading dimension of A (TA elements)
/// @param B         Input matrix B [n, k] int4, K-contiguous, ldb in int4 elements
/// @param ldb       Leading dimension of B (int4 elements, = k)
/// @param scale_b   Scale matrix [n, k/bl] TA, row-major
/// @param ldsb      Leading dimension of scale_b (TA elements, = k/bl)
/// @param C         Residual matrix C [m, n] TD (can be nullptr)
/// @param ldc       Leading dimension of C (TD elements)
/// @param D         Output matrix D [m, n] TD
/// @param ldd       Leading dimension of D (TD elements)
/// @param bias      Bias vector TD (can be nullptr)
/// @param bias_mode BiasMode::kNone or k1xN
/// @param clamp_min Minimum clamp value (TD)
/// @param clamp_max Maximum clamp value (TD)
template <typename TA, typename TD>
void reference_gemm_quantized_weight_mixed(size_t m, size_t n, size_t k, size_t bl, const TA *A,
                                           size_t lda, const uint8_t *B, size_t ldb,
                                           const TA *scale_b, size_t ldsb, const TD *C, size_t ldc,
                                           TD *D, size_t ldd, const TD *bias, BiasMode bias_mode,
                                           TD clamp_min, TD clamp_max);

extern template void reference_gemm_quantized_weight_mixed<float16_t, float16_t>(
    size_t, size_t, size_t, size_t, const float16_t *, size_t, const uint8_t *, size_t,
    const float16_t *, size_t, const float16_t *, size_t, float16_t *, size_t, const float16_t *,
    BiasMode, float16_t, float16_t);
extern template void reference_gemm_quantized_weight_mixed<float16_t, float>(
    size_t, size_t, size_t, size_t, const float16_t *, size_t, const uint8_t *, size_t,
    const float16_t *, size_t, const float *, size_t, float *, size_t, const float *, BiasMode,
    float, float);

}  // namespace test
}  // namespace torq_tile
