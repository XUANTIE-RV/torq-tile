//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "src/common/tqt_common.h"
#include "test/common/test_common.h"

namespace torq_tile
{
namespace test
{

/// Reference GEMM implementation with FP32 accumulation.
///
/// Computes: D = clamp(A * B + C + bias, clamp_min, clamp_max)
///
/// @param m       Number of rows in A and D
/// @param n       Number of columns in D
/// @param k       Common dimension (columns of A, rows/columns of B depending on layout)
/// @param A       Input matrix A [m, k], row-major, lda >= k
/// @param lda     Leading dimension of A
/// @param B       Input matrix B, layout depends on b_layout parameter
/// @param ldb     Leading dimension of B
/// @param b_layout  BLayout::kRowMajor: B is [K, N]; BLayout::kTransposed: B is [N, K]
/// @param C       Residual matrix C [m, n] (can be nullptr)
/// @param ldc     Leading dimension of C
/// @param D       Output matrix D [m, n]
/// @param ldd     Leading dimension of D
/// @param bias    Bias vector (can be nullptr)
/// @param bias_mode  BiasMode::kNone, k1xN (bias[1,N]), or kMx1 (bias[M,1])
/// @param clamp_min  Minimum clamp value (as T)
/// @param clamp_max  Maximum clamp value (as T)
template <typename T>
void reference_gemm(size_t m, size_t n, size_t k, const T *A, size_t lda, const T *B, size_t ldb,
                    BLayout b_layout, const T *C, size_t ldc, T *D, size_t ldd, const T *bias,
                    BiasMode bias_mode, T clamp_min, T clamp_max);

// Explicit instantiation declarations
extern template void reference_gemm<float>(size_t, size_t, size_t, const float *, size_t,
                                           const float *, size_t, BLayout, const float *, size_t,
                                           float *, size_t, const float *, BiasMode, float, float);
extern template void reference_gemm<float16_t>(size_t, size_t, size_t, const float16_t *, size_t,
                                               const float16_t *, size_t, BLayout,
                                               const float16_t *, size_t, float16_t *, size_t,
                                               const float16_t *, BiasMode, float16_t, float16_t);
extern template void reference_gemm<bfloat16_t>(size_t, size_t, size_t, const bfloat16_t *, size_t,
                                                const bfloat16_t *, size_t, BLayout,
                                                const bfloat16_t *, size_t, bfloat16_t *, size_t,
                                                const bfloat16_t *, BiasMode, bfloat16_t,
                                                bfloat16_t);

}  // namespace test
}  // namespace torq_tile
