//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/test_common.h"

namespace torq_tile
{
namespace test
{

/// Reference SGEMM implementation with FP32 accumulation.
///
/// Computes: D = alpha * op(A) * op(B) + beta * C
/// where op(X) is X or X^T depending on layout parameters.
///
/// @param m         Number of rows of op(A) and D
/// @param n         Number of columns of op(B) and D
/// @param k         Common dimension
/// @param A         Input matrix A, layout depends on a_layout
/// @param lda       Leading dimension of A
/// @param a_layout  ALayout::kRowMajor: A is [M, K]; ALayout::kTransposed: A is [K, M]
/// @param B         Input matrix B, layout depends on b_layout
/// @param ldb       Leading dimension of B
/// @param b_layout  BLayout::kRowMajor: B is [K, N]; BLayout::kTransposed: B is [N, K]
/// @param C         Residual matrix C [m, n] (can be nullptr when beta == 0)
/// @param ldc       Leading dimension of C
/// @param D         Output matrix D [m, n]
/// @param ldd       Leading dimension of D
/// @param alpha     Scalar multiplier for A*B
/// @param beta      Scalar multiplier for C
template <typename TA, typename TD>
void reference_sgemm(size_t m, size_t n, size_t k, const TA *A, size_t lda, ALayout a_layout,
                     const TA *B, size_t ldb, BLayout b_layout, const TD *C, size_t ldc, TD *D,
                     size_t ldd, TD alpha, TD beta);

// Explicit instantiation declarations
extern template void reference_sgemm<float, float>(size_t, size_t, size_t, const float *, size_t,
                                                   ALayout, const float *, size_t, BLayout,
                                                   const float *, size_t, float *, size_t, float,
                                                   float);

}  // namespace test
}  // namespace torq_tile
