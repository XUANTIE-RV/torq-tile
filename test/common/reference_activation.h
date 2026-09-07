//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

// Activation reference kernels over a [m, n] row-major matrix with independent
// input and output leading dimensions. The transcendental part is evaluated in
// double and only the final result is cast back to T, so the reference carries no
// error of its own beyond rounding to T. The same definition serves float,
// float16_t and bfloat16_t; explicit instantiations live in the .cpp.

/// Reference element-wise ReLU: c[i, j] = max(a[i, j], 0).
///
/// @param m   Number of rows
/// @param n   Number of columns
/// @param a   Input matrix A [m, n], row-major
/// @param lda Leading dimension of A (>= n)
/// @param c   Output matrix C [m, n], row-major
/// @param ldc Leading dimension of C (>= n)
template <typename T>
void reference_relu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc);

/// Reference element-wise SiLU (swish): c[i, j] = a[i, j] * sigmoid(a[i, j]),
/// where sigmoid(x) = 1 / (1 + exp(-x)).
///
/// @param m   Number of rows
/// @param n   Number of columns
/// @param a   Input matrix A [m, n], row-major
/// @param lda Leading dimension of A (>= n)
/// @param c   Output matrix C [m, n], row-major
/// @param ldc Leading dimension of C (>= n)
template <typename T>
void reference_silu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc);

/// Reference element-wise GeLU using the standard tanh approximation:
/// c = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
template <typename T>
void reference_gelu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc);

/// Reference element-wise ReGLU:
/// c[i, j] = max(a[i, j], 0) * b[i, j].
template <typename T>
void reference_reglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                     size_t ldc);

/// Reference element-wise GeGLU using the standard tanh-approximate GeLU:
/// c[i, j] = gelu(a[i, j]) * b[i, j].
template <typename T>
void reference_geglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                     size_t ldc);

/// Reference element-wise SwiGLU (gated SiLU):
/// c[i, j] = silu(a[i, j]) * b[i, j], where silu(x) = x * sigmoid(x).
///
/// @param m   Number of rows
/// @param n   Number of columns
/// @param a   Gate matrix A [m, n], row-major
/// @param lda Leading dimension of A (>= n)
/// @param b   Linear matrix B [m, n], row-major
/// @param ldb Leading dimension of B (>= n)
/// @param c   Output matrix C [m, n], row-major
/// @param ldc Leading dimension of C (>= n)
template <typename T>
void reference_swiglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                      size_t ldc);

}  // namespace test
}  // namespace torq_tile
