//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Micro-kernel: element-wise SiLU (swish) activation for FP32
///
/// This is an RVV (1x4vl tile) implementation using LMUL=4 configuration.
/// One inner invocation walks a single row, so m_step = 1 and
/// n_step = 4 * csrr_vlenb() / sizeof(float).
/// Computes: c[i, j] = a[i, j] * sigmoid(a[i, j]), sigmoid(x) = 1 / (1 + exp(-x))
///
/// The kernel evaluates the equivalent form c = a / (1 + exp(-a)), which needs one
/// full-precision divide instead of a reciprocal estimate plus a correction step.
/// `exp` comes from the shared cephes polynomial in `src/common/tqt_math_rvv.h`,
/// which keeps several vector groups live; LMUL=4 is therefore the widest group
/// that still leaves the register allocator room, unlike the LMUL=8 used by ReLU.
///
/// Accuracy: the polynomial `exp` carries a relative error on the order of 1e-7
/// and the divide is correctly rounded, so the absolute error of the result grows
/// with |a|. Sweeping a over [-10, 10] measures a maximum absolute error of
/// 1.1e-6 (maximum relative error 1.4e-7); over the wider [-40, 40] it stays
/// below 1.3e-6. These are range-specific measurements, not local ULP guarantees.
///
/// Far into the negative tail that bound stops holding. `exp` saturates its
/// argument at 88.376 and therefore its result at 2.4e38, so the denominator
/// stops growing and the kernel returns a / 2.4e38 where the true value is
/// ~0 - an absolute error that grows linearly with |a|. It crosses 1.3e-6 near
/// a = -3.1e32, crosses the 5e-5 budget used by the tests near a = -1.2e34, and
/// reaches -1.41 at a = -FLT_MAX, outside SiLU's mathematical range of
/// [-0.2785, +inf). Non-finite inputs are not handled either: silu(NaN) = NaN
/// and silu(+Inf) = +Inf are as expected, but silu(-Inf) = -Inf where the limit
/// is 0.
///
/// When both matrices are fully packed (lda == n and ldc == n) the M x N
/// traversal collapses into one contiguous run of m * n elements.

/// Gets m step value (number of rows per inner invocation).
///
/// @return The m step value (1).
size_t tqt_get_m_step_silu_f32_f32_1x4vl_rvv(void);

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_silu_f32_f32_1x4vl_rvv(void);

/// Runs the element-wise SiLU micro-kernel.
///
/// @param[in]  m   Number of rows.
/// @param[in]  n   Number of columns.
/// @param[in]  a   Input matrix [m, n], row-major, contiguous float array.
/// @param[in]  lda Leading dimension of a (number of float elements per row, >= n).
/// @param[out] c   Output matrix [m, n], row-major, contiguous float array. May alias a.
/// @param[in]  ldc Leading dimension of c (number of float elements per row, >= n).
void tqt_run_silu_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                    size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
