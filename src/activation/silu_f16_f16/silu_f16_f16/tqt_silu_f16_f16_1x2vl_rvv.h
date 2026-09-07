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

/// Micro-kernel: element-wise SiLU (swish) activation for FP16
///
/// This is an RVV (1x2vl tile) implementation: one f16m2 group is widened into an
/// f32m4 group, so one inner invocation walks a single row with m_step = 1 and
/// n_step = 4 * csrr_vlenb() / sizeof(float), which is exactly the element count
/// of one f16m2 group.
/// Computes: c[i, j] = a[i, j] * sigmoid(a[i, j]), sigmoid(x) = 1 / (1 + exp(-x))
///
/// Every arithmetic step runs in FP32: the input is widened once, `exp`, the add
/// and the divide are evaluated at single precision, and the result is narrowed
/// once on store. FP16 has too few significand bits for the cephes polynomial, so
/// evaluating it natively would cost far more accuracy than the two conversions
/// cost throughput. The kernel evaluates the equivalent form
/// c = a / (1 + exp(-a)), which needs one full-precision divide instead of a
/// reciprocal estimate plus a correction step. `exp` comes from the shared
/// polynomial in `src/common/tqt_math_rvv.h`; its LMUL=4 instantiation is what
/// pins the FP16 load group to f16m2.
///
/// Accuracy: sweeping a over [-10, 10] measures a maximum absolute error of
/// 2.7e-3, which is exactly the error of rounding the mathematically exact result
/// to FP16 over the same range, so the polynomial contributes nothing measurable
/// at that scale. Checked exhaustively over all 63488 finite FP16 inputs, the
/// result is bit-identical to the correctly-rounded one for all but a single
/// input: a = 2^-24 returns +0 instead of the smallest subnormal, because the
/// FP32 intermediate 1 + exp(-a) rounds to exactly 2.0f and the quotient then
/// lands on a round-to-nearest-even tie. Relative error is unbounded in the
/// saturating tails, where the result flushes to +/-0.
///
/// The FP32 kernel's negative-tail saturation error cannot occur here: FP16 caps
/// |a| at 65504, so a / 2.4e38 always rounds to +/-0, which is correct. Non-finite
/// inputs are still not handled: silu(NaN) = NaN and silu(+Inf) = +Inf are as
/// expected, but silu(-Inf) = -Inf where the limit is 0.
///
/// When both matrices are fully packed (lda == n and ldc == n) the M x N
/// traversal collapses into one contiguous run of m * n elements.

/// Gets m step value (number of rows per inner invocation).
///
/// @return The m step value (1).
size_t tqt_get_m_step_silu_f16_f16_1x2vl_rvv(void);

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_silu_f16_f16_1x2vl_rvv(void);

/// Runs the element-wise SiLU micro-kernel.
///
/// @param[in]  m   Number of rows.
/// @param[in]  n   Number of columns.
/// @param[in]  a   Input matrix [m, n], row-major, contiguous float16 array.
/// @param[in]  lda Leading dimension of a (number of float16 elements per row, >= n).
/// @param[out] c   Output matrix [m, n], row-major, contiguous float16 array. May alias a.
/// @param[in]  ldc Leading dimension of c (number of float16 elements per row, >= n).
void tqt_run_silu_f16_f16_1x2vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                    size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
