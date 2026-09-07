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

/// Micro-kernel: element-wise SwiGLU (gated SiLU) activation for FP32
///
/// This is an RVV (1x4vl tile) implementation using LMUL=4 configuration.
/// One inner invocation walks a single row, so m_step = 1 and
/// n_step = 4 * csrr_vlenb() / sizeof(float).
/// Computes: c[i, j] = silu(a[i, j]) * b[i, j], silu(x) = x * sigmoid(x)
///
/// `a` is the gate operand (the one SiLU is applied to) and `b` the linear
/// operand. Both carry their own leading dimension, so a fused [M, 2N]
/// gate-and-up buffer is addressed without repacking: pass a = base, lda = 2 * n
/// and b = base + n, ldb = 2 * n.
///
/// The SiLU part is evaluated as a / (1 + exp(-a)), which needs one full-precision
/// divide instead of a reciprocal estimate plus a correction step, and the result
/// is then multiplied by b. `exp` comes from the shared cephes polynomial in
/// `src/common/tqt_math_rvv.h`, which keeps several vector groups live; LMUL=4 is
/// therefore the widest group that still leaves the register allocator room,
/// unlike the LMUL=8 used by ReLU.
///
/// Accuracy: the polynomial `exp` carries a relative error on the order of 1e-7
/// and both the divide and the multiply are correctly rounded, so the absolute
/// error of the result is the SiLU error scaled by |b|. Sweeping a and b over
/// [-10, 10] measures a maximum absolute error of 1.5e-5 (the SiLU term
/// contributes up to 1.1e-6 per unit of |b|). The bound therefore grows with the
/// magnitude of the linear operand, not just with |a|.
///
/// Far into the negative tail of `a` that bound stops holding. `exp` saturates its
/// argument at 88.376 and therefore its result at 2.4e38, so the denominator stops
/// growing and the SiLU term returns a / 2.4e38 where the true value is ~0 - an
/// absolute error that grows linearly with |a| and is then scaled by |b|.
/// Non-finite inputs are not handled either: silu(NaN) = NaN and silu(+Inf) = +Inf
/// are as expected, but silu(-Inf) = -Inf where the limit is 0.
///
/// When all three matrices are fully packed (lda == ldb == ldc == n) the M x N
/// traversal collapses into one contiguous run of m * n elements.

/// Gets m step value (number of rows per inner invocation).
///
/// @return The m step value (1).
size_t tqt_get_m_step_swiglu_f32_f32_f32_1x4vl_rvv(void);

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_swiglu_f32_f32_f32_1x4vl_rvv(void);

/// Runs the element-wise SwiGLU micro-kernel.
///
/// @param[in]  m   Number of rows.
/// @param[in]  n   Number of columns.
/// @param[in]  a   Gate matrix [m, n], row-major, contiguous float array.
/// @param[in]  lda Leading dimension of a (number of float elements per row, >= n).
/// @param[in]  b   Linear matrix [m, n], row-major, contiguous float array.
/// @param[in]  ldb Leading dimension of b (number of float elements per row, >= n).
/// @param[out] c   Output matrix [m, n], row-major, contiguous float array. May alias a or b.
/// @param[in]  ldc Leading dimension of c (number of float elements per row, >= n).
void tqt_run_swiglu_f32_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a, size_t lda,
                                          const void *b, size_t ldb, void *c, size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
