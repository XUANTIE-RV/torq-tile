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
/// The SiLU term is evaluated as a / (1 + exp(-a)) using the shared FMA-based
/// exp in src/common/tqt_math_rvv.h and a full-precision divide. RNE rounding
/// is required; the kernel does not change frm. The existing LMUL=4 FP32
/// compute group is retained.
/// The final multiply by b also runs in FP32. Absolute error includes the SiLU
/// error scaled by |b|, the multiply rounding, and any final FP16 conversion.
///
/// The exp helper preserves subnormal outputs and overflows to +Inf. For a
/// sufficiently negative finite gate, the denominator overflows and SiLU returns
/// -0; the result no longer grows with |a| in that tail. NaNs propagate,
/// silu(+Inf) = +Inf, silu(-Inf) = NaN, and input signed zeros are preserved.
/// The -Inf result follows the direct division used by native PyTorch/GGML SiLU.
/// The gate is multiplied by b using ordinary floating-point semantics:
/// a NaN gate, zero times infinity, or a NaN linear operand produces NaN.
/// NaN payloads and floating-point exception flags are not specified.
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
