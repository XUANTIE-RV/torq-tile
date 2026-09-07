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

/// Micro-kernel: element-wise SwiGLU (gated SiLU) activation for FP16
///
/// This is an RVV (1x2vl tile) implementation: one `f16m2` group per inner step,
/// widened into the `f32m4` group the shared polynomial `exp` is instantiated for.
/// One inner invocation walks a single row, so m_step = 1 and
/// n_step = 4 * csrr_vlenb() / sizeof(float), the same width as the FP32 variant.
/// Computes: c[i, j] = silu(a[i, j]) * b[i, j], silu(x) = x * sigmoid(x)
///
/// `a` is the gate operand (the one SiLU is applied to) and `b` the linear
/// operand. Both carry their own leading dimension, so a fused [M, 2N]
/// gate-and-up buffer is addressed without repacking: pass a = base, lda = 2 * n
/// and b = base + n, ldb = 2 * n.
///
/// Both operands are widened to FP32 once, and every arithmetic step - `exp`, the
/// add, the divide and the final multiply - runs at single precision before a
/// single narrowing store. `exp` comes from the shared cephes polynomial in
/// `src/common/tqt_math_rvv.h`; it keeps several vector groups live, so LMUL=4 on
/// the FP32 side (an `f16m2` load group) is the widest tile that leaves the
/// register allocator room.
///
/// Accuracy: the FP32 intermediate error is far below FP16 resolution, so the
/// dominant term is the single rounding of the result to FP16. Sweeping a and b
/// over [-10, 10] measures a maximum absolute error of 7.8e-3.
///
/// Far into the negative tail of `a` the polynomial `exp` saturates and the SiLU
/// term stops approaching 0, but for FP16 inputs the saturation point (-88.376)
/// is well inside the representable range and the resulting error stays below
/// FP16 resolution. Non-finite inputs are not handled: silu(-Inf) = -Inf where the
/// limit is 0.
///
/// When all three matrices are fully packed (lda == ldb == ldc == n) the M x N
/// traversal collapses into one contiguous run of m * n elements.

/// Gets m step value (number of rows per inner invocation).
///
/// @return The m step value (1).
size_t tqt_get_m_step_swiglu_f16_f16_f16_1x2vl_rvv(void);

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_swiglu_f16_f16_f16_1x2vl_rvv(void);

/// Runs the element-wise SwiGLU micro-kernel.
///
/// @param[in]  m   Number of rows.
/// @param[in]  n   Number of columns.
/// @param[in]  a   Gate matrix [m, n], row-major, contiguous float16_t array.
/// @param[in]  lda Leading dimension of a (number of float16_t elements per row, >= n).
/// @param[in]  b   Linear matrix [m, n], row-major, contiguous float16_t array.
/// @param[in]  ldb Leading dimension of b (number of float16_t elements per row, >= n).
/// @param[out] c   Output matrix [m, n], row-major, contiguous float16_t array. May alias a or b.
/// @param[in]  ldc Leading dimension of c (number of float16_t elements per row, >= n).
void tqt_run_swiglu_f16_f16_f16_1x2vl_rvv(size_t m, size_t n, const void *a, size_t lda,
                                          const void *b, size_t ldb, void *c, size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
