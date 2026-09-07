//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise SwiGLU (gated SiLU) activation, FP32.
//   a layout: [M, N] row-major, stride lda (>= n) - the gate operand
//   b layout: [M, N] row-major, stride ldb (>= n) - the linear operand
//   c layout: [M, N] row-major, stride ldc (>= n)
//   c[i, j] = silu(a[i, j]) * b[i, j], silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// The two operands carry independent leading dimensions, so the fused
// gate-and-up layout used by transformer feed-forward blocks - a single
// [M, 2N] buffer whose left half is the gate and whose right half is the
// linear projection - needs no repacking:
//   a = base,     lda = 2 * n
//   b = base + n, ldb = 2 * n
//
// c may alias a and/or b: every element of both operands is read before the
// matching output element is written, and there is no cross-element dependency,
// so an in-place call (c == a with ldc == lda, or c == b with ldc == ldb) is
// well defined.
//
// The sigmoid is evaluated with a cephes polynomial exp and is not correctly
// rounded. The variant header states its measured error bound. The absolute error
// of the result is the SiLU error scaled by |b|, so it grows with the magnitude of
// the linear operand; callers should judge accuracy by absolute error over their
// own input range.

/// Returns the M-direction step (number of rows per inner invocation).
typedef size_t (*tqt_swiglu_f32_f32_f32_get_m_step_func_t)(void);

/// Returns the N-direction step (number of float elements per inner step).
typedef size_t (*tqt_swiglu_f32_f32_f32_get_n_step_func_t)(void);

/// Runs the element-wise SwiGLU.
typedef void (*tqt_swiglu_f32_f32_f32_run_swiglu_func_t)(size_t m, size_t n, const void *a,
                                                         size_t lda, const void *b, size_t ldb,
                                                         void *c, size_t ldc);

/// Micro-kernel interface.
struct tqt_swiglu_f32_f32_f32_ukernel
{
    tqt_swiglu_f32_f32_f32_get_m_step_func_t get_m_step;
    tqt_swiglu_f32_f32_f32_get_n_step_func_t get_n_step;
    tqt_swiglu_f32_f32_f32_run_swiglu_func_t run_swiglu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
