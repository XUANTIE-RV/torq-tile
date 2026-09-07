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

// Element-wise SiLU (swish) activation, FP32.
//   a layout: [M, N] row-major, stride lda (>= n)
//   c layout: [M, N] row-major, stride ldc (>= n)
//   c[i, j] = a[i, j] * sigmoid(a[i, j]), sigmoid(x) = 1 / (1 + exp(-x))
//
// c may alias a: the operation is element-wise with no cross-element dependency,
// so an in-place call (c == a, ldc == lda) is well defined.
//
// The RVV implementation uses a cephes polynomial exp approximation. It is not
// correctly rounded, and the variant header states its measured error bound.
// The relative error is much larger in the saturating
// tails (where the true value is vanishingly small), so callers should judge
// accuracy by absolute error.

/// Returns the M-direction step (number of rows per inner invocation).
typedef size_t (*tqt_silu_f32_f32_get_m_step_func_t)(void);

/// Returns the N-direction step (number of float elements per inner step).
typedef size_t (*tqt_silu_f32_f32_get_n_step_func_t)(void);

/// Runs the element-wise SiLU.
typedef void (*tqt_silu_f32_f32_run_silu_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                 void *c, size_t ldc);

/// Micro-kernel interface.
struct tqt_silu_f32_f32_ukernel
{
    tqt_silu_f32_f32_get_m_step_func_t get_m_step;
    tqt_silu_f32_f32_get_n_step_func_t get_n_step;
    tqt_silu_f32_f32_run_silu_func_t run_silu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
