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

// Element-wise ReLU activation, FP32.
//   a layout: [M, N] row-major, stride lda (>= n)
//   c layout: [M, N] row-major, stride ldc (>= n)
//   c[i, j] = max(a[i, j], 0), i in [0, M), j in [0, N)
//
// c may alias a: the operation is element-wise with no cross-element dependency,
// so an in-place call (c == a, ldc == lda) is well defined.
//
// NaN inputs follow the hardware maximum instruction, which returns the non-NaN
// operand, so relu(NaN) yields 0 rather than propagating NaN.

/// Returns the M-direction step (number of rows per inner invocation).
typedef size_t (*tqt_relu_f32_f32_get_m_step_func_t)(void);

/// Returns the N-direction step (number of float elements per inner step).
typedef size_t (*tqt_relu_f32_f32_get_n_step_func_t)(void);

/// Runs the element-wise ReLU.
typedef void (*tqt_relu_f32_f32_run_relu_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                 void *c, size_t ldc);

/// Micro-kernel interface.
struct tqt_relu_f32_f32_ukernel
{
    tqt_relu_f32_f32_get_m_step_func_t get_m_step;
    tqt_relu_f32_f32_get_n_step_func_t get_n_step;
    tqt_relu_f32_f32_run_relu_func_t run_relu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
