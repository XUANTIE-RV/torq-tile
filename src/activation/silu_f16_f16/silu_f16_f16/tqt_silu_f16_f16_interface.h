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

// Element-wise SiLU (swish) activation, FP16.
//   a layout: [M, N] row-major, stride lda (>= n)
//   c layout: [M, N] row-major, stride ldc (>= n)
//   c[i, j] = a[i, j] * sigmoid(a[i, j]), sigmoid(x) = 1 / (1 + exp(-x))
//
// c may alias a: the operation is element-wise with no cross-element dependency,
// so an in-place call (c == a, ldc == lda) is well defined.
//
// RVV widens to FP32, evaluates the sigmoid with a cephes polynomial exp,
// and narrows back. The approximation is not correctly rounded, and each
// variant header states its measured absolute error over
// x in [-10, 10]. Those are range-specific measurements, not local ULP
// guarantees; relative error is unbounded in the saturating tails.

/// Returns the M-direction step (number of rows per inner invocation).
typedef size_t (*tqt_silu_f16_f16_get_m_step_func_t)(void);

/// Returns the N-direction step (number of float16 elements per inner step).
typedef size_t (*tqt_silu_f16_f16_get_n_step_func_t)(void);

/// Runs the element-wise SiLU.
typedef void (*tqt_silu_f16_f16_run_silu_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                 void *c, size_t ldc);

/// Micro-kernel interface.
struct tqt_silu_f16_f16_ukernel
{
    tqt_silu_f16_f16_get_m_step_func_t get_m_step;
    tqt_silu_f16_f16_get_n_step_func_t get_n_step;
    tqt_silu_f16_f16_run_silu_func_t run_silu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
