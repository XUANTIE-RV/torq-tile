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

/// Micro-kernel: element-wise ReLU activation for FP32
///
/// This is an RVV (1x8vl tile) implementation using LMUL=8 configuration.
/// One inner invocation walks a single row, so m_step = 1 and
/// n_step = 8 * csrr_vlenb() / sizeof(float). ReLU needs only one live vector
/// register group, so the widest LMUL is used to maximise load/store bandwidth.
/// Computes: c[i, j] = max(a[i, j], 0)
///
/// When both matrices are fully packed (lda == n and ldc == n) the M x N
/// traversal collapses into one contiguous run of m * n elements.

/// Gets m step value (number of rows per inner invocation).
///
/// @return The m step value (1).
size_t tqt_get_m_step_relu_f32_f32_1x8vl_rvv(void);

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (8 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_relu_f32_f32_1x8vl_rvv(void);

/// Runs the element-wise ReLU micro-kernel.
///
/// @param[in]  m   Number of rows.
/// @param[in]  n   Number of columns.
/// @param[in]  a   Input matrix [m, n], row-major, contiguous float array.
/// @param[in]  lda Leading dimension of a (number of float elements per row, >= n).
/// @param[out] c   Output matrix [m, n], row-major, contiguous float array. May alias a.
/// @param[in]  ldc Leading dimension of c (number of float elements per row, >= n).
void tqt_run_relu_f32_f32_1x8vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                    size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
