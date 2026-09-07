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

/// Micro-kernel: row-wise sum reduction for FP32
///
/// This is an RVV (4x4vl tile) implementation using LMUL=4 configuration.
/// Processes up to 4 rows simultaneously, each with an independent accumulator
/// so the reduction chains issue without cross-row data dependencies.
/// m_step = 4, n_step = 4 * csrr_vlenb() / sizeof(float).
/// Computes: c[m] = init[m] + sum_{n=0}^{N-1} a[m, n]

/// Gets m step value (non-reduction dimension tile size).
///
/// @return The m step value (4).
size_t tqt_get_m_step_sumrow_f32_f32_4x4vl_rvv(void);

/// Gets n step value (reduction dimension tile size).
///
/// @return The n step value (4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_sumrow_f32_f32_4x4vl_rvv(void);

/// Runs the row-wise sum reduction micro-kernel.
///
/// @param[in]  m    Number of rows.
/// @param[in]  n    Number of columns.
/// @param[in]  a    Input matrix [m, n], row-major, contiguous float array.
/// @param[in]  lda  Leading dimension of a (number of float elements per row).
/// @param[out] c    Output vector [m], contiguous float array.
/// @param[in]  init Initial values [m], or nullptr for zero initialization.
void tqt_run_sumrow_f32_f32_4x4vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                      const void *init);

#ifdef __cplusplus
}  // extern "C"
#endif
