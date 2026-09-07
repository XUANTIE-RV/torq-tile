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

/// Micro-kernel: column-wise sum reduction for FP32
///
/// This is an RVV (2x8vl tile) implementation using LMUL=4 configuration.
/// Covers n_step = 2 * 4 * vlmax columns (2 LMUL=4 groups) and unrolls the M loop
/// by 2 with a pair of partial accumulators per group, breaking the M-direction
/// dependency chain; the partials are combined before store.
/// m_step = 2 (M-direction unroll), n_step = 2 * 4 * vlmax.
/// Computes: c[n] = init[n] + sum_{m=0}^{M-1} a[m, n]

/// Gets m step value (reduction-direction unroll factor).
///
/// @return The m step value (2).
size_t tqt_get_m_step_sumcol_f32_f32_2x8vl_rvv(void);

/// Gets n step value (non-reduction dimension tile size).
///
/// @return The n step value (2 * 4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_sumcol_f32_f32_2x8vl_rvv(void);

/// Runs the column-wise sum reduction micro-kernel.
///
/// @param[in]  m    Number of rows.
/// @param[in]  n    Number of columns.
/// @param[in]  a    Input matrix [m, n], row-major, contiguous float array.
/// @param[in]  lda  Leading dimension of a (number of float elements per row).
/// @param[out] c    Output vector [n], contiguous float array.
/// @param[in]  init Initial values [n], or nullptr for zero initialization.
void tqt_run_sumcol_f32_f32_2x8vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                      const void *init);

#ifdef __cplusplus
}  // extern "C"
#endif
