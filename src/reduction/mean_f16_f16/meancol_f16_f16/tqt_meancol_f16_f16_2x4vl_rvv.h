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

/// Micro-kernel: column-wise mean reduction for FP16
///
/// This is an RVV (2x4vl tile) implementation using an FP32 widening accumulator
/// for precision: FP16 inputs are loaded as LMUL=2 and widened to FP32 (LMUL=4)
/// during accumulation; the mean and init bias are applied in FP32 and each column
/// is narrowed back to FP16 on store.
/// Covers 2 f16m2 column groups and unrolls the M loop by 2 with a pair of FP32
/// partial accumulators per group, breaking the M-direction dependency chain; the
/// partials are combined before the 1/M scaling.
/// m_step = 2 (M-direction unroll), n_step = 2 * 4 * vlmax.
/// Computes: c[n] = init[n] + (1/M) * sum_{m=0}^{M-1} a[m, n]

/// Gets m step value (reduction-direction unroll factor).
///
/// @return The m step value (2).
size_t tqt_get_m_step_meancol_f16_f16_2x4vl_rvv(void);

/// Gets n step value (non-reduction dimension tile size).
///
/// @return The n step value (2 * 4 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_meancol_f16_f16_2x4vl_rvv(void);

/// Runs the column-wise mean reduction micro-kernel.
///
/// @param[in]  m    Number of rows.
/// @param[in]  n    Number of columns.
/// @param[in]  a    Input matrix [m, n], row-major, contiguous float16_t array.
/// @param[in]  lda  Leading dimension of a (number of float16_t elements per row).
/// @param[out] c    Output vector [n], contiguous float16_t array.
/// @param[in]  init Initial values [n], or nullptr for zero initialization.
void tqt_run_meancol_f16_f16_2x4vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                       const void *init);

#ifdef __cplusplus
}  // extern "C"
#endif
