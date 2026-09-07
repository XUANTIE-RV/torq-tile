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

// Column-wise sum reduction, FP32.
//   a layout: [M, N] row-major, stride lda
//   c layout: [N] contiguous
//   c[n] = init[n] + sum_{m=0}^{M-1} a[m, n]
//   When init is nullptr, c[n] = sum_{m=0}^{M-1} a[m, n].

/// Returns the M-direction step (reduction dimension tile size).
typedef size_t (*tqt_sumcol_f32_f32_get_m_step_func_t)(void);

/// Returns the N-direction step (non-reduction dimension tile size).
typedef size_t (*tqt_sumcol_f32_f32_get_n_step_func_t)(void);

/// Runs the column-wise sum reduction.
typedef void (*tqt_sumcol_f32_f32_run_sumcol_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                     void *c, const void *init);

/// Micro-kernel interface.
struct tqt_sumcol_f32_f32_ukernel
{
    tqt_sumcol_f32_f32_get_m_step_func_t get_m_step;
    tqt_sumcol_f32_f32_get_n_step_func_t get_n_step;
    tqt_sumcol_f32_f32_run_sumcol_func_t run_sumcol;
};

#ifdef __cplusplus
}  // extern "C"
#endif
