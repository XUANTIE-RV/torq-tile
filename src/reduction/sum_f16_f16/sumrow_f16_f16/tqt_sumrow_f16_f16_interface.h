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

// Row-wise sum reduction, FP16.
//   a layout: [M, N] row-major, stride lda
//   c layout: [M] contiguous
//   c[m] = init[m] + sum_{n=0}^{N-1} a[m, n]
//   When init is nullptr, c[m] = sum_{n=0}^{N-1} a[m, n].

/// Returns the M-direction step (non-reduction dimension tile size).
typedef size_t (*tqt_sumrow_f16_f16_get_m_step_func_t)(void);

/// Returns the N-direction step (reduction dimension tile size).
typedef size_t (*tqt_sumrow_f16_f16_get_n_step_func_t)(void);

/// Runs the row-wise sum reduction.
typedef void (*tqt_sumrow_f16_f16_run_sumrow_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                     void *c, const void *init);

/// Micro-kernel interface.
struct tqt_sumrow_f16_f16_ukernel
{
    tqt_sumrow_f16_f16_get_m_step_func_t get_m_step;
    tqt_sumrow_f16_f16_get_n_step_func_t get_n_step;
    tqt_sumrow_f16_f16_run_sumrow_func_t run_sumrow;
};

#ifdef __cplusplus
}  // extern "C"
#endif
