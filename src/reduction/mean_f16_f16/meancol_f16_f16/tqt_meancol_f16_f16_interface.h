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

// Column-wise mean reduction, FP16.
//   a layout: [M, N] row-major, stride lda
//   c layout: [N] contiguous
//   c[n] = init[n] + (1/M) * sum_{m=0}^{M-1} a[m, n]
//   When init is nullptr, c[n] = (1/M) * sum_{m=0}^{M-1} a[m, n].

/// Returns the M-direction step (reduction dimension tile size).
typedef size_t (*tqt_meancol_f16_f16_get_m_step_func_t)(void);

/// Returns the N-direction step (non-reduction dimension tile size).
typedef size_t (*tqt_meancol_f16_f16_get_n_step_func_t)(void);

/// Runs the column-wise mean reduction.
typedef void (*tqt_meancol_f16_f16_run_meancol_func_t)(size_t m, size_t n, const void *a,
                                                       size_t lda, void *c, const void *init);

/// Micro-kernel interface.
struct tqt_meancol_f16_f16_ukernel
{
    tqt_meancol_f16_f16_get_m_step_func_t get_m_step;
    tqt_meancol_f16_f16_get_n_step_func_t get_n_step;
    tqt_meancol_f16_f16_run_meancol_func_t run_meancol;
};

#ifdef __cplusplus
}  // extern "C"
#endif
