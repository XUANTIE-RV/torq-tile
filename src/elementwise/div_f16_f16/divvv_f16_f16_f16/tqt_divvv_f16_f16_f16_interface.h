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

// Elementwise vector-vector divide, FP16.
//   a layout: [N] contiguous
//   b layout: [N] contiguous
//   c layout: [N] contiguous
//   c[i] = a[i] / b[i], i in [0, N)

/// Returns the processing width (number of float16_t elements per inner step).
typedef size_t (*tqt_divvv_f16_f16_f16_get_n_step_func_t)(void);

/// Runs the vector-vector divide.
typedef void (*tqt_divvv_f16_f16_f16_run_divvv_func_t)(size_t n, const void *a, const void *b,
                                                       void *c);

/// Micro-kernel interface.
struct tqt_divvv_f16_f16_f16_ukernel
{
    tqt_divvv_f16_f16_f16_get_n_step_func_t get_n_step;
    tqt_divvv_f16_f16_f16_run_divvv_func_t run_divvv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
