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

// Elementwise vector-vector add, FP32.
//   a layout: [N] contiguous
//   b layout: [N] contiguous
//   c layout: [N] contiguous
//   c[i] = a[i] + b[i], i in [0, N)

/// Returns the processing width (number of float elements per inner step).
typedef size_t (*tqt_addvv_f32_f32_f32_get_n_step_func_t)(void);

/// Runs the vector-vector add.
typedef void (*tqt_addvv_f32_f32_f32_run_addvv_func_t)(size_t n, const void *a, const void *b,
                                                       void *c);

/// Micro-kernel interface.
struct tqt_addvv_f32_f32_f32_ukernel
{
    tqt_addvv_f32_f32_f32_get_n_step_func_t get_n_step;
    tqt_addvv_f32_f32_f32_run_addvv_func_t run_addvv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
