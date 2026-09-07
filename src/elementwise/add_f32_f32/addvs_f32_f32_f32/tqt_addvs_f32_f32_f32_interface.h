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

// Elementwise vector-scalar add, FP32.
//   a layout: [N] contiguous
//   b:        scalar float
//   c layout: [N] contiguous
//   c[i] = a[i] + b, i in [0, N)

/// Returns the processing width (number of float elements per inner step).
typedef size_t (*tqt_addvs_f32_f32_f32_get_n_step_func_t)(void);

/// Runs the vector-scalar add.
typedef void (*tqt_addvs_f32_f32_f32_run_addvs_func_t)(size_t n, const void *a, float b, void *c);

/// Micro-kernel interface.
struct tqt_addvs_f32_f32_f32_ukernel
{
    tqt_addvs_f32_f32_f32_get_n_step_func_t get_n_step;
    tqt_addvs_f32_f32_f32_run_addvs_func_t run_addvs;
};

#ifdef __cplusplus
}  // extern "C"
#endif
