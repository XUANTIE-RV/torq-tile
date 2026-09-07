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

// Elementwise vector-scalar divide, FP32.
//   a layout: [N] contiguous
//   b:        scalar float
//   c layout: [N] contiguous
//   c[i] = a[i] / b, i in [0, N)

/// Returns the processing width (number of float elements per inner step).
typedef size_t (*tqt_divvs_f32_f32_f32_get_n_step_func_t)(void);

/// Runs the vector-scalar divide.
typedef void (*tqt_divvs_f32_f32_f32_run_divvs_func_t)(size_t n, const void *a, float b, void *c);

/// Micro-kernel interface.
struct tqt_divvs_f32_f32_f32_ukernel
{
    tqt_divvs_f32_f32_f32_get_n_step_func_t get_n_step;
    tqt_divvs_f32_f32_f32_run_divvs_func_t run_divvs;
};

#ifdef __cplusplus
}  // extern "C"
#endif
