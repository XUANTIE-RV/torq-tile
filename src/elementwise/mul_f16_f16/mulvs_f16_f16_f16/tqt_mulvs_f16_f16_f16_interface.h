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

// Elementwise vector-scalar multiply, FP16.
//   a layout: [N] contiguous
//   b:        scalar float16_t
//   c layout: [N] contiguous
//   c[i] = a[i] * b, i in [0, N)

/// Returns the processing width (number of float16_t elements per inner step).
typedef size_t (*tqt_mulvs_f16_f16_f16_get_n_step_func_t)(void);

/// Runs the vector-scalar multiply.
typedef void (*tqt_mulvs_f16_f16_f16_run_mulvs_func_t)(size_t n, const void *a, float16_t b,
                                                       void *c);

/// Micro-kernel interface.
struct tqt_mulvs_f16_f16_f16_ukernel
{
    tqt_mulvs_f16_f16_f16_get_n_step_func_t get_n_step;
    tqt_mulvs_f16_f16_f16_run_mulvs_func_t run_mulvs;
};

#ifdef __cplusplus
}  // extern "C"
#endif
